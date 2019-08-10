import copy
import numpy as np
import torch
import torch.nn as nn

import util

from models import AbstractUNet
from models import conv1x1
from models.baseline import UNet as SubUNet

class AbstractWeightNetwork(AbstractUNet):
    """This class encapsulates common operations of the weight networks.
    """

    def __init__(self, config):
        self.num_subnets = config['NUM_SUBNETS']
        self.sub_net_depth = config['SUB_NET_DEPTH']
        self.weight_mode = config['WEIGHT_MODE']
        self.weight_constraint = config['WEIGHT_CONSTRAINT']
        self.weight_constraint_lambda = config['WEIGHT_CONSTRAINT_LAMBDA']
        self.subnet_config = copy.deepcopy(config)
        self.subnet_config['DEPTH'] = config['SUB_NET_DEPTH']

        config['DEPTH'] = config['MAIN_NET_DEPTH']
        super(AbstractWeightNetwork, self).__init__(config)

    def _build_network_head(self, outs):
        self.subnets = nn.ModuleList()
        self.final_ops = nn.ModuleList()
        for _ in range(self.num_subnets):
            # We create each requested subnet
            # TODO Make main and subnets individually configurable
            self.subnets.append(SubUNet(self.subnet_config))
        self._build_final_ops(outs)

    def _build_final_ops(self, outs):
        for _ in range(self.num_subnets):
            self.final_ops.append(conv1x1(outs, 1))

    def loss_function(self, result):
        if self.weight_constraint == 'entropy':
            return self.loss_function_with_entropy(result)
        return self.loss_function_without_entropy(result)

    def loss_function_without_entropy(self, result):
        output, ground_truth, mask = result['output'], result['gt'], result['mask']
        mask_sum = torch.sum(mask)
        difference = torch.sum(mask * (ground_truth - output)**2)
        loss = difference / mask_sum
        return loss

    def loss_function_with_entropy(self, result):
        output = result['output']
        ground_truth = result['gt']
        mask = result['mask']
        weights = result['weights']
        loss = torch.sum(mask * (ground_truth - output)**2) / torch.sum(mask)
        # Weights are in shape [batch_size, num_subnets]
        # Sum up the individual pairs of the weights for the subnetworks
        # to be able to normalize them
        weights_sum = torch.sum(weights, 1)
        if self.weight_mode == 'image':
            # Add dimension s.t. summed up weights can be broadcasted
            weights_sum = weights_sum.unsqueeze(-1)
        else:
            # weight mode 'pixel'
            weights_sum = weights_sum.unsqueeze(1)
        weights = weights / weights_sum
        # Mean along dimension 0 to obtain the mean of the weight of a
        # subnetwork for the given batch
        weights = torch.mean(weights, 0)
        entropy = -torch.sum(weights * torch.log(weights))
        return loss - self.weight_constraint_lambda * entropy

    def forward(self, x):
        raise NotImplementedError

    def training_predict(self, sample):
        raise NotImplementedError

    def predict(self, image):
        raise NotImplementedError

    def predict_patch(self, patch):
        raise NotImplementedError

class ImageWeightUNet(AbstractWeightNetwork):
    """This network computes the weights for the subnetworks on a per-image basis.
    This means that each subnetwork gets one weight. The whole output of each
    subnetwork is multiplied by the respective weight. These weighted outputs
    are then summed up and divided by the sum of the weights.
    """

    def __init__(self, config):
        super(ImageWeightUNet, self).__init__(config)
        self.counter = 0

    def forward(self, x):
        encoder_outs = []

        # encoder pathway, save outputs for merging
        for i, module in enumerate(self.down_convs):
            x, before_pool = module(x)
            encoder_outs.append(before_pool)

        for i, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i+2)]
            x = module(before_pool, x)

        # x = [num_subnets, batch_size, C, H, W]
        x = torch.stack([final_op(x) for final_op in self.final_ops])
        # We want weights for the subnets, i.e. we need to take the mean
        # (this is the image-wise weight network) along all other dimensions
        # except for the batch dimension, since the individual entries in a
        # batch do not necessarily belong to the same image
        x = torch.mean(x, (2, 3, 4))
        x = torch.exp(x)
        # NOTE we can't put the computation of the subnetworks in here because
        # during inference we need to compute the final output image patch-wise
        # (due to memory on the GPU). This would mean that we compute the weight
        # only based on the patch and not the whole input image.
        return x

    def training_predict(self, sample):
        raw, ground_truth, mask = sample['raw'], sample['gt'], sample['mask']
        raw, ground_truth, mask = raw.to(self.device),\
                                  ground_truth.to(self.device),\
                                  mask.to(self.device)
        raw, ground_truth, mask = raw.to(
            self.device), ground_truth.to(self.device), mask.to(self.device)

        # NOTE we can't move the amalgamation code to the forward() method
        # because during prediction we predict the sub images patch-wise.
        # If we assembled the final image within the forward() method we would
        # multiply the patches with patch-wise weights and not for the whole image.

        # Forward step
        # Actually we compute the weights for only one patch here, not the whole
        # image. We could compute the weights for one image if we ran the network
        # on all images that are fused in the batch that has been created above.
        # This could fail if the images are too large. Also, we would have to
        # obtain the indices of the images used in the batch to retrieve them.
        # We assume that training the network on patches and during prediction
        # averaging the weights works just as well since the network performs
        # some kind of averaging already but only on the patch-level.
        # [subnets, batch]
        original_weights = self(raw)
        # [subnets, batch, channels, H, W]
        sub_outputs = torch.stack([sub(raw) for sub in self.subnets])
        # We compute weights on a per-image basis -> one weight for one image
        # If we were to perform Probabilistic Noise2Void we would need
        # the reconstruction of based on the output samples and weight this
        # reconstruction.
        # torch is not able to broadcast weights directly thus we have to
        # expand the shape of the weights.
        weights = original_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        weighted_sub_outputs = weights * sub_outputs

        # [batch, H, W] / [batch]
        # i.e. we divide the patches that are comprised of weighted sums
        # of the outputs of the subnetworks by the sum of the weights
        output = torch.sum(weighted_sub_outputs, dim=0) / torch.sum(weights, dim=0)

        # Transpose s.t. batch size is the first entry and remove unecessary
        # dimensions
        weights = weights.squeeze(-1).squeeze(-1).squeeze(-1)
        weights = weights.transpose(1, 0)
        # Bring batch size to front [batch_size, num_subnets, C, H, W]
        weighted_sub_outputs = weighted_sub_outputs.transpose(1, 0)
        
        return  {'gt'         : ground_truth,
                 'mask'       : mask,
                 'output'     : output,
                 'weights'    : weights,
                 'sub_outputs': weighted_sub_outputs}

    def predict(self, image):
        # weights for each subnet for the whole imagess
        weights = []
        # sub_images = [subnets, batch_size, C, H, W]
        sub_images = np.zeros((self.num_subnets,) + image.shape)
        
        image_height = image.shape[-2]
        image_width = image.shape[-1]

        xmin = 0
        ymin = 0
        xmax = self.patch_size
        ymax = self.patch_size
        ovLeft = 0
        while (xmin < image_width):
            ovTop = 0
            while (ymin < image_height):
                patch = image[:, :, ymin:ymax, xmin:xmax]
                # NOTE: We save all weights for all patches and average later ->
                # this is not identical to computing one weight based on the
                # whole image but we have use tiling due to memory constraints.
                # In training we take the exp of the mean so in order to average
                # over the patches we take the log here and then exp later.
                weights.append(np.log(self.predict_patch(patch)))
                sub_outputs = np.array([subnet.predict_patch(patch)\
                                        for subnet in self.subnets])
                sub_images[:, :, :, ymin:ymax, xmin:xmax][:, :, :, ovTop:, ovLeft:]\
                    = sub_outputs[:, :, :, ovTop:, ovLeft:]
                ymin = ymin-self.overlap+self.patch_size
                ymax = ymin+self.patch_size
                ovTop = self.overlap//2
            ymin = 0
            ymax = self.patch_size
            xmin = xmin-self.overlap+self.patch_size
            xmax = xmin+self.patch_size
            ovLeft = self.overlap//2

        ### Amalgamte image

        # [num_subnets] = weights for the whole image for each subnet
        weights = np.mean(np.array(weights), axis=0)
        # NOTE: we do exp here since this normally happens in training but
        # we "undid" it above taking the log to be able to average the weights
        # across the patches.
        weights = np.exp(weights)
        # Expand dimensions to match the image's
        # This shape is [num_subnets, batch_size, C, H, W]
        weights = weights.reshape((2, 1, 1, 1, 1))
        # sub_images * weights = [num_subnets, batch_size, H, W] * [num_subnets]
        weighted_sub_outputs = sub_images * weights
        output = np.sum(weighted_sub_outputs, axis=0) / np.sum(weights)

        ### Transpose in correct order
        
        # Transepose from [batch_size, C, H, W] to [batch_size, H, W, C]
        output = output.transpose((0, 2, 3, 1))
        # Keep [num_subnets, batch_size] as shape
        batch_size = weights.shape[1]
        weights.shape = (self.num_subnets, batch_size)
        # Transpose to [batch_size, num_subnets]
        weights = weights.transpose((1, 0))
        # Transpose from [num_subnets, batch_size, C, H, W] to
        # [batch_size, num_subnets, H, W, C]
        weighted_sub_outputs = weighted_sub_outputs.transpose((1, 0, 3, 4, 2))

        # We assume we have batch size 1 always, remove if not the case
        output = output[0]
        weights = weights[0]
        weighted_sub_outputs = weighted_sub_outputs[0]

        return {'output'     : output,
                'weights'    : weights,
                'sub_outputs': weighted_sub_outputs}

    def predict_patch(self, patch):
        # Add one dimension for the batch size which is 1 during prediction
        inputs = patch.to(self.device)
        output = self(inputs)

        weights = output.cpu().detach().numpy()
        return weights

class PixelWeightUNet(AbstractWeightNetwork):
    """This network computes weights for each subnetwork on a per-pixel basis.
    This means that each pixel gets a weights for each subnet. The outputs of
    the subnets are multiplied with their respective weights and added up.
    Afterwards the results is divided by the sum of the weights.
    """

    def __init__(self, config):
        super(PixelWeightUNet, self).__init__(config)

    def forward(self, x):
        encoder_outs = []
        # Compute the outputs of the subnetworks - this is ok here since we
        # do not want weights for the whole image. I.e. we can directly multiply
        # output of the subnetworks and computed weights.
        # [num_subnets, batch_size, C, H, W]
        sub_outputs = torch.stack([sub(x) for sub in self.subnets])

        # encoder pathway, save outputs for merging
        for i, module in enumerate(self.down_convs):
            x, before_pool = module(x)
            encoder_outs.append(before_pool)

        for i, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i+2)]
            x = module(before_pool, x)

        # Compute the pixel-wise weights for the subnetworks
        # x = [num_subnets, batch_size, C, H, W]
        weights = torch.stack([final_op(x) for final_op in self.final_ops])
        weights = torch.exp(weights)
        sub_images = weights * sub_outputs
        # Sum along first axis, i.e. subnets, and divide by sum of weights
        # Shape of sub_images is [num_subnets, batch_size, C, H, W]
        amalgamted_image = torch.sum(sub_images, 0) / torch.sum(weights, 0)
        return amalgamted_image, sub_images, weights

    def training_predict(self, sample):
        raw, ground_truth, mask = sample['raw'], sample['gt'], sample['mask']
        raw, ground_truth, mask = raw.to(self.device),\
                                    ground_truth.to(self.device),\
                                    mask.to(self.device)
        raw, ground_truth, mask = raw.to(
            self.device), ground_truth.to(self.device), mask.to(self.device)
        
        amalgamted_image, sub_outputs, weights = self(raw)

        # Switch back to [batch_size, num_subnets, H, W] - 2 is channel dim
        weights = weights.transpose(1, 0).squeeze(2)
        # Switch batch size and num_subnets too
        sub_outputs = sub_outputs.transpose(1, 0)
        
        return  {'gt'         : ground_truth,
                    'mask'       : mask,
                    'output'     : amalgamted_image,
                    'weights'    : weights,
                    'sub_outputs': sub_outputs}

    def predict(self, image):
        # weights for each subnet for each pixel (without channel dim)
        # [num_subnets, batch_size, H, W]
        weights = np.zeros((self.num_subnets, image.shape[0]) + image.shape[2:])
        # [num_subnets, batch_size, C, H, W]
        sub_outputs = np.zeros((self.num_subnets,) + image.shape)
        # sub_images = [batch_size, C, H, W] since the network amalgamtes images
        amalgamted_image = np.zeros(image.shape)

        image_width = image.shape[-1]
        image_height = image.shape[-2]

        xmin = 0
        ymin = 0
        xmax = self.patch_size
        ymax = self.patch_size
        ovLeft = 0
        while (xmin < image_width):
            ovTop = 0
            while (ymin < image_height):
                patch = image[:, :, ymin:ymax, xmin:xmax]
                _amalgamted_image, _sub_outputs, _weights = self.predict_patch(patch)
                amalgamted_image[:, :, ymin:ymax, xmin:xmax][:, :, ovTop:, ovLeft:]\
                    = _amalgamted_image[:, :, ovTop:, ovLeft:]
                sub_outputs[:, :, :, ymin:ymax, xmin:xmax][:, :, :, ovTop:, ovLeft:]\
                    = _sub_outputs[:, :, :, ovTop:, ovLeft:]
                weights[:, :, ymin:ymax, xmin:xmax][:, :, ovTop:, ovLeft:]\
                    = _weights[:, :, ovTop:, ovLeft:]
                ymin = ymin-self.overlap+self.patch_size
                ymax = ymin+self.patch_size
                ovTop = self.overlap//2
            ymin = 0
            ymax = self.patch_size
            xmin = xmin-self.overlap+self.patch_size
            xmax = xmin+self.patch_size
            ovLeft = self.overlap//2

        # Transepose from [batch_size, C, H, W] to [batch_size, H, W, C]
        amalgamted_image = amalgamted_image.transpose((0, 2, 3, 1))
        # Transpose from [num_subnets, batch_size, C, H, W] to
        # [batch_size, num_subnets, H, W, C]
        sub_outputs = sub_outputs.transpose((1, 0, 3, 4, 2))
        # Transpose from [num_subnets, batch_size, H, W] to
        # [batch_size, num_subnets, H, W]
        weights = weights.transpose((1, 0, 2, 3))

        # We always have only one batch during prediciton
        amalgamted_image = amalgamted_image[0]
        sub_outputs = sub_outputs[0]
        weights = weights[0]

        return {'output'     : amalgamted_image,
                'weights'    : weights,
                'sub_outputs': sub_outputs}

    def predict_patch(self, patch):
        inputs = patch.to(self.device)
        amalgamted_image, sub_images, weights = self(inputs)
        
        amalgamted_image = amalgamted_image.cpu().detach().numpy()
        sub_images = sub_images.cpu().detach().numpy()
        weights = weights.cpu().detach().numpy()
        # We don't need the channel dimension in the weights, it was only added
        # in the forward method to ensure shape compatibility between the weights
        # and the sub images
        weights = weights[:, :, 0]
        # Since we only call the forward() method on the subimages which does
        # not denormalize images we need to do this here manually
        amalgamted_image = util.denormalize(amalgamted_image, self.mean, self.std)
        sub_images = util.denormalize(sub_images, self.mean, self.std)
        return amalgamted_image, sub_images, weights
