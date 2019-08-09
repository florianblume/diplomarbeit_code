import torch
import numpy as np

from models import conv1x1
from models.average import AbstractWeightNetwork

class ImageWeightUNet(AbstractWeightNetwork):
    """This network computes the weights for the subnetworks on a per-image basis.
    This means that each subnetwork gets one weight. The whole output of each
    subnetwork is multiplied by the respective weight. These weighted outputs
    are then summed up and divided by the sum of the weights.
    """

    def __init__(self, num_classes, mean, std, in_channels=1,
                 main_net_depth=1, sub_net_depth=3, num_subnets=2,
                 weight_constraint=None, weights_lambda=0,
                 start_filts=64, up_mode='transpose', merge_mode='add',
                 augment_data=True,
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        super(ImageWeightUNet, self).__init__(num_classes, mean, std,
                                              weight_mode='image',
                                              weight_constraint=weight_constraint,
                                              weights_lambda=weights_lambda,
                                              in_channels=in_channels,
                                              main_net_depth=main_net_depth,
                                              sub_net_depth=sub_net_depth,
                                              num_subnets=num_subnets,
                                              start_filts=start_filts,
                                              up_mode=up_mode,
                                              merge_mode=merge_mode,
                                              augment_data=augment_data,
                                              device=device)
        self.counter = 0

    def _build_final_ops(self, outs):
        for _ in range(self.num_subnets):
            self.final_ops.append(conv1x1(outs, 1))

    def forward(self, x):
        encoder_outs = []

        # encoder pathway, save outputs for merging
        for i, module in enumerate(self.down_convs):
            x, before_pool = module(x)
            encoder_outs.append(before_pool)

        for i, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i+2)]
            x = module(before_pool, x)

        # x = [num_subnets, batch_size, num_classes, H, W]
        # where num_classes is only used in Probabilistic Noise2Void
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
        # [subnets, batch, num_classes, H, W]
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

    def predict(self, image, patch_size, overlap):
        # weights for each subnet for the whole imagess
        weights = []
        # sub_images = [subnets, batch_size, C, H, W]
        sub_images = np.zeros((self.num_subnets,) + image.shape)
        
        image_height = image.shape[-2]
        image_width = image.shape[-1]

        xmin = 0
        ymin = 0
        xmax = patch_size
        ymax = patch_size
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
                ymin = ymin-overlap+patch_size
                ymax = ymin+patch_size
                ovTop = overlap//2
            ymin = 0
            ymax = patch_size
            xmin = xmin-overlap+patch_size
            xmax = xmin+patch_size
            ovLeft = overlap//2

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
