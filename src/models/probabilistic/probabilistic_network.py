import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import tifffile as tif

from models import AbstractUNet
from models import conv1x1
from models.probabilistic import SubUNet

class ImageProbabilisticUNet(AbstractUNet):

    def __init__(self, config):
        self.sub_net_depth = config['SUB_NET_DEPTH']
        self.num_subnets = config['NUM_SUBNETS']
        self.weight_mode = 'image'
        self.subnet_config = copy.deepcopy(config)
        self.subnet_config['DEPTH'] = config['SUB_NET_DEPTH']
        self.subnet_config['IS_INTEGRATED'] = True

        config['DEPTH'] = config['MAIN_NET_DEPTH']
        super(ImageProbabilisticUNet, self).__init__(config)

    def _build_network_head(self, outs):
        self.subnets = nn.ModuleList()
        for _ in range(self.num_subnets):
            # create the subnets
            self.subnets.append(SubUNet(self.subnet_config))

        if self.num_subnets == 2:
            # In case that we only have two subnets we produce only one output
            # with sigmoid, the second probability is implicit in this case
            count = self.num_subnets - 1
        else:
            count = self.num_subnets

        # create the probability weights, this can be seen as p(z|x), i.e. the probability
        # of a decision given the input image. To obtain a weighted "average" of the
        # predictions of the subnets, we multiply this weight to their output.
        self.weight_probabilities = conv1x1(outs, count)

    def loss_function(self, result):
        # Probabilites for each subnetwork per image, i.e. [batch_size, num_subnets]
        probabilities = result['probabilities']


        # Assemble dict for subnets on-the-fly because list comprehension is
        # faster in Python
        losses = []
        for i, subnet in enumerate(self.subnets):
            subnet_result = {'mean' : result['sub_outputs'][i][0],
                             'std'  : result['sub_outputs'][i][1],
                             'gt'   : result['gt']}
            loss = subnet.loss_function_integrated(subnet_result)
            losses.append(loss)
            
        sub_losses = torch.stack(losses)
        ### We now do a log-exp modification to be able to subtract a constant
        # Transpose to [batch, subnet, ...]
        sub_losses = sub_losses.transpose(1, 0)
        # Add a small factor to avoid log(0)
        sub_losses = torch.log(sub_losses + 1e-10)
        # We can now sum up (instead of multiply) over all pixels (and channels)
        sub_losses = torch.sum(torch.sum(torch.sum(sub_losses, dim=-1), dim=-1), dim=-1)
        # We don't want gradients to be propagated along this path as it's a
        # constant
        # We need to take the maxima for each batch individually
        max_sub_losses = torch.max(sub_losses, dim=1).values
        # Add singleton dimension to ensure shape compatibility
        max_sub_losses = max_sub_losses.unsqueeze(-1)
        sub_losses -= max_sub_losses
        sub_losses = torch.exp(sub_losses)
        loss = torch.sum(probabilities * sub_losses, dim=1)
        # Undo that we subtracted a constant
        loss = torch.log(loss)
        # Sum over all decisions (i.e. subnets)
        summed_loss = torch.sum(loss) + torch.sum(max_sub_losses)
        return -summed_loss

    def forward(self, x):
        encoder_outs = []

        # encoder pathway, save outputs for merging
        for i, module in enumerate(self.down_convs):
            x, before_pool = module(x)
            encoder_outs.append(before_pool)

        for i, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i+2)]
            x = module(before_pool, x)

        # Compute the individual weight probabilities
        # One probability (weight) for each subnet
        if self.num_subnets == 2:
            # probabilities are of shape [batch, subnet, H, W] before taking mean
            probabilities = self.weight_probabilities(x)
            # Take mean across H and W (and subnets because that dim is 1 anyways)
            probabilities = torch.mean(probabilities, (1, 2, 3))
            first_prob = torch.sigmoid(probabilities)
            ones = torch.ones(first_prob.shape).to(self.device)
            outs = torch.stack([first_prob, ones - first_prob])
            outs = outs.transpose(1, 0)
        else:
            # probabilities are of shape [batch, subnet, H, W] before taking mean
            outs = torch.mean(self.weight_probabilities(x), (-2, -1))
            # Dim 1 is the subnet dimension
            outs = F.softmax(outs, dim=1)
        return outs

    def training_predict(self, sample):
        raw, ground_truth, mask = sample['raw'], sample['gt'], sample['mask']
        raw, ground_truth, mask = raw.to(self.device),\
                                   ground_truth.to(self.device),\
                                   mask.to(self.device)

        # Forward step
        probabilities = self(raw)
        sub_outputs = [subnet(raw) for subnet in self.subnets]

        return {'probabilities': probabilities,
                'sub_outputs'  : sub_outputs,
                'gt'           : ground_truth,
                'mask'         : mask}

    def _pre_process_predict(self, image):
        # list because we need to average the probabilities for the whole image
        # but can only compute them patch-wise
        probabilities = []
        mean = np.zeros((self.num_subnets,) + image.shape)
        std = np.zeros((self.num_subnets,) + image.shape)
        return {'image'         : image,
                'probabilities' : probabilities,
                'mean'          : mean,
                'std'           : std}

    def _process_patch(self, data, ymin, ymax, xmin, xmax, ovTop, ovLeft):
        image = data['image']
        probabilities = data['probabilities']
        mean = data['mean']
        std = data['std']
        patch = image[:, :, ymin:ymax, xmin:xmax]
        probabilities.append(self.predict_patch(patch))
        sub_outputs = np.array([subnet.predict_patch(patch)\
                                for subnet in self.subnets])
        mean[:, :, :, ymin:ymax, xmin:xmax]\
                [:, :, :, ovTop:, ovLeft:] = sub_outputs[:, 0, :, :, ovTop:, ovLeft:]
        std[:, :, :, ymin:ymax, xmin:xmax]\
                [:, :, :, ovTop:, ovLeft:] = sub_outputs[:, 1, :, :, ovTop:, ovLeft:]

    def predict_patch(self, patch):
        inputs = patch.to(self.device)
        probabilities = self(inputs)
        probabilities = probabilities.cpu().detach().numpy()
        return probabilities

    def _post_process_predict(self, result):
        # [batch, subnet]
        probabilities = result['probabilities']
        mean = result['mean']
        std = result['std']
        # We do [0] because we only have one batch
        # Transpose from [subnet, batch, C, H, W] to [batch, subnet, H, W, C]
        mean = mean.transpose((1, 0, 3, 4, 2))[0]
        std = std.transpose((1, 0, 3, 4, 2))[0]
        # Take the mean of the probabilities computed for the individual patches
        # to obtain the probabilities for the whole images of the subnetworks
        probabilities = np.mean(probabilities, axis=0)[0]
        probabilities = probabilities.reshape((probabilities.shape[0], 1, 1, 1))
        amalgamted_image = probabilities * mean
        amalgamted_image = np.sum(amalgamted_image, axis=0)
        return {'output' : amalgamted_image,
                'mean'   : mean,
                'std'    : std}

class PixelProbabilisticUNet(AbstractUNet):

    def __init__(self, config):
        self.sub_net_depth = config['SUB_NET_DEPTH']
        self.num_subnets = config['NUM_SUBNETS']
        self.weight_mode = 'pixel'
        self.subnet_config = copy.deepcopy(config)
        self.subnet_config['DEPTH'] = config['SUB_NET_DEPTH']
        self.subnet_config['IS_INTEGRATED'] = True

        config['DEPTH'] = config['MAIN_NET_DEPTH']
        super(PixelProbabilisticUNet, self).__init__(config)

    def _build_network_head(self, outs):
        self.subnets = nn.ModuleList()
        for _ in range(self.num_subnets):
            # create the subnets
            self.subnets.append(SubUNet(self.subnet_config))

        if self.num_subnets == 2:
            # In case that we only have two subnets we produce only one output
            # with sigmoid, the second probability is implicit in this case
            count = self.num_subnets - 1
        else:
            count = self.num_subnets

        # create the probability weights, this can be seen as p(z|x), i.e. the probability
        # of a decision given the input image. To obtain a weighted "average" of the
        # predictions of the subnets, we multiply this weight to their output.
        self.weight_probabilities = conv1x1(outs, count)

    def loss_function(self, result):
        # Probabilites for each subnetwork per pixel,
        # i.e. [batch_size, num_subnets, H, W]
        probabilities = result['probabilities']
        mask = result['mask']

        # Assemble dict for subnets on-the-fly because list comprehension is
        # faster in Python
        sub_losses = [self.subnets[i].loss_function_integrated(
                                    {'mean' : result['sub_outputs'][i][0],
                                     'std'  : result['sub_outputs'][i][1],
                                     'gt'   : result['gt']}
                      ) for i in range(self.num_subnets)]
        # [subnet, batch, C, H, W]
        sub_losses = torch.stack(sub_losses)
        # Transpose to [batch, subnet, C, H, W]
        sub_losses = sub_losses.transpose(1, 0)
        # To match sub_losses shape
        probabilities = probabilities.unsqueeze(2)
        loss = probabilities * sub_losses
        # Sum over subnet dimension
        loss = torch.sum(loss, 1)
        loss = mask * loss
        loss = torch.log(loss + 1e-10)
        # Mean instead of sum is only a factor
        return -torch.mean(loss)

    def forward(self, x):
        encoder_outs = []

        # encoder pathway, save outputs for merging
        for i, module in enumerate(self.down_convs):
            x, before_pool = module(x)
            encoder_outs.append(before_pool)

        for i, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i+2)]
            x = module(before_pool, x)

        # Compute the individual weight probabilities
        # One probability (weight) for each subnet
        if self.num_subnets == 2:
            # probabilities are of shape [batch, subnet, H, W] before taking mean
            probabilities = self.weight_probabilities(x)
            first_prob = torch.sigmoid(probabilities)
            ones = torch.ones(first_prob.shape).to(self.device)
            outs = torch.stack([first_prob, ones - first_prob])
            # Transpose to [batch, subnet]
            outs = outs.transpose(1, 0)
        else:
            # probabilities are of shape [batch, subnet, H, W] before taking mean
            outs = self.weight_probabilities(x)
            outs = F.softmax(outs, dim=1)
        # Remove unnecessary channel dimension
        outs = outs.squeeze(2)
        return outs

    def training_predict(self, sample):
        raw, ground_truth, mask = sample['raw'], sample['gt'], sample['mask']
        raw, ground_truth, mask = raw.to(self.device),\
                                   ground_truth.to(self.device),\
                                   mask.to(self.device)

        # Forward step
        probabilities = self(raw)
        sub_outputs = [subnet(raw) for subnet in self.subnets]

        return {'probabilities': probabilities,
                'sub_outputs'  : sub_outputs,
                'gt'           : ground_truth,
                'mask'         : mask}

    def _pre_process_predict(self, image):
        # Omit channel dimension and add subnet dimension
        probabilities = np.zeros((image.shape[0], self.num_subnets) + image.shape[2:])
        mean = np.zeros((self.num_subnets,) + image.shape)
        std = np.zeros((self.num_subnets,) + image.shape)
        return {'image'         : image,
                'probabilities' : probabilities,
                'mean'          : mean,
                'std'           : std}

    def _process_patch(self, data, ymin, ymax, xmin, xmax, ovTop, ovLeft):
        image = data['image']
        probabilities = data['probabilities']
        mean = data['mean']
        std = data['std']
        patch = image[:, :, ymin:ymax, xmin:xmax]
        probabilities_ = self.predict_patch(patch)
        probabilities[:, :, ymin:ymax, xmin:xmax]\
                [:, :, ovTop:, ovLeft:] = probabilities_[:, :, ovTop:, ovLeft:]
        sub_outputs = np.array([subnet.predict_patch(patch)\
                                for subnet in self.subnets])
        # Dim 0 is mean and dim 1 is std
        mean[:, :, :, ymin:ymax, xmin:xmax]\
            [:, :, :, ovTop:, ovLeft:] = sub_outputs[:, 0, :, :, ovTop:, ovLeft:]
        std[:, :, :, ymin:ymax, xmin:xmax]\
            [:, :, :, ovTop:, ovLeft:] = sub_outputs[:, 1, :, :, ovTop:, ovLeft:]

    def predict_patch(self, patch):
        inputs = patch.to(self.device)
        probabilities = self(inputs)
        probabilities = probabilities.cpu().detach().numpy()
        return probabilities

    def _post_process_predict(self, result):
        # [batch, subnet]
        probabilities = result['probabilities']
        mean = result['mean']
        std = result['std']
        # Transpose from [subnet, batch, C, H, W] to [batch, subnet, H, W, C]
        mean = mean.transpose((1, 0, 3, 4, 2))
        std = std.transpose((1, 0, 3, 4, 2))
        # Add channel dimension to ensure compatibility between shapes
        probabilities = np.expand_dims(probabilities, axis=-1)
        amalgamted_image = probabilities * mean
        # Sum up over the subnets
        amalgamted_image = np.sum(amalgamted_image, axis=1)
        amalgamted_image = amalgamted_image[0]
        probabilities = np.squeeze(probabilities)
        #mean = np.squeeze(mean)
        #std = np.squeeze(std)
        return {'output'        : amalgamted_image,
                'probabilities' : probabilities,
                'mean'          : mean,
                'std'           : std}
