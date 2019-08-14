import math
import torch
import numpy as np

import util
from models import AbstractUNet
from models import conv1x1

class SubUNet(AbstractUNet):
    """Base class for the subnetworks. The reason is that when training the
    probabilistic (predicts mean and std) subnetwork in standalone mode we need
    a different loss compared to the integrated version.
    """

    def __init__(self, config):
        self.is_integrated = config['IS_INTEGRATED']
        super(SubUNet, self).__init__(config)

    def _build_network_head(self, outs):
        # 2 output channels for mean and std, respectively
        self.conv_final = conv1x1(outs, 2)

    def loss_function(self, result):
        if self.is_integrated:
            return self.loss_function_integrated(result)
        return self.loss_function_standalone(result)

    def loss_function_standalone(self, result):
        # Mean and std predicted for the pixels
        mean = result['mean']
        std = result['std']
        ground_truth = result['gt']
        mask = result['mask']

        # We formulate the loss as maximizing the probability of an output
        # pixel drawn from a Gaussian distribution
        factor = torch.log(1.0 / torch.sqrt(2.0 * math.pi * (std**2)))
        # exp is no exponential here because we take the log of the loss
        exp = ((ground_truth - mean)**2)/(2.0 * (std**2))
        loss = torch.sum(mask * (factor - exp)) / torch.sum(mask)
        # -loss because we want to maximize the probability of the output
        # i.e. minimize the negative loss
        return -loss

    def loss_function_integrated(self, result):
        # Mean and std predicted for the pixels
        mean = result['mean']
        std = result['std']
        ground_truth = result['gt']

        c = 1 / (torch.sqrt(2 * math.pi * std**2))
        exp = torch.exp(-((ground_truth - mean)**2)/(2 * std**2))
        # We do not want to sum here as the loss is continued in the main network
        #loss = torch.sum(masks * (c * exp)/torch.sum(masks)
        # Return the Gaussian
        return c + exp

    def forward(self, x):
        encoder_outs = []

        # encoder pathway, save outputs for merging
        for i, module in enumerate(self.down_convs):
            x, before_pool = module(x)
            encoder_outs.append(before_pool)

        for i, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i+2)]
            x = module(before_pool, x)

        output = self.conv_final(x)
        mean, std = output[:, 0], output[:, 1]
        # exp makes std positive (which it always is)
        std = torch.exp(std)
        return mean, std

    def training_predict(self, sample):
        raw, ground_truth, mask = sample['raw'], sample['gt'], sample['mask']
        raw, ground_truth, mask = raw.to(self.device),\
                                   ground_truth.to(self.device),\
                                   mask.to(self.device)
        mean, std = self(raw)
        # Return mean as output to ensure compatibility to Trainer code
        return {'output' : mean,
                'mean'   : mean,
                'std'    : std,
                'gt'     : ground_truth,
                'mask'   : mask,
                'raw'    : raw}

    def _pre_process_predict(self, image):
        mean = np.zeros(image.shape)
        std = np.zeros(image.shape)
        return {'image': image,
                'mean' : mean,
                'std'  : std}

    def _process_patch(self, data, ymin, ymax, xmin, xmax, ovTop, ovLeft):
        image = data['image']
        mean = data['mean']
        std = data['std']
        _mean, _std = self.predict_patch(image[:, :, ymin:ymax, xmin:xmax])
        mean[:, :, ymin:ymax, xmin:xmax][:, :, ovTop:, ovLeft:] =\
                                            _mean[:, :, ovTop:, ovLeft:]
        std[:, :, ymin:ymax, xmin:xmax][:, :, ovTop:, ovLeft:] =\
                                            _std[:, :, ovTop:, ovLeft:]
        data['mean'] = mean
        data['std'] = std
        return data

    def predict_patch(self, patch):
        inputs = patch.to(self.device)

        mean, std = self(inputs)

        # Get data from GPU
        mean = mean.cpu().detach().numpy()
        std = std.cpu().detach().numpy()

        # Denormalize
        mean = util.denormalize(mean, self.mean, self.std)
        std = util.denormalize(std, self.mean, self.std)
        return mean, std

    def _post_process_predict(self, result):
        mean = result['mean']
        std = result['std']
        # At the moment we always have implicit batch size 1
        out_image = mean[0]
        # Transpose to [C, H, W]
        out_image = out_image.transpose(1, 2, 0)
        
        return {'output' : out_image,
                'mean'   : mean,
                'std'    : std}
