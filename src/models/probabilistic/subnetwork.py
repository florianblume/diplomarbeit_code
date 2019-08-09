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

    def __init__(self, mean, std, is_integrated, in_channels=1,
                 depth=5, start_filts=64, up_mode='transpose',
                 merge_mode='add', augment_data=True,
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        self.is_integrated = is_integrated
        super(SubUNet, self).__init__(mean, std, in_channels,
                                      depth, start_filts, up_mode, merge_mode,
                                      augment_data, device)

    def _build_network_head(self, outs):
        self.conv_final_mean = conv1x1(outs, self.in_channels)
        self.conv_final_std = conv1x1(outs, self.in_channels)

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

        c = torch.log(1 / (torch.sqrt(2 * math.pi * std**2)))
        # exp is no exponential here because we take the log of the loss
        exp = ((ground_truth - mean)**2)/(2 * std**2)
        loss = torch.sum(mask * (c - exp))/torch.sum(mask)
        return loss

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

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x):
        encoder_outs = []

        # encoder pathway, save outputs for merging
        for i, module in enumerate(self.down_convs):
            x, before_pool = module(x)
            encoder_outs.append(before_pool)

        for i, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i+2)]
            x = module(before_pool, x)

        mean = self.conv_final_mean(x)
        # exp makes std positive (which it always is)
        std = torch.exp(self.conv_final_std(x))
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
                'mask'   : mask}

    def predict(self, image, patch_size, overlap):
        mean = np.zeros(image.shape)
        std = np.zeros(image.shape)
        
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
                _mean, _std = self.predict_patch(image[:, :, ymin:ymax, xmin:xmax])
                mean[:, :, ymin:ymax, xmin:xmax][:, :, ovTop:, ovLeft:] =\
                                                    _mean[:, :, ovTop:, ovLeft:]
                std[:, :, ymin:ymax, xmin:xmax][ovTop:, ovLeft:] =\
                                                    _std[:, :, ovTop:, ovLeft:]
                ymin = ymin - overlap + patch_size
                ymax = ymin + patch_size
                ovTop = overlap//2
            ymin = 0
            ymax = patch_size
            xmin = xmin - overlap + patch_size
            xmax = xmin + patch_size
            ovLeft = overlap//2
        return mean, std

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
