import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import util
from models import AbstractUNet
from models import conv1x1
from models.probabilistic import IntegratedSubUNet

class AbstractSubUNet(AbstractUNet):
    """Base class for the subnetworks. The reason is that when training the
    probabilistic (predicts mean and std) subnetwork in standalone mode we need 
    a different loss compared to the integrated version.
    """

    def __init__(self, num_classes, mean, std, in_channels=1, depth=5,
                 start_filts=64, up_mode='transpose',
                 merge_mode='add',
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):

        super(AbstractSubUNet, self).__init__(num_classes, mean, std, in_channels, 
                main_net_depth, start_filts, up_mode, merge_mode, augment_data, device)

    self _build_network_heads(self, outs):
        self.conv_final_mean = conv1x1(outs, self.num_classes)
        self.conv_final_std = conv1x1(outs, self.num_classes)

    @staticmethod
    def loss_function(mean, std, labels, masks):
        raise 'This function needs to be implemented by the subclasses.'

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

    def training_predict(self, train_data, train_data_clean, data_counter, size, box_size, bs):
        inputs, labels, masks = self.assemble_training__batch(bs, size, box_size,
                                    data_counter, train_data, train_data_clean)
        inputs, labels, masks = inputs.to(
            self.device), labels.to(self.device), masks.to(self.device)
        # We just need the mean (=^ gray color) but the std gives interesting insights
        mean, std = self(inputs)
        return mean, std, labels, masks, data_counter

    def predict(self, image, patch_size, overlap):
        mean = np.zeros(image.shape)
        std = np.zeros(image.shape)
        # We have to use tiling because of memory constraints on the GPU
        xmin = 0
        ymin = 0
        xmax = patch_size
        ymax = patch_size
        ovLeft = 0
        while (xmin < image.shape[1]):
            ovTop = 0
            while (ymin < image.shape[0]):
                _mean, _std = self.predict_patch(image[ymin:ymax, xmin:xmax])
                mean[ymin:ymax, xmin:xmax][ovTop:, ovLeft:] = _mean[ovTop:, ovLeft:]
                std[ymin:ymax, xmin:xmax][ovTop:, ovLeft:] = _std[ovTop:, ovLeft:]
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
        inputs = torch.zeros(1, 1, patch.shape[0], patch.shape[1])
        inputs[0, :, :, :] = util.img_to_tensor(patch)

        # copy to GPU
        inputs = inputs.to(self.device)

        mean, std = self(inputs)

        # TODO check if permutation and taking the 0-th element is actually necessary
        mean_samples = (mean).permute(1, 0, 2, 3)
        std_samples = (std).permute(1, 0, 2, 3)

        # In contrast to probabilistic N2V we only have one sample
        means = mean_samples[0, ...]
        stds = std_samples[0, ...]

        # Get data from GPU
        means = means.cpu().detach().numpy()
        stds = stds.cpu().detach().numpy()

        # Reshape to 2D images and remove padding
        means.shape = (mean.shape[2], mean.shape[3])
        stds.shape = (std.shape[2], std.shape[3])

        # Denormalize
        means = util.denormalize(means, self.mean, self.std)
        stds = util.denormalize(stds, self.mean, self.std)
        return means, std
