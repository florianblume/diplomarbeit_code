import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from torch.nn import init
import numpy as np

import util
from models import AbstractUNet
from models import conv1x1
from models.average import SubUNet

class PixelWeightUNet(nn.Module):

    def __init__(self, num_classes, mean, std, in_channels=1,
                 main_net_depth=1, sub_net_depth=3, num_subnets=2,
                 start_filts=64, up_mode='transpose', merge_mode='add',
                 augment_data=True,
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        """
        NOTE: mean and std will be persisted by the model and restored on loading

        Arguments:
            mean: int, the mean of the raw data that this network was trained with. 
            std: int, the std of the raw data that this network was trained with.
            in_channels: int, number of channels in the input tensor.
                Default is 3 for RGB images.
            depth: int, number of MaxPools in the U-Net.
            start_filts: int, number of convolutional filters for the
                first conv.
            up_mode: string, type of upconvolution. Choices: 'transpose'
                for transpose convolution or 'upsample' for nearest neighbour
                upsampling.
        """
        self.num_subnets = num_subnets
        self.sub_net_depth = sub_net_depth
        super(PixelWeightUNet, self).__init__(num_classes, mean, std,
                                              in_channels,
                                              depth=main_net_depth,
                                              start_filts=start_filts,
                                              up_mode=up_mode,
                                              merge_mode=merge_mode,
                                              augment_data=augment_data,
                                              device=device)

    def _build_network_head(self, outs):
        self.subnets = nn.ModuleList()
        self.final_op = nn.ModuleList()

        for _ in range(self.num_subnets):
            # We create each requested subnet
            # TODO Make main and subnets individually configurable
            self.subnets.append(SubUNet(self.num_classes, self.mean, self.std,
                                        in_channels=self.in_channels,
                                        depth=self.sub_net_depth,
                                        start_filts=self.start_filts,
                                        up_mode=self.up_mode,
                                        merge_mode=self.merge_mode,
                                        augment_data=self.augment_data,
                                        device=self.device))
            # And for each pixel we output a weight for each subnet
            self.final_op.append(conv1x1(outs, 1))


    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal(m.weight)
            init.constant(m.bias, 0)

    @staticmethod
    def loss_function(outputs, labels, masks):
        # This is the leftover of Probabilistic N2V where the network outputs
        # 800 means per pixel instead of only 1
        outs = outputs[:, 0, ...]
        loss = torch.sum(masks * (labels - outs)**2) / torch.sum(masks)
        return loss

    def reset_params(self):
        for m in self.modules():
            self.weight_init(m)

    def forward(self, x):
        encoder_outs = []
        # Compute the outputs of the subnetworks - this is ok here since we
        # do not want weights for the whole image. I.e. we can directly multiply
        # output of the subnetworks and computed weights.
        # [num_subnets, batch_size, num_classes, H, W]
        sub_outputs = torch.stack([sub(x) for sub in self.subnets])

        # encoder pathway, save outputs for merging
        for i, module in enumerate(self.down_convs):
            x, before_pool = module(x)
            encoder_outs.append(before_pool)

        for i, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i+2)]
            x = module(before_pool, x)

        # Compute the pixel-wise weights for the subnetworks
        # x = [num_subnets, batch_size, num_classes, H, W]
        # where num_classes is only used in Probabilistic Noise2Void
        weights = torch.stack([final_op(x) for final_op in self.final_ops])
        weights = torch.exp(weights)
        # Sum along first axis, i.e. subnets, and divide by sum of weights
        amalgamted_image = torch.sum(weights * sub_outputs, 0) / torch.sum(weights, 0)
        return amalgamted_image, weights

    def training_predict(self, train_data, train_data_clean, data_counter, size, box_size, bs):
        # Init Variables
        inputs, labels, masks = self.assemble_training__batch(bs, size, box_size,
                                    data_counter, train_data, train_data_clean)

        # Move to GPU
        inputs, labels, masks = inputs.to(
            self.device), labels.to(self.device), masks.to(self.device)
        
        # Forward step
        amalgamted_image, weights = self(inputs)
        
        return amalgamted_image, weights, labels, masks, data_counter

    def predict(self, image, patch_size, overlap):
        # weights for each subnet per pixel
        weights = np.zeros((self.num_subnets, image.shape[0], image.shape[1]))
        # sub_images = [subnets, H, W]
        amalgamted_image = np.zeros((self.num_subnets, image.shape[0], image.shape[1]))
        # We have to use tiling because of memory constraints on the GPU
        xmin = 0
        ymin = 0
        xmax = patch_size
        ymax = patch_size
        ovLeft = 0
        while (xmin < image.shape[1]):
            ovTop = 0
            while (ymin < image.shape[0]):
                patch = image[ymin:ymax, xmin:xmax]
                _weights, _amalgamted_image = self.predict_patch(patch)
                weights[:, ymin:ymax, xmin:xmax][:, ovTop:, ovLeft:]\
                    = _weights(patch)
                amalgamted_image[:, ymin:ymax, xmin:xmax][:, ovTop:, ovLeft:]\
                    = _amalgamted_image[:, ovTop:, ovLeft:]
                ymin = ymin-overlap+patch_size
                ymax = ymin+patch_size
                ovTop = overlap//2
            ymin = 0
            ymax = patch_size
            xmin = xmin-overlap+patch_size
            xmax = xmin+patch_size
            ovLeft = overlap//2

        return amalgamted_image, weights

    def predict_patch(self, patch):
        # [batch_size, num_channels, H, W]
        # TODO use num_channels of our class instead of setting it to 1 manually
        inputs = torch.zeros(1, 1, patch.shape[0], patch.shape[1])
        inputs[0, :, :, :] = util.img_to_tensor(patch)
        # copy to GPU
        inputs = inputs.to(self.device)
        amalgamted_image, weights = self(inputs)

        # In contrast to probabilistic N2V we only have one sample
        # weights = samples[0, ...]

        # Get data from GPU
        amalgamted_image = amalgamted_image.cpu().detach().numpy()
        weights = weights.cpu().detach().numpy()
        # Remove unnecessary dimensions
        weights = np.squeeze(weights)
        return amalgamted_image, weights
