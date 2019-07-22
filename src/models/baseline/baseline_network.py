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

class UNet(AbstractUNet):
    """Baseline network that is only a simple UNet. This network is used to
    compare the performance of more complex network structures.
    """
    
    def _build_network_head(self, outs):
        # TODO actually output should be self.num_channels x self.num_classes
        # to account for RGB images and Probabilistic Noise2Void
        self.network_head = conv1x1(outs, 1)

    @staticmethod
    def loss_function(outputs, labels, masks):
        outs = outputs[:, 0, ...]
        # print(outs.shape,labels.shape,masks.shape)
        # Simple L2 loss
        loss = torch.sum(masks*(labels-outs)**2)/torch.sum(masks)
        return loss

    def forward(self, x):
        encoder_outs = []

        # encoder pathway, save outputs for merging
        for i, module in enumerate(self.down_convs):
            x, before_pool = module(x)
            encoder_outs.append(before_pool)

        for i, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i+2)]
            x = module(before_pool, x)

        # No softmax is used. This means you need to use
        # nn.CrossEntropyLoss is your training script,
        # as this module includes a softmax already.
        x = self.network_head(x)
        return x

    def training_predict(self, train_data, train_data_clean, data_counter, size, box_size, bs):
        inputs, labels, masks = self.assemble_training__batch(bs, size, box_size, 
                                    data_counter, train_data, train_data_clean)

        # Move to GPU
        inputs, labels, masks = inputs.to(
            self.device), labels.to(self.device), masks.to(self.device)

        # Forward step
        outputs = self(inputs)
        return outputs, labels, masks, data_counter

    def predict(self, image, patch_size, overlap):
        means = np.zeros(image.shape)
        # We have to use tiling because of memory constraints on the GPU
        xmin = 0
        ymin = 0
        xmax = patch_size
        ymax = patch_size
        ovLeft = 0
        while (xmin < image.shape[1]):
            ovTop = 0
            while (ymin < image.shape[0]):
                a = self.predict_patch(image[ymin:ymax, xmin:xmax])
                means[ymin:ymax, xmin:xmax][ovTop:,
                                            ovLeft:] = a[ovTop:, ovLeft:]
                ymin = ymin-overlap+patch_size
                ymax = ymin+patch_size
                ovTop = overlap//2
            ymin = 0
            ymax = patch_size
            xmin = xmin-overlap+patch_size
            xmax = xmin+patch_size
            ovLeft = overlap//2
        return means

    def predict_patch(self, patch):
        # In case of Probabilistic Noise2Void we would have samples from
        # multiple Gaussian distributions and inputs[0, :, :, :] would become
        # inputs[num_classes, :, :, :]
        # Shape of inputs is [num_classes, channels, H, W]
        inputs = torch.zeros(1, 1, patch.shape[0], patch.shape[1])
        inputs[0, :, :, :] = util.img_to_tensor(patch)

        # copy to GPU
        inputs = inputs.to(self.device)
        output = self(inputs)
        # Permute batch and samples
        samples = (output).permute(1, 0, 2, 3)

        # In contrast to probabilistic N2V we only have one sample
        means = samples[0, ...]
        # Get data from GPU
        means = means.cpu().detach().numpy()
        # Reshape to 2D images and remove padding
        means.shape = (output.shape[2], output.shape[3])
        # Denormalize
        means = util.denormalize(means, self.mean, self.std)
        return means
