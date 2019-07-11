import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from torch.nn import init
import numpy as np

import util
import abstract_network

class SubUNet(abstract_network.AbstractUNet):
    """ `UNet` class is based on https://arxiv.org/abs/1505.04597
    The U-Net is a convolutional encoder-decoder neural network.
    Contextual spatial information (from the decoding,
    expansive pathway) about an input tensor is merged with
    information representing the localization of details
    (from the encoding, compressive pathway).
    Modifications to the original paper:
    (1) padding is used in 3x3 convolutions to prevent loss
        of border pixels
    (2) merging outputs does not require cropping due to (1)
    (3) residual connections can be used by specifying
        UNet(merge_mode='add')
    (4) if non-parametric upsampling is used in the decoder
        pathway (specified by upmode='upsample'), then an
        additional 1x1 2d convolution occurs after upsampling
        to reduce channel dimensionality by a factor of 2.
        This channel halving happens with the convolution in
        the tranpose convolution (specified by upmode='transpose')
    """

    def __init__(self, num_classes, mean, std, in_channels=1, depth=5,
                 start_filts=64, up_mode='transpose',
                 merge_mode='add',
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
        super(SubUNet, self).__init__(num_classes, mean, std, in_channels, 
                main_net_depth, start_filts, up_mode, merge_mode, augment_data, device)

    def _build_network_head(self, outs):
        self.conv_final = conv1x1(outs, self.num_classes)

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
        x = self.conv_final(x)
        return x

    def training_predict(self, train_data, train_data_clean, data_counter, size, box_size, bs):
        """Performs a forward step during training.

        Arguments:
            train_data {np.array} -- the normalized raw training data
            train_data_clean {np.array} -- the normalized ground-truth targets, if available
            data_counter {int} -- the counter when to shuffle the training data
            size {int} -- the patch size
            bs {int} -- the batch size

        Returns:
            np.array, np.array, np.array, int -- outputs, labels, masks, data_counter
        """
        # Init Variables
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
        """Performs network prediction on a patch of an image using the
        specified parameters. The network expects the image to be normalized
        with its mean and std. Likewise, it denormalizes the output images
        using the same mean and std.

        Arguments:
            patch {np.array} -- the patch to perform prediction on
            mean {int} -- the mean of the data the network was trained with
            std {int} -- the std of the data the network was trained with

        Returns:
            np.array -- the denoised and denormalized patch
        """
        inputs = torch.zeros(1, 1, patch.shape[0], patch.shape[1])
        inputs[0, :, :, :] = util.img_to_tensor(patch)

        # copy to GPU
        inputs = inputs.to(self.device)

        output = self(inputs)

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