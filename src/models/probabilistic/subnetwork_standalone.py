import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from torch.nn import init
import numpy as np
import math

import util


def conv3x3(in_channels, out_channels, stride=1,
            padding=1, bias=True, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups)


def upconv2x2(in_channels, out_channels, mode='transpose'):
    if mode == 'transpose':
        return nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2)
    else:
        # out_channels is always going to be the same
        # as in_channels
        return nn.Sequential(
            nn.Upsample(mode='bilinear', scale_factor=2),
            conv1x1(in_channels, out_channels))


def conv1x1(in_channels, out_channels, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        groups=groups,
        stride=1)


class DownConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 MaxPool.
    A ReLU activation follows each convolution.
    """

    def __init__(self, in_channels, out_channels, pooling=True):
        super(DownConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling

        self.conv1 = conv3x3(self.in_channels, self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)

        if self.pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        before_pool = x
        if self.pooling:
            x = self.pool(x)
        return x, before_pool


class UpConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 UpConvolution.
    A ReLU activation follows each convolution.
    """

    def __init__(self, in_channels, out_channels,
                 merge_mode='concat', up_mode='transpose'):
        super(UpConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.merge_mode = merge_mode
        self.up_mode = up_mode

        self.upconv = upconv2x2(self.in_channels, self.out_channels,
                                mode=self.up_mode)

        if self.merge_mode == 'concat':
            self.conv1 = conv3x3(
                2*self.out_channels, self.out_channels)
        else:
            # num of input channels to conv2 is same
            self.conv1 = conv3x3(self.out_channels, self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)

    def forward(self, from_down, from_up):
        """ Forward pass
        Arguments:
            from_down: tensor from the encoder pathway
            from_up: upconv'd tensor from the decoder pathway
        """
        from_up = self.upconv(from_up)
        if self.merge_mode == 'concat':
            x = torch.cat((from_up, from_down), 1)
        else:
            x = from_up + from_down
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x


class SubUNet(nn.Module):
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

    This UNet predicts the mean and standard deviation of the
    probability density of the clean pixels. We assume the noise
    to be i.i.d. This means the probability of the output image
    given the input image is the product of the single probabilities
    of the output pixels conditioned on the input pixels.
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
        super(UNet, self).__init__()

        if up_mode in ('transpose', 'upsample'):
            self.up_mode = up_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for "
                             "upsampling. Only \"transpose\" and "
                             "\"upsample\" are allowed.".format(up_mode))

        if merge_mode in ('concat', 'add'):
            self.merge_mode = merge_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for"
                             "merging up and down paths. "
                             "Only \"concat\" and "
                             "\"add\" are allowed.".format(up_mode))

        # NOTE: up_mode 'upsample' is incompatible with merge_mode 'add'
        if self.up_mode == 'upsample' and self.merge_mode == 'add':
            raise ValueError("up_mode \"upsample\" is incompatible "
                             "with merge_mode \"add\" at the moment "
                             "because it doesn't make sense to use "
                             "nearest neighbour to reduce "
                             "depth channels (by half).")

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.start_filts = start_filts
        self.depth = depth
        self.device = device
        self.mean = mean
        self.std = std

        self.down_convs = []
        self.up_convs = []

        self.noiseSTD = nn.Parameter(data=torch.log(torch.tensor(0.5)))

        # create the encoder pathway and add to a list
        for i in range(depth):
            ins = self.in_channels if i == 0 else outs
            outs = self.start_filts*(2**i)
            pooling = True if i < depth-1 else False

            down_conv = DownConv(ins, outs, pooling=pooling)
            self.down_convs.append(down_conv)

        # create the decoder pathway and add to a list
        # - careful! decoding only requires depth-1 blocks
        for i in range(depth-1):
            ins = outs
            outs = ins // 2
            up_conv = UpConv(ins, outs, up_mode=up_mode,
                             merge_mode=merge_mode)
            self.up_convs.append(up_conv)

        self.conv_final_mean = conv1x1(outs, self.num_classes)
        self.conv_final_std = conv1x1(outs, self.num_classes)

        # add the list of modules to current module
        self.down_convs = nn.ModuleList(self.down_convs)
        self.up_convs = nn.ModuleList(self.up_convs)

        self.reset_params()

        # Push network to respective device
        self.to(self.device)

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal(m.weight)
            init.constant(m.bias, 0)

    @staticmethod
    def loss_function(mean, std, labels, masks):
        # Mean and std for all pixel that the network pred
        mean = mean[:, 0, ...]
        std = std[:, 0, ...]
        ###################
        # We have pixel-wise independent probabilities of the noise. This means
        # the probability of the output image is the product over the probabilities
        # of the individual pixels. To get rid of the product we take the log.
        # This way we get a sum of logs of gaussians. The gaussians can be split
        # (due to the log rule) into the constant based only on std and the exponential
        # which vaniches due to the log.
        ###################
        # N(x; mean, std) = 1 / sqrt(2 * pi * std^2) * e^(-(x - mean)^2 / 2 * std^2)
        # log(a * b) = log(a) + log(b)
        # c is the factor of a gauss prob density based on the standard deviation in
        # front of the exponential
        c = 1 / (torch.sqrt(2 * math.pi * std**2))
        # exp is no exponential here because we take the log of the loss
        exp = torch.exp(-(labels - mean)**2)/(2 * std**2))
        loss = torch.sum(masks * (c + exp)/torch.sum(masks)
        return loss

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

        # No softmax is used. This means you need to use
        # nn.CrossEntropyLoss is your training script,
        # as this module includes a softmax already.
        mean = self.conv_final_mean(x)
        # exp makes std positive (which it always is)
        std = torch.exp(self.conv_final_std(x))
        return mean, std

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
        inputs = torch.zeros(bs, 1, size, size)
        labels = torch.zeros(bs, size, size)
        masks = torch.zeros(bs, size, size)

        # Assemble mini batch
        for j in range(bs):
            im, l, m, data_counter = util.random_crop_fri(
                train_data, size, size, box_size, counter=data_counter, dataClean=train_data_clean)
            inputs[j, :, :, :] = util.img_to_tensor(im)
            labels[j, :, :] = util.img_to_tensor(l)
            masks[j, :, :] = util.img_to_tensor(m)

        # Move to GPU
        inputs, labels, masks = inputs.to(
            self.device), labels.to(self.device), masks.to(self.device)

        # Forward step
        # We just need the mean (=^ gray color) but the std gives interesting insights
        mean, std = self(inputs)
        return mean, std, labels, masks, data_counter

    def predict(self, image, patch_size, overlap):
        """Performs denoising on the given image using the specified patch size and overlap.
        
        Arguments:
            image {(H, W)} -- the image to denoise
            patch_size {int} -- the patch size to use for prediction on individual pixels
            overlap {int} -- overlap between patches
        
        Returns:
            mean -- the predicted pixel values 
                    (mean because the network actually estimates a normal distribution)
            std  -- the standard deviation for each pixel
        """
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

        mean, std = self(inputs)

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
