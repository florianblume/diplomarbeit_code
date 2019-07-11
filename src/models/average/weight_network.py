import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from torch.nn import init
import numpy as np

import util
import abstract_network
from subnetwork import SubUNet

class ImageWeightUNet(abstract_network.AbstractUNet):
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
        super(ImageWeightUNet, self).__init__(num_classes, mean, std, in_channels, 
                main_net_depth, start_filts, up_mode, merge_mode, augment_data, device)

        self.num_subnets = num_subnets
        self.sub_net_depth = sub_net_depth

    def _build_network_head(self, outs):
        self.subnets = nn.ModuleList()
        self.final_op = nn.ModuleList()

        for i in range(self.num_subnets):
            # We create each requested subnet
            self.subnets.append(SubUNet(num_classes, mean, std, depth=sub_net_depth))
            # And we output a weight for each subnet based on the whole image
            # That's why we employ torch.sum
            self.final_op.append(torch.sum(conv1x1(outs, self.num_classes)))

    @staticmethod
    def loss_function(outputs, labels, masks):
        # This is the leftover of Probabilistic N2V where the network outputs
        # 800 means per pixel instead of only 1
        outs = outputs[:, 0, ...]
        loss = torch.sum(masks * (labels - outs)**2) / torch.sum(masks)
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
        x = self.final_op(x)
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
        # Actually we compute the weights for only one patch here, not the whole
        # image. We could compute the weights for one image if we ran the network
        # on all images that are fused in the batch that has been created above.
        # This could fail if the images are too large. Also, we would have to
        # obtain the indices of the images used in the batch to retrieve them.
        # We assume that training the network on patches and during prediction
        # averaging the weights works just as well since the network performs
        # some kind of averaging already but only on the patch-level.
        # [batch, subnets]
        weights = self(inputs)
        # [subnets, batch, H, W]
        sub_outputs = [sub(inputs) for sub in self.subnets]
        # [batch, subnets, H, W]
        sub_outputs = np.transpose(sub_outputs, axes=(1, 0, 2, 3))
        # All pixels in each sub_output get multiplied by the same weight
        # sum([batch, subnets] x [batch, subnets, H, W]) = [batch, H, W]
        outputs = np.sum(weights * sub_outputs, axis=1)
        # [batch, H, W] / [batch]
        outputs /= np.sum(weights, axis=1)
        
        return sub_outputs, weights, labels, masks, data_counter

    def predict(self, image, patch_size, overlap):
        # weights for each subnet for the whole imagess
        weights = []
        # sub_images = [subnets, H, W]
        sub_images = np.zeros((self.num_subnets, image.shape))
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
                # We save all weights for all patches and average later
                weights.append(self.predict_patch(patch))
                sub_images[:, ymin:ymax, xmin:xmax][ovTop:, ovLeft:] 
                    = [subnet.predict_patch(patch) for subnet in self.subnets][:, ovTop:, ovLeft:] 
                ymin = ymin-overlap+patch_size
                ymax = ymin+patch_size
                ovTop = overlap//2
            ymin = 0
            ymax = patch_size
            xmin = xmin-overlap+patch_size
            xmax = xmin+patch_size
            ovLeft = overlap//2

        # [image.shape[0] x image.shape[1] = num_patches, num_subnets]
        weights = np.array(weights)
        # [num_subnets] = weights for the whole image for each subnet
        weights = np.mean(weights, axis=0)
        # sub_images * weights = [num_subnets, H, W] * [num_subnets]
        weighted_average = np.sum(sub_images * weights, axis=0) / np.sum(weights)

        return weighted_average, weights

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

        # In contrast to the other networks we only have a single output number
        # per patch and not an output of shape [H, W] that's why we need to
        # write (1, 0, 2) instead of (1, 0, 2, 3)
        samples = (output).permute(1, 0, 2)

        # In contrast to probabilistic N2V we only have one sample
        weights = samples[0, ...]

        # Get data from GPU
        weights = weights.cpu().detach().numpy()

        return weights

class PixelWeightUNet(nn.Module):
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
                 main_net_depth=1, sub_net_depth=3, num_subnets=2,
                 start_filts=64, up_mode='transpose', merge_mode='add',
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
        super(PixelWeightUNet, self).__init__(num_classes, mean, std, in_channels, 
                main_net_depth, start_filts, up_mode, merge_mode, augment_data, device)

        self.num_subnets = num_subnets
        self.sub_net_depth = sub_net_depth

    def _build_network_head(self, outs):
        self.subnets = nn.ModuleList()
        self.final_op = nn.ModuleList()

        for i in range(self.num_subnets):
            # We create each requested subnet
            self.subnets.append(SubUNet(num_classes, mean, std, depth=sub_net_depth))
            # And for each pixel we output a weight for each subnet
            self.final_op.append(conv1x1(outs, self.num_classes))


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
        x = self.final_op(x)
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
        # Each pixel has its individual weight
        # [batch, subnets, H, W]
        weights = self(inputs)
        # [subnets, batch, H, W]
        sub_outputs = [sub(inputs) for sub in self.subnets]
        # [batch, subnets, H, W]
        sub_outputs = np.transpose(sub_outputs, axes=(1, 0, 2, 3))
        # [batch, subnets, H, W] x [batch, subnets, H, W]
        outputs = np.sum(weights * sub_outputs, axis=1)
        # TODO check if this actually works correctly
        # This works in pixel-wise and image-wise case
        outputs /= np.sum(weights, axis=1)
        
        return sub_outputs, weights, labels, masks, data_counter

    def predict(self, image, patch_size, overlap):
        # weights for each subnet per pixel
        weights = np.zeros((self.num_subnets, image.shape))
        # sub_images = [subnets, H, W]
        sub_images = np.zeros((self.num_subnets, image.shape))
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
                # self.predict_patch(patch) returns [num_subnets, H, W]
                # i.e. one weight for each pixel
                weights[:, ymin:ymax, xmin:xmax][ovTop:, ovLeft:]  
                    = self.predict_patch(patch)
                sub_images[:, ymin:ymax, xmin:xmax][ovTop:, ovLeft:] 
                    = [subnet.predict_patch(patch) for subnet in self.subnets][:, ovTop:, ovLeft:] 
                ymin = ymin-overlap+patch_size
                ymax = ymin+patch_size
                ovTop = overlap//2
            ymin = 0
            ymax = patch_size
            xmin = xmin-overlap+patch_size
            xmax = xmin+patch_size
            ovLeft = overlap//2

        # weights * sub_images = [num_subnets, H, W] x [num_subnets, H, W]
        weighted_average = np.sum(weights * sub_images, axis=0) / np.sum(weights, axis=0)

        return weighted_average, weights

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
        weights = samples[0, ...]

        # Get data from GPU
        weights = weights.cpu().detach().numpy()

        # Reshape to 2D images and remove padding
        weights.shape = (output.shape[2], output.shape[3])

        # No denormalizing here as we output the weights
        return weights