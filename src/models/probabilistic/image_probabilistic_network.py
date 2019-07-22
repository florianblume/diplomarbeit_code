import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import util
from models import AbstractUNet
from models import conv1x1
from models.probabilistic import IntegratedSubUNet

class ImageProbabilisticUNet(AbstractUNet):

    def __init__(self, num_classes, mean, std, in_channels=1, 
                 main_net_depth=1, sub_net_depth=3, num_subnets=2,
                 start_filts=64, up_mode='transpose',
                 merge_mode='add', augment_data=True,
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):

        super().__init__(num_classes=num_classes, mean=mean, std=std, 
                in_channels=in_channels, main_net_depth=main_net_depth, 
                start_filts=start_filts, up_mode=up_mode, merge_mode=merge_mode, 
                augment_data=augment_data, device=device)

        self.sub_net_depth = sub_net_depth
        self.num_subnets = num_subnets

    def _build_network_head(self, outs):

        self.subnets = nn.ModuleList()
        self.weight_probabilities = nn.ModuleList()

        for i in range(self.num_subnets):
            # create the two subnets
            self.subnets.append(IntegratedSubUNet(self.num_classes, self.mean, 
                                        self.std, depth=self.sub_net_depth))
            # create the probability weights, this can be seen as p(z|x), i.e. the probability
            # of a decision given the input image. To obtain a weighted "average" of the
            # predictions of the subnets, we multiply this weight to their output.
            self.weight_probabilities.append(conv1x1(outs, self.num_classes))

    @staticmethod
    def loss_function(outputs, labels, masks):
        outs = outputs[:, 0, ...]
        raise 'Need to implement!'

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
        outs = torch.stack([prob(x) for prob in self.weight_probabilities])
        outs = F.softmax(outs, dim=0)
        return outs

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
        weights = self(inputs)
        means, stds = [subnet(inputs) for subnet in self.subnets]

        # TODO THIS IS NOT CORRECT!!!
        # The correct formula is w_i = 1/sigma_i^2
        # bar(x) = sum(x_i sigma_i^(-2))/sum(sigma_i^(-2)) (mean)
        # sigma_bar(x) = sqrt(1 / sum(sigma_i^(-2)))
        # See also https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Variance_weights
        means_out = np.array([w * m for w in weights for m in means])
        stds_out = np.array([w * m for w in weights for m in stds])
        #TODO this might be wrong
        means_out = np.sum(means_out, axis=0)
        stds_out = np.sum(stds_out, axis=0)
        return means_out, stds_out, labels, masks, data_counter

    def predict(self, image, patch_size, overlap):
        # TODO
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
        return means, stds

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