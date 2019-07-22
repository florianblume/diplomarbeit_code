import torch

import util
from models.probabilistic import AbstractSubUNet

class StandaloneSubUNet(AbstractSubUNet):
    """This version of the subnetwork has a different loss function than the
    integrated version as we can take the log of the loss to simplify the 
    equation. This means that we cannot use it from within the main network
    (Probabilistic UNet).
    """

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
        c = torch.log(1 / (torch.sqrt(2 * math.pi * std**2)))
        # exp is no exponential here because we take the log of the loss
        exp = -(labels - mean)**2)/(2 * std**2)
        loss = torch.sum(masks * (c + exp)/torch.sum(masks)
        return loss
