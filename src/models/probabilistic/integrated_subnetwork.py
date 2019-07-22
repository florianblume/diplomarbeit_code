import torch

import util
from models.probabilistic import AbstractSubUNet

class IntegratedSubUNet(AbstractSubUNet):
    """This version of the subnetwork can be integrated into the main network
    (Probabilistic UNet). This is why we do not take the log of the loss
    (not a legal operation since the loss is part of an equation).
    """

    def loss_function(mean, std, lables, masks):
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
        c = 1 / (torch.sqrt(2 * math.pi * std**2))
        exp = torch.exp(-(labels - mean)**2)/(2 * std**2))
        # We do not want to sum here as the loss is continued in the main network
        #loss = torch.sum(masks * (c * exp)/torch.sum(masks)
        return loss