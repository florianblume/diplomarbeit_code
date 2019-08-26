import torch

from models.baseline import UNet

class SubUNet(UNet):

    def loss_function_integrated(self, result):
        output, ground_truth, mask = result['output'], result['gt'], result['mask']
        # We only sum along C, H and W to get an output of shape [batch]
        loss = (mask * (ground_truth - output)**2).mean(-1).mean(-1).mean(-1)
        return loss