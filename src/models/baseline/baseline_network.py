import torch
import numpy as np
import tifffile as tif

import util
from models import AbstractUNet
from models import conv1x1

class UNet(AbstractUNet):
    """Baseline network that is only a simple UNet. This network is used to
    compare the performance of more complex network structures.
    """
    
    def _build_network_head(self, outs):
        self.network_head = conv1x1(outs, self.in_channels)

    @staticmethod
    def loss_function(result):
        output, ground_truth, mask = result['output'], result['gt'], result['mask']
        mask_sum = torch.sum(mask, dim=(1, 2, 3))
        difference = torch.sum(mask * (ground_truth - output)**2, dim=(1, 2, 3))
        # NOTE: if the config for the network is wrong and no hot pixels were
        # selected to be replaced during N2V training, we divide by 0 because
        # of mask_sum resulting in NaN loss.
        loss = difference / mask_sum
        loss = torch.mean(loss)
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

    def training_predict(self, sample):
        raw, gt, mask = sample['raw'], sample['gt'], sample['mask']

        # Move to GPU
        raw, gt, mask = raw.to(
            self.device), gt.to(self.device), mask.to(self.device)

        # Forward step
        output = self(raw)
        return {'output' : output,
                'gt'     : gt,
                'mask'   : mask}

    def predict(self, image, patch_size, overlap):
        output = np.zeros(image.shape)
        # We have to use tiling because of memory constraints on the GPU
        xmin = 0
        ymin = 0
        xmax = patch_size
        ymax = patch_size
        ovLeft = 0
        # Image is in [C, H, W] shape
        while (xmin < image.shape[2]):
            ovTop = 0
            while (ymin < image.shape[1]):
                a = self.predict_patch(image[:, ymin:ymax, xmin:xmax])
                output[:, ymin:ymax, xmin:xmax][:, ovTop:,
                                                ovLeft:] = a[:, ovTop:, ovLeft:]
                ymin = ymin-overlap+patch_size
                ymax = ymin+patch_size
                ovTop = overlap//2
            ymin = 0
            ymax = patch_size
            xmin = xmin-overlap+patch_size
            xmax = xmin+patch_size
            ovLeft = overlap//2
        # Transpose image back from [C, H, W] Pytorch format to [H, W, C]
        output = np.transpose(output, (1, 2, 0))
        return {'output' : output}

    def predict_patch(self, patch):
        # Add one dimension for the batch size which is 1 during prediction
        inputs = torch.zeros((1,) + patch.size())
        inputs[0, :, :, :] = patch

        # copy to GPU
        inputs = inputs.to(self.device)
        output = self(inputs)[0]

        # Get data from GPU
        image = output.cpu().detach().numpy()
        # Denormalize
        image = util.denormalize(image, self.mean, self.std)
        return image
