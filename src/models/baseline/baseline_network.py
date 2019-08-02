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

    counter = 0
    
    def _build_network_head(self, outs):
        # TODO actually output should be self.num_channels x self.num_classes
        # to account for RGB images and Probabilistic Noise2Void
        self.network_head = conv1x1(outs, 1)

    @staticmethod
    def loss_function(result):
        output, gt, mask = result['output'], result['gt'], result['mask']
        mask_sum = torch.sum(mask, dim=0)
        loss = torch.sum(mask * (gt - output)**2, dim=0) / mask_sum
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
        tif.imsave('raw{}.tif'.format(UNet.counter), raw.cpu().detach().numpy())
        tif.imsave('gt{}.tif'.format(UNet.counter), gt.cpu().detach().numpy())
        UNet.counter += 1

        # Move to GPU
        raw, gt, mask = raw.to(
            self.device), gt.to(self.device), mask.to(self.device)

        # Forward step
        output = self(raw)
        return {'output' : output,
                'gt'     : gt,
                'mask'   : mask}

    def predict(self, image, patch_size, overlap):
        result = np.zeros(image.shape)
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
                result[ymin:ymax, xmin:xmax][ovTop:,
                                            ovLeft:] = a[ovTop:, ovLeft:]
                ymin = ymin-overlap+patch_size
                ymax = ymin+patch_size
                ovTop = overlap//2
            ymin = 0
            ymax = patch_size
            xmin = xmin-overlap+patch_size
            xmax = xmin+patch_size
            ovLeft = overlap//2
        return {'result' : result}

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

        # Get data from GPU
        image = output.cpu().detach().numpy()
        # Reshape to 2D images and remove padding
        image = image.squeeze()
        # Denormalize
        image = util.denormalize(image, self.mean, self.std)
        return image
