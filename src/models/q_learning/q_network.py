import torch
import numpy as np

import util

from models import AbstractUNet

class QUNet(AbstractUNet):

    def __init__(self,config):
        super(QUNet, self).__init__(config)

    def _build_network_head(self, outs):
        pass

    def loss_function(self, result):
        output = result['output']
        ground_truth = result['gt']
        mask = result['mask']
        loss = torch.sum(mask*(ground_truth - output)**2)/torch.sum(mask)
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

    def training_predict(self, sample):
        raw, ground_truth, mask = sample['raw'], sample['gt'], sample['mask']
        raw, ground_truth, mask = raw.to(self.device),\
                                   ground_truth.to(self.device),\
                                   mask.to(self.device)
        outputs = self(raw)
        return {}

    def predict(self, image):
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
        inputs = patch.to(self.device)
        output = self(inputs)

        output = output.cpu().detach().numpy()

        output = util.denormalize(output, self.mean, self.std)
        return means
