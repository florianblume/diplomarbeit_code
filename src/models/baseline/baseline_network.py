import torch
import numpy as np

import util
from models import AbstractUNet
from models import conv1x1

class UNet(AbstractUNet):
    """Baseline network that is only a simple UNet. This network is used to
    compare the performance of more complex network structures.
    """
    
    def _build_network_head(self, outs):
        self.network_head = conv1x1(outs, self.in_channels)

    def loss_function(self, result):
        output, ground_truth, mask = result['output'], result['gt'], result['mask']
        mask_sum = torch.sum(mask)
        difference = torch.sum(mask * (ground_truth - output)**2)
        # NOTE: if the config for the network is wrong and no hot pixels were
        # selected to be replaced during N2V training, we get a NaN loss because
        # of division by 0 (mask_sum).
        loss = difference / mask_sum
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
        raw, ground_truth, mask = sample['raw'], sample['gt'], sample['mask']
        raw, ground_truth, mask = raw.to(self.device),\
                                   ground_truth.to(self.device),\
                                   mask.to(self.device)
        output = self(raw)
        return {'output' : output,
                'gt'     : ground_truth,
                'mask'   : mask}

    def _pre_process_predict(self, image):
        return {'image'  : image,
                'output' : np.zeros(image.shape)}

    def _process_patch(self, data, ymin, ymax, xmin, xmax, ovTop, ovLeft):
        image = data['image']
        output = data['output']
        
        patch = image[:, :, ymin:ymax, xmin:xmax]
        prediction = self.predict_patch(patch)
        output[:, :, ymin:ymax, xmin:xmax][:, :, ovTop:,
                                    ovLeft:] = prediction[:, :, ovTop:, ovLeft:]
        # Actually we have a pointer to the data but to make the code more
        # understandable we add this here
        data['output'] = output
        return data

    def predict_patch(self, patch):
        inputs = patch.to(self.device)
        prediction = self(inputs)
        prediction = prediction.cpu().detach().numpy()
        prediction = util.denormalize(prediction, self.mean, self.std)
        return prediction

    def _post_process_predict(self, result):
        output = result['output']
        # Transpose image back from [batch_size, C, H, W] Pytorch format to
        # [batch_size, H, W, C]
        output = np.transpose(output, (0, 2, 3, 1))
        # Remove batch size dim
        output = output[0]
        return {'output' : output}
