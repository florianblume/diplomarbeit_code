import torch
import numpy as np

import util
from models import conv1x1
from models.average import AbstractWeightNetwork

class PixelWeightUNet(AbstractWeightNetwork):
    """This network computes weights for each subnetwork on a per-pixel basis.
    This means that each pixel gets a weights for each subnet. The outputs of
    the subnets are multiplied with their respective weights and added up.
    Afterwards the results is divided by the sum of the weights.
    """

    def __init__(self, num_classes, mean, std, in_channels=1,
                 main_net_depth=1, sub_net_depth=3, num_subnets=2,
                 start_filts=64, up_mode='transpose', merge_mode='add',
                 augment_data=True,
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        super(PixelWeightUNet, self).__init__(num_classes, mean, std,
                                              in_channels=in_channels,
                                              main_net_depth=main_net_depth,
                                              sub_net_depth=sub_net_depth,
                                              num_subnets=num_subnets,
                                              start_filts=start_filts,
                                              up_mode=up_mode,
                                              merge_mode=merge_mode,
                                              augment_data=augment_data,
                                              device=device)

    def _build_final_ops(self, outs):
        for _ in range(self.num_subnets):
            # And for each pixel we output a weight for each subnet
            # 1 is num_classes in the subnets, but we do not sample the weight
            # or something like but just produce one weight per pixel
            self.final_ops.append(conv1x1(outs, 1))

    def forward(self, x):
        encoder_outs = []
        # Compute the outputs of the subnetworks - this is ok here since we
        # do not want weights for the whole image. I.e. we can directly multiply
        # output of the subnetworks and computed weights.
        # [num_subnets, batch_size, num_classes, H, W]
        sub_outputs = torch.stack([sub(x) for sub in self.subnets])

        # encoder pathway, save outputs for merging
        for i, module in enumerate(self.down_convs):
            x, before_pool = module(x)
            encoder_outs.append(before_pool)

        for i, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i+2)]
            x = module(before_pool, x)

        # Compute the pixel-wise weights for the subnetworks
        # x = [num_subnets, batch_size, num_classes, H, W]
        # where num_classes is only used in Probabilistic Noise2Void
        weights = torch.stack([final_op(x) for final_op in self.final_ops])
        weights = torch.exp(weights)
        sub_images = weights * sub_outputs
        # Sum along first axis, i.e. subnets, and divide by sum of weights
        amalgamted_image = torch.sum(sub_images, 1) / torch.sum(weights, 1)
        return amalgamted_image, sub_images, weights

    def training_predict(self, sample):
        raw, ground_truth, mask = sample['raw'], sample['gt'], sample['mask']
        raw, ground_truth, mask = raw.to(self.device),\
                                   ground_truth.to(self.device),\
                                   mask.to(self.device)
        raw, ground_truth, mask = raw.to(
            self.device), ground_truth.to(self.device), mask.to(self.device)
        
        # Forward step
        amalgamted_image, sub_outputs, weights = self(raw)
        
        return  {'gt'         : ground_truth,
                 'mask'       : mask,
                 'output'     : amalgamted_image,
                 'weights'    : weights,
                 'sub_outputs': sub_outputs}

    def predict(self, image, patch_size, overlap):
        # weights for each subnet per pixel
        weights = np.zeros((self.num_subnets, image.shape[0], image.shape[1]))
        sub_outputs = np.zeros((self.num_subnets, image.shape[0], image.shape[1]))
        # sub_images = [C, H, W] since the network performs amalgamation
        # already in its forward method
        amalgamted_image = np.zeros(image.shape)
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
                _amalgamted_image, _sub_outputs, _weights = self.predict_patch(patch)
                _amalgamted_image = _amalgamted_image.squeeze()
                amalgamted_image[ymin:ymax, xmin:xmax][ovTop:, ovLeft:]\
                    = _amalgamted_image[ovTop:, ovLeft:]
                sub_outputs[:, ymin:ymax, xmin:xmax][:, ovTop:, ovLeft:]\
                    = _sub_outputs[:, ovTop:, ovLeft:]
                weights[:, ymin:ymax, xmin:xmax][:, ovTop:, ovLeft:]\
                    = _weights[:, ovTop:, ovLeft:]
                ymin = ymin-overlap+patch_size
                ymax = ymin+patch_size
                ovTop = overlap//2
            ymin = 0
            ymax = patch_size
            xmin = xmin-overlap+patch_size
            xmax = xmin+patch_size
            ovLeft = overlap//2

                # Transepose from [C, H, W] to [H, W, C]
        return {'output'     : amalgamted_image.transpose((1, 2, 0)),
                # Remove padding that was added to match image's dimension
                'weights'    : weights.squeeze(),
                # Transpose from [num_subnets, C, H, W] to [num_subnets, H, W, C]
                'sub_outputs': sub_outputs.transpose((0, 2, 3, 1))}

    def predict_patch(self, patch):
        inputs = torch.zeros((1,) + patch.size())
        inputs[0, :, :, :] = patch
        
        inputs = inputs.to(self.device)
        amalgamted_image, sub_images, weights = self(inputs)
        
        amalgamted_image = amalgamted_image.cpu().detach().numpy()
        sub_images = sub_images.cpu().detach().numpy()
        sub_images = sub_images.squeeze()
        weights = weights.cpu().detach().numpy()
        # Remove unnecessary dimensions
        weights = np.squeeze(weights)
        return amalgamted_image, sub_images, weights
