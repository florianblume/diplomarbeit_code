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
        amalgamted_image = torch.sum(sub_images, 0) / torch.sum(weights, 0)
        return amalgamted_image, sub_images, weights

    def training_predict(self, train_data, train_data_clean, data_counter, size, box_size, bs):
        # Init Variables
        inputs, labels, masks = self.assemble_training__batch(bs, size, box_size,
                                    data_counter, train_data, train_data_clean)

        # Move to GPU
        inputs, labels, masks = inputs.to(
            self.device), labels.to(self.device), masks.to(self.device)
        
        # Forward step
        amalgamted_image, _, weights = self(inputs)
        
        return amalgamted_image, weights, labels, masks, data_counter

    def predict(self, image, patch_size, overlap):
        # weights for each subnet per pixel
        weights = np.zeros((self.num_subnets, image.shape[0], image.shape[1]))
        sub_images = np.zeros((self.num_subnets, image.shape[0], image.shape[1]))
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
                _amalgamted_image, _sub_images, _weights = self.predict_patch(patch)
                print(_sub_images.squeeze().shape)
                print(_sub_images.squeeze()[:, ovTop:, ovLeft:].shape)
                print(_sub_images[:, ovTop:, ovLeft:].shape)
                print(_sub_images[:, ovTop:, ovLeft:][0].shape)
                print(_sub_images[:, ovTop:, ovLeft:][0].squeeze().shape)
                print(sub_images[:, ymin:ymax, xmin:xmax][:, ovTop:, ovLeft:].shape)
                # Remove unnecessary dimensions that are still there
                _amalgamted_image = _amalgamted_image.squeeze()
                amalgamted_image[ymin:ymax, xmin:xmax][ovTop:, ovLeft:]\
                    = _amalgamted_image[ovTop:, ovLeft:]
                sub_images[:, ymin:ymax, xmin:xmax][:, ovTop:, ovLeft:]\
                    = [_sub_images[:, ovTop:, ovLeft:][0].squeeze(),
                       _sub_images[:, ovTop:, ovLeft:][1].squeeze()]
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

        return amalgamted_image, sub_images, weights

    def predict_patch(self, patch):
        # network expects a batch size even in prediction
        # [batch_size, num_channels, H, W]
        # NOTE the shape has nothing to do with the number of classes yet
        # (Probabilistic Noise2Void) as its the input to the network!
        # TODO use num_channels of our class instead of setting it to 1 manually
        inputs = torch.zeros(1, 1, patch.shape[0], patch.shape[1])
        inputs[0, :, :, :] = util.img_to_tensor(patch)
        # copy to GPU
        inputs = inputs.to(self.device)
        amalgamted_image, sub_images, weights = self(inputs)

        # In contrast to probabilistic N2V we only have one sample
        # weights = samples[0, ...]

        # Get data from GPU
        amalgamted_image = amalgamted_image.cpu().detach().numpy()
        sub_images = sub_images.cpu().detach().numpy()
        sub_images = sub_images.squeeze()
        weights = weights.cpu().detach().numpy()
        # Remove unnecessary dimensions
        weights = np.squeeze(weights)
        return amalgamted_image, sub_images, weights
