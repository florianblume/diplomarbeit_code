import torch
import torch.nn as nn
import numpy as np

import util
from models import conv1x1
from models.average import AbstractWeightNetwork
from models.average import SubUNet

class ImageWeightUNet(AbstractWeightNetwork):
    """This network computes the weights for the subnetworks on a per-image basis.
    This means that each subnetwork gets one weight. The whole output of each
    subnetwork is multiplied by the respective weight. These weighted outputs
    are then summed up and divided by the sum of the weights.
    """

    def __init__(self, num_classes, mean, std, in_channels=1,
                 main_net_depth=1, sub_net_depth=3, num_subnets=2,
                 start_filts=64, up_mode='transpose', merge_mode='add',
                 augment_data=True,
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        super(ImageWeightUNet, self).__init__(num_classes, mean, std,
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
            self.final_ops.append(conv1x1(outs, 1))

    def forward(self, x):
        encoder_outs = []

        # encoder pathway, save outputs for merging
        for i, module in enumerate(self.down_convs):
            x, before_pool = module(x)
            encoder_outs.append(before_pool)

        for i, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i+2)]
            x = module(before_pool, x)

        # x = [num_subnets, batch_size, num_classes, H, W]
        # where num_classes is only used in Probabilistic Noise2Void
        x = torch.stack([final_op(x) for final_op in self.final_ops])
        # We want weights for the subnets, i.e. we need to take the mean
        # (this is the image-wise weight network) along all other dimensions
        # except for the batch dimension, since the individual entries in a
        # batch do not necessarily belong to the same image
        x = torch.mean(x, (2, 3, 4))
        x = torch.exp(x)
        # NOTE we can't put the computation of the subnetworks in here because
        # during inference we need to compute the final output image patch-wise
        # (due to memory on the GPU). This would mean that we compute the weight
        # only based on the patch and not the whole input image.
        return x

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
        # Actually we compute the weights for only one patch here, not the whole
        # image. We could compute the weights for one image if we ran the network
        # on all images that are fused in the batch that has been created above.
        # This could fail if the images are too large. Also, we would have to
        # obtain the indices of the images used in the batch to retrieve them.
        # We assume that training the network on patches and during prediction
        # averaging the weights works just as well since the network performs
        # some kind of averaging already but only on the patch-level.
        # [subnets, batch]
        weights = self(inputs)
        # [subnets, batch, num_classes, H, W]
        sub_outputs = torch.stack([sub(inputs) for sub in self.subnets])
        # We compute weights on a per-image basis -> one weight for one image
        # If we were to perform Probabilistic Noise2Void we would need
        # the reconstruction of based on the output samples and weight this
        # reconstruction.
        # torch is not able to broadcast weights directly thus we have to
        # expand the shape of the weights.
        weights = weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        outputs = torch.sum(weights * sub_outputs, 0)

        # [batch, H, W] / [batch]
        # i.e. we divide the patches that are comprised of weighted sums
        # of the outputs of the subnetworks by the sum of the weights
        outputs /= torch.sum(weights, 0)
        
        return outputs, weights, labels, masks, data_counter

    def predict(self, image, patch_size, overlap):
        # weights for each subnet for the whole imagess
        weights = []
        # sub_images = [subnets, H, W]
        sub_images = np.zeros((self.num_subnets, image.shape[0], image.shape[1]))
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
                # NOTE: We save all weights for all patches and average later ->
                # this is not identical to computing one weight based on the
                # whole image but we have use tiling due to memory constraints.
                # In training we take the exp of the mean so in order to average
                # over the patches we take the log here and then exp later.
                weights.append(np.log(self.predict_patch(patch)))
                sub_outputs = np.array([subnet.predict_patch(patch)\
                                        for subnet in self.subnets])
                sub_images[:, ymin:ymax, xmin:xmax][:, ovTop:, ovLeft:]\
                    = sub_outputs[:, ovTop:, ovLeft:]
                ymin = ymin-overlap+patch_size
                ymax = ymin+patch_size
                ovTop = overlap//2
            ymin = 0
            ymax = patch_size
            xmin = xmin-overlap+patch_size
            xmax = xmin+patch_size
            ovLeft = overlap//2

        # [num_subnets] = weights for the whole image for each subnet
        weights = np.mean(np.array(weights), axis=0)
        # NOTE: we do exp here since this normally happens in training but
        # we "undid" it above taking the log to be able to average the weights
        # across the patches.
        weights = np.exp(weights)
        # Expand dimensions to match the image's
        weights = np.expand_dims(np.expand_dims(weights, -1), -1)
        # sub_images * weights = [num_subnets, H, W] * [num_subnets]
        mult = sub_images * weights
        weighted_average = np.sum(mult, axis=0) / np.sum(weights)
        return weighted_average, mult, weights

    def predict_patch(self, patch):
        inputs = torch.zeros(1, 1, patch.shape[0], patch.shape[1])
        inputs[0, :, :, :] = util.img_to_tensor(patch)
        # copy to GPU
        inputs = inputs.to(self.device)
        output = self(inputs)

        # In contrast to probabilistic N2V we only have one sample
        # weights = samples[0, ...]

        # Get data from GPU
        weights = output.cpu().detach().numpy()
        # Remove unnecessary dimensions
        weights = np.squeeze(weights)
        return weights
