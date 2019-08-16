import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import util
from models import AbstractUNet
from models import conv1x1
from models.probabilistic import SubUNet

class ImageProbabilisticUNet(AbstractUNet):

    def __init__(self, config):
        self.sub_net_depth = config['SUB_NET_DEPTH']
        self.num_subnets = config['NUM_SUBNETS']
        self.weight_mode = config['WEIGHT_MODE']
        self.subnet_config = copy.deepcopy(config)
        self.subnet_config['DEPTH'] = config['SUB_NET_DEPTH']

        config['DEPTH'] = config['MAIN_NET_DEPTH']
        super(ImageProbabilisticUNet, self).__init__(config)

    def _build_network_head(self, outs):
        self.subnets = nn.ModuleList()
        for _ in range(self.num_subnets):
            # create the subnets
            self.subnets.append(SubUNet(self.subnet_config))

        if self.num_subnets == 2:
            # In case that we only have two subnets we produce only one output
            # with sigmoid, the second probability is implicit in this case
            count = self.num_subnets - 1
        else:
            count = self.num_subnets

        # create the probability weights, this can be seen as p(z|x), i.e. the probability
        # of a decision given the input image. To obtain a weighted "average" of the
        # predictions of the subnets, we multiply this weight to their output.
        self.weight_probabilities = conv1x1(outs, count)

    def loss_function(self, result):
        # Probabilites for each subnetwork per image, i.e. [batch_size, num_subnets]
        probabilities = result['probabilities']

        # Assemble dict for subnets on-the-fly because list comprehension is
        # faster in Python
        sub_losses = torch.stack([self.subnets[i].loss_function_integrated(
                                    {'mean' : result['sub_outputs'][i][0],
                                     'std'  : result['sub_outputs'][i][1],
                                     'gt'   : result['gt']}
                                  ) for i in range(self.num_subnets)])
        # Multiply over all pixels
        sub_losses = torch.prod(torch.prod(sub_losses, dim=-1), dim=-1)
        # Get rid of unnecessary channel dimension as a leftover from the
        # integrated loss function
        sub_losses = sub_losses.squeeze()
        # Transpose to [batch, subnet]
        sub_losses = sub_losses.transpose(1, 0)
        weighted_losses = probabilities * sub_losses
        # First sum up along "decision"-dimension, then take log, then sum up
        return torch.sum(torch.log(torch.sum(weighted_losses, dim=1)))

    def forward(self, x):
        encoder_outs = []

        # encoder pathway, save outputs for merging
        for i, module in enumerate(self.down_convs):
            x, before_pool = module(x)
            encoder_outs.append(before_pool)

        for i, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i+2)]
            x = module(before_pool, x)

        # Compute the individual weight probabilities
        # One probability (weight) for each subnet
        if self.num_subnets == 2:
            # probabilities are of shape [batch, subnet, H, W] before taking mean
            probabilities = self.weight_probabilities(x)
            probabilities = torch.mean(probabilities, (1, 2, 3))
            first_prob = torch.sigmoid(probabilities)
            outs = torch.stack([first_prob, 1 - first_prob])
            outs = outs.transpose(1, 0)
        else:
            # probabilities are of shape [batch, subnet, H, W] before taking mean
            outs = torch.mean(self.weight_probabilities(x), (-2, -1))
            outs = F.softmax(outs, dim=0)
        return outs

    def training_predict(self, sample):
        raw, ground_truth, mask = sample['raw'], sample['gt'], sample['mask']
        raw, ground_truth, mask = raw.to(self.device),\
                                   ground_truth.to(self.device),\
                                   mask.to(self.device)

        # Forward step
        probabilities = self(raw)
        sub_outputs = [subnet(raw) for subnet in self.subnets]

        return {'probabilities': probabilities,
                'sub_outputs'  : sub_outputs,
                'gt'           : ground_truth,
                'mask'         : mask}

    def _pre_process_predict(self, image):
        # list because we need to average the probabilities for the whole image
        # but can only compute them patch-wise
        probabilities = []
        mean = np.zeros((self.num_subnets,) + image.shape)
        std = np.zeros((self.num_subnets,) + image.shape)
        return {'image'         : image,
                'probabilities' : probabilities,
                'mean'          : mean,
                'std'           : std}

    def _process_patch(self, data, ymin, ymax, xmin, xmax, ovTop, ovLeft):
        image = data['image']
        probabilities = data['probabilities']
        mean = data['mean']
        std = data['std']
        patch = image[:, :, ymin:ymax, xmin:xmax]
        probabilities.append(self.predict_patch(patch))
        sub_outputs = np.array([subnet.predict_patch(patch)\
                                for subnet in self.subnets])
        mean[:, :, :, ymin:ymax, xmin:xmax]\
                [:, :, :, ovTop:, ovLeft:] = sub_outputs[:, 0, :, :, ovTop:, ovLeft:]
        std[:, :, :, ymin:ymax, xmin:xmax]\
                [:, :, :, ovTop:, ovLeft:] = sub_outputs[:, 1, :, :, ovTop:, ovLeft:]

    def predict_patch(self, patch):
        inputs = patch.to(self.device)
        probabilities = self(inputs)
        probabilities = probabilities.cpu().detach().numpy()
        return probabilities

    def _post_process_predict(self, result):
        # [batch, subnet]
        probabilities = result['probabilities']
        mean = result['mean']
        std = result['std']
        # Transpose from [subnet, batch, C, H, W] to [batch, subnet, H, W, C]
        mean = mean.transpose((1, 0, 3, 4, 2))
        std = std.transpose((1, 0, 3, 4, 2))
        # Take the mean of the probabilities computed for the individual patches
        # to obtain the probabilities for the whole images of the subnetworks
        probabilities = np.mean(probabilities, axis=0)
        amalgamted_image = probabilities * mean
        return {'output' : amalgamted_image,
                'mean'   : mean,
                'std'    : std}

class PixelProbabilisticUNet(AbstractUNet):

    def __init__(self, mean, std, in_channels=1,
                 main_net_depth=1, sub_net_depth=3, num_subnets=2,
                 start_filts=64, up_mode='transpose',
                 merge_mode='add', augment_data=True,
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):

        super().__init__(mean=mean, std=std, 
                in_channels=in_channels, main_net_depth=main_net_depth, 
                start_filts=start_filts, up_mode=up_mode, merge_mode=merge_mode, 
                augment_data=augment_data, device=device)

        self.sub_net_depth = sub_net_depth
        self.num_subnets = num_subnets

    def _build_network_head(self, outs):
        self.subnets = nn.ModuleList()
        self.weight_probabilities = nn.ModuleList()

        for _ in range(self.num_subnets):
            # create the two subnets
            self.subnets.append(SubUNet(self.mean, self.std, is_integrated=True,
                                        in_channels=self.in_channels,
                                        depth=self.depth,
                                        start_filts=self.start_filts,
                                        up_mode=self.up_mode,
                                        merge_mode=self.merge_mode,
                                        augment_data=self.augment_data,
                                        device=self.device))
            # create the probability weights, this can be seen as p(z|x), i.e. the probability
            # of a decision given the input image. To obtain a weighted "average" of the
            # predictions of the subnets, we multiply this weight to their output.
            self.weight_probabilities.append(conv1x1(outs, 1))

    def loss_function(self, result):
        output = result['output']
        ground_truth = result['gt']
        mask = result['mask']

    def forward(self, x):
        encoder_outs = []

        # encoder pathway, save outputs for merging
        for i, module in enumerate(self.down_convs):
            x, before_pool = module(x)
            encoder_outs.append(before_pool)

        for i, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i+2)]
            x = module(before_pool, x)

        # Compute the individual weight probabilities
        # One probability (weight) for each subnet
        outs = torch.stack([prob(x) for prob in self.weight_probabilities])
        outs = F.softmax(outs, dim=0)
        return outs

    def training_predict(self, sample):
        raw, ground_truth, mask = sample['raw'], sample['gt'], sample['mask']
        raw, ground_truth, mask = raw.to(self.device),\
                                   ground_truth.to(self.device),\
                                   mask.to(self.device)
        weights = self(raw)
        # TODO move to forward method
        means, stds = [subnet(raw) for subnet in self.subnets]
        # TODO THIS IS NOT CORRECT!!!
        # NOTE this part depends on whether 
        # The correct formula is w_i = 1/sigma_i^2
        # bar(x) = sum(x_i sigma_i^(-2))/sum(sigma_i^(-2)) (mean)
        # sigma_bar(x) = sqrt(1 / sum(sigma_i^(-2)))
        # See also https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Variance_weights
        means_out = np.array([w * m for w in weights for m in means])
        stds_out = np.array([w * m for w in weights for m in stds])
        #TODO this might be wrong
        means_out = np.sum(means_out, axis=0)
        stds_out = np.sum(stds_out, axis=0)
        return {'output' : means_out,
                'mean'   : means_out,
                'std'    : stds_out,
                'gt'     : ground_truth,
                'mask'   : mask}

    def predict(self, image, patch_size, overlap):
        mean = np.zeros(image.shape)
        std = np.zeros(image.shape)
        
        image_height = image.shape[-2]
        image_width = image.shape[-1]

        xmin = 0
        ymin = 0
        xmax = patch_size
        ymax = patch_size
        ovLeft = 0
        while (xmin < image_width):
            ovTop = 0
            while (ymin < image_height):
                patch = image[:, :, ymin:ymax, xmin:xmax]
                mean_, std_ = self.predict_patch(patch)
                mean[:, :, ymin:ymax, xmin:xmax]\
                            [:, :, ovTop:, ovLeft:] = mean_[:, :, ovTop:, ovLeft:]
                std[:, :, ymin:ymax, xmin:xmax]\
                            [:, :, ovTop:, ovLeft:] = std_[:, :, ovTop:, ovLeft:]
                ymin = ymin-overlap+patch_size
                ymax = ymin+patch_size
                ovTop = overlap//2
            ymin = 0
            ymax = patch_size
            xmin = xmin-overlap+patch_size
            xmax = xmin+patch_size
            ovLeft = overlap//2
        return {'output' : mean,
                'mean'   : mean,
                'std'    : std}

    def predict_patch(self, patch):
        inputs = patch.to(self.device)

        mean, std = self(inputs)

        mean = mean.cpu().detach().numpy()
        std = std.cpu().detach().numpy()

        mean = util.denormalize(mean, self.mean, self.std)
        std = util.denormalize(std, self.mean, self.std)
        return mean, std