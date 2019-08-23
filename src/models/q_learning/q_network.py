import copy
import torch
import torch.nn as nn
import numpy as np

from models import AbstractUNet
from models import conv1x1
from models.baseline import UNet as SubUNet

class QUNet(AbstractUNet):

    def __init__(self, config):
        self.num_subnets = config['NUM_SUBNETS']
        self.sub_net_depth = config['SUB_NET_DEPTH']
        self.epsilon = config['EPSILON']
        self.subnet_config = copy.deepcopy(config)
        self.subnet_config['DEPTH'] = config['SUB_NET_DEPTH']

        config['DEPTH'] = config['MAIN_NET_DEPTH']
        super(QUNet, self).__init__(config)

    def _build_network_head(self, outs):
        self.subnets = nn.ModuleList()
        for _ in range(self.num_subnets):
            # We create each requested subnet
            # TODO Make main and subnets individually configurable
            self.subnets.append(SubUNet(self.subnet_config))
        self.final_conv = conv1x1(outs, self.num_subnets)

    def loss_function(self, result):
        q_values = result['q_values']
        sub_outputs = result['sub_outputs']
        ground_truth = result['gt']
        mask = result['mask']
        sub_losses = torch.stack([subnet.loss_function({'output' : sub_outputs[i],
                                                        'gt'     : ground_truth,
                                                        'mask'   : mask})
                                  for i, subnet in enumerate(self.subnets)])
        # We use the sub losses to compute the probabilities of the updates
        # (i.e. their weights), along these probabilities we do not want a
        # gradient to be computed
        detached_sub_losses = sub_losses.detach()
        # We try to approximate the loss functions of the subnetworks through
        # the q values
        loss_diff = (q_values - detached_sub_losses)**2
        # The probability of each subnetwork to be chosen in the random case
        # With probability 1 - epsilon we draw a subnetwork completely at random
        random_probability = (1.0 / self.num_subnets) * (1 - self.epsilon)
        # shape[0] is batch size
        shape = (mask.shape[0], self.num_subnets)
        probabilities = torch.tensor(random_probability).repeat(shape)
        probabilities[q_values == torch.max(q_values)] += self.epsilon
        # We do not want any gradient along the probabilities
        probabilities = probabilities.detach()
        # Full param update inlcudes the losses of the subnetworks scaled
        # by their respective probability
        print(probabilities)
        return loss_diff * probabilities + sub_losses * probabilities

    def forward(self, x):
        encoder_outs = []

        # encoder pathway, save outputs for merging
        for i, module in enumerate(self.down_convs):
            x, before_pool = module(x)
            encoder_outs.append(before_pool)

        for i, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i+2)]
            x = module(before_pool, x)

        x = self.final_conv(x)
        # Mean along H and W dim -> [batch, subnet]
        x = torch.mean(x, dim=(2, 3))
        return x

    def training_predict(self, sample):
        raw, ground_truth, mask = sample['raw'], sample['gt'], sample['mask']
        raw, ground_truth, mask = raw.to(self.device),\
                                   ground_truth.to(self.device),\
                                   mask.to(self.device)
        q_values = self(raw)
        sub_outputs = torch.stack([subnet(raw) for subnet in self.subnets])
        return {'q_values'    : q_values,
                'sub_outputs' : sub_outputs,
                'gt'          : ground_truth,
                'mask'        : mask}

    def _pre_process_predict(self, image):
        q_values = []
        sub_outputs = np.zeros(image.shape)
        return {'q_values'     : q_values,
                'sub_outputs' : sub_outputs}

    def _process_patch(self, data, ymin, ymax, xmin, xmax, ovTop, ovLeft):
        q_values = data['q_values']
        sub_outputs = data['sub_outputs']
        image = data['image']
        patch = image[ymin:ymax, xmin:xmax]
        output = self.predict_patch(patch)
        q_values.append(output[ovTop:, ovLeft:])
        _sub_outputs = np.stack([subnet.predict_patch(patch) for subnet in self.subnets])
        sub_outputs[:, :, :, ymin:ymax, xmin:xmax][:, :, :, ovTop:, ovLeft:]\
            = _sub_outputs[:, :, :, ovTop:, ovLeft:]

    def _post_process_predict(self, result):
        q_values = result['q_values']
        sub_outputs = result['sub_outputs']
        index = np.where(q_values == np.min(q_values))
        return {'output'      : sub_outputs[index],
                'q_values'    : q_values,
                'sub_outputs' : sub_outputs}

    def predict_patch(self, patch):
        inputs = patch.to(self.device)
        weigths = self(inputs)
        weigths = weigths.cpu().detach().numpy()
        return weigths
