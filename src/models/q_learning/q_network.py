import copy
import torch
import torch.nn as nn
import numpy as np

from models import AbstractUNet
from models import conv1x1
from models.q_learning import SubUNet

class QUNet(AbstractUNet):

    def __init__(self, config):
        self.num_subnets = config['NUM_SUBNETS']
        self.sub_net_depth = config['SUB_NET_DEPTH']
        self.epsilon = config['EPSILON']
        self.subnet_config = copy.deepcopy(config)
        self.subnet_config['DEPTH'] = config['SUB_NET_DEPTH']
        if config.get('FREEZE_SUBNETS', False):
            self.subnet_config['FREEZE_WEIGHTS'] = True

        config['DEPTH'] = config['MAIN_NET_DEPTH']
        super(QUNet, self).__init__(config)

    def _build_network_head(self, outs):
        self.subnets = nn.ModuleList()
        for i in range(self.num_subnets):
            # We create each requested subnet
            self.subnets.append(SubUNet(self.subnet_config))
        self.final_conv = conv1x1(outs, self.num_subnets)

    def params_for_key(self, key):
        if key == "LEARNING_RATE_MAIN_NET":
            # Filter out params of subnets
            return [param for name, param in self.named_parameters() if 'subnets' not in name]
        if key == "LEARNING_RATE_SUB_NETS":
            return [param for name, param in self.named_parameters() if 'subnets' in name]
        raise ValueError("Unrecognized learning rate.")

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

    def loss_function(self, result):
        q_values = result['q_values']
        sub_outputs = result['sub_outputs']
        ground_truth = result['gt']
        mask = result['mask']
        sub_losses = torch.stack([subnet.loss_function_integrated(
                                                    {'output' : sub_outputs[i],
                                                     'gt'     : ground_truth,
                                                     'mask'   : mask})
                                  for i, subnet in enumerate(self.subnets)])
        sub_losses = sub_losses.transpose(1, 0)
        # We use the sub losses to compute the probabilities of the updates
        # (i.e. their weights), along these probabilities we do not want a
        # gradient to be computed
        detached_sub_losses = sub_losses.detach()
        # We try to approximate the loss functions of the subnetworks through
        # the q values
        loss_diff = (q_values - detached_sub_losses)**2
        # The probability of each subnetwork to be chosen in the random case
        # With probability 1 - epsilon we draw a subnetwork completely at random
        random_probability = (1.0 / self.num_subnets) * self.epsilon
        # shape[0] is batch size
        shape = (mask.shape[0], self.num_subnets)
        probabilities = torch.tensor(random_probability).repeat(shape).to(self.device)
        # Construct index array to add epsilon probability to the probabilities
        # that correspond to the subnetworks with the higher Q-value -> these
        # are the ones that get the actual update
        _, indices = torch.min(q_values, dim=1)
        length = indices.shape[0]
        primary_index = torch.linspace(0, length - 1, length).long().to(self.device)
        probabilities[primary_index, indices] += (1 - self.epsilon)
        # We do not want any gradient along the probabilities
        probabilities = probabilities.detach()
        # Full param update inlcudes the losses of the subnetworks scaled
        # by their respective probability
        loss = loss_diff * probabilities + sub_losses * probabilities
        return loss.mean()

    def training_predict(self, sample):
        raw, ground_truth, mask = sample['raw'], sample['gt'], sample['mask']
        raw, ground_truth, mask = raw.to(self.device),\
                                   ground_truth.to(self.device),\
                                   mask.to(self.device)
        q_values = self(raw)
        sub_outputs = torch.stack([subnet(raw) for subnet in self.subnets])
        
        _, indices = torch.min(q_values, dim=1)
        length = indices.shape[0]
        primary_index = torch.linspace(0, length - 1, length).long().to(self.device)
        transposed_sub_outputs = sub_outputs.transpose(1, 0)

        return {'output'      : transposed_sub_outputs[primary_index, indices],
                'q_values'    : q_values,
                'indices'     : indices,
                'sub_outputs' : sub_outputs,
                'gt'          : ground_truth,
                'mask'        : mask}

    def _pre_process_predict(self, image):
        q_values = []
        return {'image'       : image,
                'q_values'    : q_values}

    def _process_patch(self, data, ymin, ymax, xmin, xmax, ovTop, ovLeft):
        q_values = data['q_values']
        image = data['image']
        patch = image[:, :, ymin:ymax, xmin:xmax]
        output = self.predict_patch(patch)
        # output[0] because we only have batch size 1
        q_values.append(output[0])

    def predict_patch(self, patch):
        inputs = patch.to(self.device)
        weigths = self(inputs)
        weigths = weigths.cpu().detach().numpy()
        return weigths

    def _post_process_predict(self, result):
        image = result['image']
        q_values = np.array(result['q_values'])
        print('Patch weight std', np.std(q_values, axis=0))
        q_values = np.mean(q_values, axis=0)
        index = np.where(q_values == np.min(q_values))
        # Somehow np.where produces a tuple with an array inside
        index = index[0][0]
        output = self.subnets[index].predict(image)
        return {'output'      : output['output'],
                'q_values'    : q_values}
