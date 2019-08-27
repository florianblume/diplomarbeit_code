import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import AbstractUNet
from models import conv1x1
from models.q_learning import SubUNet

class ReinforceUNet(AbstractUNet):

    def __init__(self, config):
        self.num_subnets = config['NUM_SUBNETS']
        self.sub_net_depth = config['SUB_NET_DEPTH']
        self.subnet_config = copy.deepcopy(config)
        self.subnet_config['DEPTH'] = config['SUB_NET_DEPTH']

        config['DEPTH'] = config['MAIN_NET_DEPTH']
        super(ReinforceUNet, self).__init__(config)

    def _build_network_head(self, outs):
        self.subnets = nn.ModuleList()
        for _ in range(self.num_subnets):
            self.subnets.append(SubUNet(self.subnet_config))
        self.final_conv = conv1x1(outs, self.num_subnets)

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
        # Do not take softmax here because we are processing images patch-wise
        # during prediction which forces us to average the output after
        # processing the full image and not individually for every patch
        #x = F.softmax(x, dim=1)
        return x

    def get_action(self, output):
        """Returns an action for each entry in the batch in 'output', i.e. an
        array of batch-size with actions as numbers.
        
        Arguments:
            output {torch.tensor} -- the output of the neural network
        
        Returns:
            torch.tensor -- array containting the actions for each entry in the
                            batch
            torch.tensor -- array containting the log probabilities of the
                            selected actions
            torch.tensor -- the softmax probabilities of all batches
        """
        probs = F.softmax(output, dim=1)
        # Sample 1 action according to the action probabilities
        actions = torch.multinomial(probs, 1)
        actions = actions.squeeze(-1)
        log_probs = torch.log(probs[actions])
        return actions, log_probs, probs

    def loss_function(self, result):
        sub_outputs = result['sub_outputs']
        actions = result['actions']
        log_probs = result['log_probs']
        ground_truth = result['gt']
        mask = result['mask']
        sub_losses = torch.stack([subnet.loss_function_integrated(
                                                    {'output' : sub_outputs[i],
                                                     'gt'     : ground_truth,
                                                     'mask'   : mask})
                                  for i, subnet in enumerate(self.subnets)])
        # Put batch first
        sub_losses = sub_losses.transpose(1, 0)
        # shape[0] is batch size
        length = actions.shape[0]
        primary_index = torch.linspace(0, length - 1, length).long().to(self.device)
        primary_index = primary_index.detach()
        actions = actions.detach()
        loss = sub_losses[primary_index, actions] * log_probs[primary_index, actions]
        return -loss.mean()

    def training_predict(self, sample):
        raw, ground_truth, mask = sample['raw'], sample['gt'], sample['mask']
        raw, ground_truth, mask = raw.to(self.device),\
                                   ground_truth.to(self.device),\
                                   mask.to(self.device)
        reinforce = self(raw)
        sub_outputs = torch.stack([subnet(raw) for subnet in self.subnets])
        transposed_sub_outputs = sub_outputs.transpose(1, 0)
        actions, log_probs, _ = self.get_action(reinforce)
        # Detach indices, just to be sure
        actions = actions.detach()
        # shape[0] is batch size
        length = actions.shape[0]
        primary_index = torch.linspace(0, length - 1, length).long().to(self.device)
        return {'output'      : transposed_sub_outputs[primary_index, actions],
                'sub_outputs' : sub_outputs,
                'actions'     : actions,
                'log_probs'   : log_probs,
                'gt'          : ground_truth,
                'mask'        : mask}

    def _pre_process_predict(self, image):
        reinforce = []
        return {'reinforce' : reinforce,
                'image'     : image}

    def _process_patch(self, data, ymin, ymax, xmin, xmax, ovTop, ovLeft):
        image = data['image']
        patch = image[:, :, ymin:ymax, xmin:xmax]
        output = self.predict_patch(patch)
        data['reinforce'].append(output)

    def predict_patch(self, patch):
        inputs = patch.to(self.device)
        output = self(inputs)
        output = output.cpu().detach().numpy()
        return output

    def _post_process_predict(self, result):
        image = result['image']
        reinforce = np.mean(result['reinforce'], axis=0)[0]
        action_probs = np.exp(reinforce) / np.sum(np.exp(reinforce), axis=0)
        action = np.random.choice(self.num_subnets, 1, p=action_probs)
        output = self.subnets[action[0]](image.to(self.device))
        output = output.cpu().detach().numpy()
        # We only have one batch and want channels as last dim
        output = output[0].transpose((1, 2, 0))
        return {'output'       : output,
                'action_probs' : action_probs}
