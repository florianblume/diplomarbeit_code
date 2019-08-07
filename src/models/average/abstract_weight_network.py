import torch
import torch.nn as nn

from models import AbstractUNet
from models.baseline import UNet as SubUNet

class AbstractWeightNetwork(AbstractUNet):
    """This class encapsulates common operations of the weight networks.
    """

    def __init__(self, num_classes, mean, std, in_channels=1,
                 main_net_depth=1, sub_net_depth=3, num_subnets=2,
                 weight_constraint=None, weights_lambda=0,
                 start_filts=64, up_mode='transpose', 
                 merge_mode='add', augment_data=True,
                 device=torch.device(
                     "cuda:0" if torch.cuda.is_available() else "cpu")):
        self.num_subnets = num_subnets
        self.sub_net_depth = sub_net_depth
        self.weights_lambda = weights_lambda
        self.weight_constraint = weight_constraint

        super(AbstractWeightNetwork, self).__init__(num_classes, mean, std,
                                                    in_channels=in_channels,
                                                    depth=main_net_depth,
                                                    start_filts=start_filts,
                                                    up_mode=up_mode,
                                                    merge_mode=merge_mode,
                                                    augment_data=augment_data,
                                                    device=device)

    def _build_network_head(self, outs):
        self.subnets = nn.ModuleList()
        self.final_ops = nn.ModuleList()
        for _ in range(self.num_subnets):
            # We create each requested subnet
            # TODO Make main and subnets individually configurable
            self.subnets.append(SubUNet(self.num_classes, self.mean, self.std,
                                        in_channels=self.in_channels,
                                        depth=self.sub_net_depth,
                                        start_filts=self.start_filts,
                                        up_mode=self.up_mode,
                                        merge_mode=self.merge_mode,
                                        augment_data=self.augment_data,
                                        device=self.device))
        self._build_final_ops(outs)

    def _build_final_ops(self, outs):
        raise NotImplementedError

    def loss_function(self, result):
        if self.weight_constraint == 'entropy':
            return self.loss_function_with_entropy(result)
        return self.loss_function_without_entropy(result)

    def loss_function_without_entropy(self, result):
        output, ground_truth, mask = result['output'], result['gt'], result['mask']
        mask_sum = torch.sum(mask)
        difference = torch.sum(mask * (ground_truth - output)**2)
        loss = difference / mask_sum
        return loss

    def loss_function_with_entropy(self, result):
        output = result['output']
        ground_truth = result['gt']
        mask = result['mask']
        weights = result['weights']
        loss = torch.sum(mask * (ground_truth - output)**2) / torch.sum(mask)
        weights = weights / torch.sum(weights, 0)
        weights = torch.mean(weights, 1)
        entropy = -torch.sum(weights * torch.log(weights))
        return loss - self.weights_lambda * entropy