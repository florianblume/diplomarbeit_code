import torch
import torch.nn as nn

from models import AbstractUNet
from models.average import SubUNet

class AbstractWeightNetwork(AbstractUNet):
    """This class encapsulates common operations of the weight networks.
    """

    def __init__(self, num_classes, mean, std, in_channels=1,
                 main_net_depth=1, sub_net_depth=3, num_subnets=2,
                 start_filts=64, up_mode='transpose', merge_mode='add',
                 augment_data=True,
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        self.num_subnets = num_subnets
        self.sub_net_depth = sub_net_depth
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

    @staticmethod
    def loss_function(outputs, labels, masks):
        # This is the leftover of Probabilistic N2V where the network outputs
        # 800 means per pixel instead of only 1
        outs = outputs[:, 0, ...]
        loss = torch.sum(masks * (labels - outs)**2) / torch.sum(masks)
        return loss

    @staticmethod
    def loss_function_with_entropy(outputs, labels, masks, weights, weights_lambda):
        # This is the leftover of Probabilistic N2V where the network outputs
        # 800 means per pixel instead of only 1
        outs = outputs[:, 0, ...]
        loss = torch.sum(masks * (labels - outs)**2) / torch.sum(masks)
        weights = weights / torch.sum(weights, 0)
        weights = torch.mean(weights, 1)
        entropy = -torch.sum(weights * torch.log(weights))
        return loss - weights_lambda * entropy