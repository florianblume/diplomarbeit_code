import torch

from models.baseline import UNet
from models import AbstractPredictor

class Predictor(AbstractPredictor):
    """The predictor for the baseline network.
    """

    def _load_net(self):
        checkpoint = torch.load(self.network_path)
        self.config['MEAN'] = checkpoint['mean']
        self.config['STD'] = checkpoint['std']
        net = UNet(self.config)
        state_dict = checkpoint['model_state_dict']
        net.load_state_dict(state_dict)
        return net
        