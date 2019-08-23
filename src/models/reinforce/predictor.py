import torch

from models import AbstractPredictor
from models.reinforce import ReinforceUNet

class Predictor(AbstractPredictor):

    def _load_net(self):
        """
        weight_mode = self.config['WEIGHT_MODE']
        assert weight_mode in ['image', 'pixel']
        checkpoint = torch.load(self.network_path)
        if weight_mode == 'image':
            Network = ImageProbabilisticUNet
        else:
            Network = PixelProbabilisticUNet
        """
        checkpoint = torch.load(self.network_path)
        self.config['MEAN'] = checkpoint['mean']
        self.config['STD'] = checkpoint['std']
        net = ReinforceUNet(self.config)
        state_dict = checkpoint['model_state_dict']
        net.load_state_dict(state_dict)
        return net
