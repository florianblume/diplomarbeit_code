import torch

from models import AbstractPredictor
from models.probabilistic import ImageProbabilisticUNet
from models.probabilistic import PixelProbabilisticUNet

class Predictor(AbstractPredictor):

    def _load_net(self):
        weight_mode = self.config['WEIGHT_MODE']
        assert weight_mode in ['image', 'pixel']
        checkpoint = torch.load(self.network_path)
        if weight_mode == 'image':
            Network = ImageProbabilisticUNet
        else:
            Network = PixelProbabilisticUNet
        net = Network(checkpoint['mean'], checkpoint['std'], depth=self.config['DEPTH'])
        net.load_state_dict(checkpoint['model_state_dict'])
        return net

    def _predict(self, image):
        image = self.net.predict(image, self.ps, self.overlap)
        return image