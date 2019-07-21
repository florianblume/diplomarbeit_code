import torch

from models import AbstractPredictor
from models.average import ImageWeightUNet
from models.average import PixelWeightUNet

class Predictor(AbstractPredictor):

    def _load_net(self):
        weight_mode = self.config['WEIGHT_MODE']
        assert weight_mode in ['image', 'pixel']
        checkpoint = torch.load(self.network_path)
        if weight_mode == 'image':
            Network = ImageWeightUNet
        else:
            Network = PixelWeightUNet
        self.net = Network(self.config['NUM_CLASSES'], checkpoint['mean'], 
                        checkpoint['std'], depth=self.config['DEPTH'])
        self.net.load_state_dict(checkpoint['model_state_dict'])

    def _predict(self, image):
        image, weights = self.net.predict(image, self.ps, self.overlap)
        print("Weights of subnetworks: {}".format(weights))
        return image