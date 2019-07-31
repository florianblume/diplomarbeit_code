import torch

from models.baseline import UNet
from models import AbstractPredictor

class Predictor(AbstractPredictor):
    """The predictor for the baseline network.
    """

    def _load_net(self):
        checkpoint = torch.load(self.network_path)
        net = UNet(self.config['NUM_CLASSES'], checkpoint['mean'],
                        checkpoint['std'], depth=self.config['DEPTH'])
        state_dict = checkpoint['model_state_dict']
        # Legacy weight adjustment
        if 'conv_final.weight' in state_dict:
            state_dict['network_head.weight'] = state_dict['conv_final.weight']
            state_dict['network_head.bias'] = state_dict['conv_final.bias']
            del state_dict['conv_final.weight']
            del state_dict['conv_final.bias']
        net.load_state_dict(state_dict)
        return net

    def _predict(self, image):
        image = self.net.predict(image, self.ps, self.overlap)
        return image

    def _store_additional_intermediate_results(self, image_name, results):
        # Nothing to do here
        pass

    def _store_additional_results(self, results):
        # Nothing to do here
        pass