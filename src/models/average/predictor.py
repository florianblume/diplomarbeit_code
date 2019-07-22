import os
import torch
import numpy as np

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
        # mean and std get set in the load_state_dict function
        self.net = Network(self.config['NUM_CLASSES'],
                    checkpoint['mean'], checkpoint['std'],
                    main_net_depth=self.config['MAIN_NET_DEPTH'],
                    sub_net_depth=self.config['SUB_NET_DEPTH'],
                    num_subnets=self.config['NUM_SUBNETS'])
        self.net.load_state_dict(checkpoint['model_state_dict'])

    def _predict(self, image):
        image, weights = self.net.predict(image, self.ps, self.overlap)
        weights = weights.squeeze()
        format_list = ['{:.4f}' for weight in weights]
        s = ', '.join(format_list)
        formatted_weights = s.format(*weights)
        print("Weights of subnetworks: {}".format(formatted_weights))
        # Save weights for evaluation, etc.
        weights_filename = self.pred_image_filename_base + '_weights.npy'
        np.save(os.path.join(self.pred_output_path, weights_filename), weights)
        return image