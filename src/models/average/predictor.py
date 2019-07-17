import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import importlib
import json
import os

from models import abstract_predictor

class Predictor(abstract_predictor.AbstractPredictor):

    def _load_net(self):
        weight_mode = self.config['WEIGHT_MODE']
        assert weight_mode in ['image', 'pixel']
        checkpoint = torch.load(self.network_path)
        if weight_mode == 'image':
            from weight_network import ImageWeightUNet as Network
        else:
            from weight_network import PixelWeightUNet as Network
        net = Network(self.config['NUM_CLASSES'], checkpoint['mean'], 
                        checkpoint['std'], depth=self.config['DEPTH'])
        net.load_state_dict(checkpoint['model_state_dict'])
        return net

    def _predict(self, image):
        image = self.net.predict(image, self.ps, self.overlap)
        return image