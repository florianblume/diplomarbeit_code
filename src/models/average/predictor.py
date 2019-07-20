import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import importlib
import json
import os

from models import abstract_predictor
from models.average import weight_network

class Predictor(abstract_predictor.AbstractPredictor):

    def _load_net(self):
        weight_mode = self.config['WEIGHT_MODE']
        assert weight_mode in ['image', 'pixel']
        checkpoint = torch.load(self.network_path)
        if weight_mode == 'image':
            Network = weight_network.ImageWeightUNet
        else:
            Network = weight_network.PixelWeightUNet
        self.net = Network(self.config['NUM_CLASSES'], checkpoint['mean'], 
                        checkpoint['std'], depth=self.config['DEPTH'])
        self.net.load_state_dict(checkpoint['model_state_dict'])

    def _predict(self, image):
        image, weights = self.net.predict(image, self.ps, self.overlap)
        print("Weights of subnetworks: {}".format(weights))
        return image