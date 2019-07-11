import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import os

from models import abstract_predictor
from . import baseline_network

class Predictor(abstract_predictor.AbstractPredictor):

    def _load_net(self):
        checkpoint = torch.load(self.network_path)
        self.net = baseline_network.UNet(1, checkpoint['mean'], 
                        checkpoint['std'], depth=self.config['DEPTH'])
        state_dict = checkpoint['model_state_dict']
        # Legacy weight adjustment
        if 'conv_final.weight' in state_dict:
            state_dict['network_head.weight'] = state_dict['conv_final.weight']
            state_dict['network_head.bias'] = state_dict['conv_final.bias']
            del state_dict['conv_final.weight']
            del state_dict['conv_final.bias']
        self.net.load_state_dict(state_dict)

    def _predict(self, image):
        image = self.net.predict(image, self.ps, self.overlap)
        return image