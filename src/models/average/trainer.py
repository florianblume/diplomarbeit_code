import torch
import numpy as np
import os

import util
from models import AbstractTrainer
from models.average import ImageWeightUNet
from models.average import PixelWeightUNet

class Trainer(AbstractTrainer):

    def __init__(self, config, config_path):
        super(Trainer, self).__init__(config, config_path)
        self.train_loss = 0.0
        self.train_losses = []
        self.val_loss = 0.0
        self.weight_constraint =\
                self.config.get('WEIGHT_CONSTRAINT', None)
        self.weight_constraint_lambda =\
                self.config.get('WEIGHT_CONSTRAINT_LAMBDA', 0)

    def _load_network(self):
        if self.config['WEIGHT_MODE'] == 'image':
            self.weight_mode = 'image'
            Network = ImageWeightUNet
        elif self.config['WEIGHT_MODE'] == 'pixel':
            self.weight_mode = 'pixel'
            Network = PixelWeightUNet
        else:
            raise 'Invalid config value for \"weight_mode\".'

        return Network(self.config['NUM_CLASSES'], self.dataset.mean,
                       self.dataset.std,
                       main_net_depth=self.config['MAIN_NET_DEPTH'],
                       sub_net_depth=self.config['SUB_NET_DEPTH'],
                       num_subnets=self.config['NUM_SUBNETS'],
                       augment_data=self.config['AUGMENT_DATA'])

    def _write_custom_tensorboard_data_for_example(self, 
                                                   example_result, 
                                                   example_index):
        # In case that we predict the weights for the whole image we only
        # want to plot their histograms. In case that we predict the weights
        # on a per-pixel basis we store it as an image.
        weights = example_result['weights']
        for i in range(weights.shape[0]):
            weights_name = 'example_{}-weights_subnet_{}'.format(example_index, i)
            if self.weight_mode == 'image':
                self.writer.add_histogram(weights_name,
                                          weights,
                                          self.current_epoch,
                                          bins='auto')
            elif self.weight_mode == 'pixel':
                # Normalize weights
                weights = weights / np.sum(weights, axis=0)
                weights = weights * 255
                weights = weights.astype(np.uint8)
                self.writer.add_image(weights_name, weights, self.current_epoch)
            else:
                raise ValueError('Unkown weight mode.')
