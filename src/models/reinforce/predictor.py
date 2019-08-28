import os
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tif
import torch

import util

from models import AbstractPredictor
from models.reinforce import ReinforceUNet

class Predictor(AbstractPredictor):

    def __init__(self, config, config_path):
        self.weights_list = []
        super(Predictor, self).__init__(config, config_path)

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

    def _post_process_intermediate_results(self, image_name, raw_results,
                                           processed_results):
        weights = raw_results['action_probs']
        # Store to be able to compute the mean later
        self.weights_list.append(weights)
        weights_sum = np.sum(weights)
        weights_percentages = [(weight / float(weights_sum)) * 100.0 for
                               weight in weights]
        formatted_weights = util.pretty_string_with_percentages(weights,
                                                    weights_percentages)
        print("Weights of subnetworks: {}".format(formatted_weights))
        processed_results[image_name]['weights'] = weights.tolist()

    def _post_process_final_results(self, processed_results):
        weights_list = np.array(self.weights_list)
        # Compute average over all weights
        weights_average = np.mean(weights_list, axis=0)

        weights_average_percentage = weights_average / np.sum(weights_average)
        formatted_weights = util.pretty_string_with_percentages(weights_average,
                                                    weights_average_percentage)
        print('Average weights: {}'.format(formatted_weights))
        processed_results['average_weights'] = weights_average.tolist()
        weights_std = np.std(self.weights_list, axis=0)
        weights_std_string = ', '.join('{:.4f}'.format(std) for std in weights_std)
        print('Weights std: {}'.format(weights_std_string))
        processed_results['weights_std'] = weights_std.tolist()

