import os
import torch
import numpy as np

from models import AbstractPredictor
from models.average import ImageWeightUNet
from models.average import PixelWeightUNet

class Predictor(AbstractPredictor):
    """Predictor for averaging network.
    """

    @staticmethod
    def pretty_string(weights, percentages):
        # Print it in a nice format like 15.6754 (3.1203%)
        string = ', '.join('{:.4f} ({:.4f}%)'.format(*t) for
                           t in zip(weights, percentages))
        formatted_weights = string.format(weights, percentages)
        return formatted_weights

    def __init__(self, config, config_path):
        self.weights = None
        self.weights_list = []
        self.weights_percentages = None
        super(Predictor, self).__init__(config, config_path)

    def _load_net(self):
        weight_mode = self.config['WEIGHT_MODE']
        assert weight_mode in ['image', 'pixel']
        checkpoint = torch.load(self.network_path)
        if weight_mode == 'image':
            Network = ImageWeightUNet
        else:
            Network = PixelWeightUNet
        # mean and std get set in the load_state_dict function
        net = Network(self.config['NUM_CLASSES'],
                      checkpoint['mean'], checkpoint['std'],
                      main_net_depth=self.config['MAIN_NET_DEPTH'],
                      sub_net_depth=self.config['SUB_NET_DEPTH'],
                      num_subnets=self.config['NUM_SUBNETS'])
        net.load_state_dict(checkpoint['model_state_dict'])
        return net

    def _predict(self, image):
        image, weights = self.net.predict(image, self.ps, self.overlap)
        self.weights = weights.squeeze()
        self.weights_list.append(self.weights)
        weights_sum = np.sum(weights)
        weights_percentages = [weight / float(weights_sum) for
                               weight in self.weights]
        formatted_weights = Predictor.pretty_string(self.weights, weights_percentages)
        print("Weights of subnetworks: {}".format(formatted_weights))
        # Save weights for evaluation, etc.
        weights_filename = self.pred_image_filename_base + '_weights.npy'
        np.save(os.path.join(self.pred_output_path, weights_filename), self.weights)
        return image

    def _store_additional_intermediate_results(self, image_name, results):
        results[image_name]['weights'] = self.weights.tolist()

    def _store_additional_results(self, results):
        self.weights_list = np.array(self.weights_list)
        # Compute average over all weights
        weights_average = np.mean(self.weights_list, axis=0)
        weights_average_percentage = weights_average / np.sum(weights_average)
        formatted_weights = Predictor.pretty_string(weights_average, 
                                                    weights_average_percentage)
        print('Average weights {}'.format(formatted_weights))
        results['average_weights'] = weights_average.tolist()
