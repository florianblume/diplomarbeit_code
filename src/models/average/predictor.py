import os
import torch
import numpy as np
import tifffile as tif
import matplotlib.pyplot as plt

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
        self.weights_list = []
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
                      in_channels=self.config['IN_CHANNELS'],
                      main_net_depth=self.config['MAIN_NET_DEPTH'],
                      sub_net_depth=self.config['SUB_NET_DEPTH'],
                      num_subnets=self.config['NUM_SUBNETS'])
        net.load_state_dict(checkpoint['model_state_dict'])
        return net

    def _write_data_to_output_path(self, raw_results, pred_image_filename):
        # Squeeze to get rid of batch dimension
        weights = raw_results['weights']
        sub_images = raw_results['sub_outputs']

        if self.config['WRITE_SUBNETWORK_WEIGHTS']:
            # Save weights for evaluation, etc.
            weights_filename = pred_image_filename + '_weights.tif'
            tif.imsave(os.path.join(self.pred_output_path, 
                                    weights_filename), weights)
        if self.config['WRITE_SUBNETWORK_IMAGES']:
            for i, sub_image in enumerate(sub_images):
                # Squeeze to get rid of batch dimension
                sub_image = sub_image.squeeze()
                image_filename = '{}_sub_{}'.format(pred_image_filename, i)
                # Store it once for inspection purposes (tif) and once to
                # view it (png)
                if 'tif' in self.config['OUTPUT_IMAGE_FORMATS']:
                    tif.imsave(os.path.join(self.pred_output_path,
                                            image_filename + '.tif'),
                               sub_image.astype(np.float32))
                if 'png' in self.config['OUTPUT_IMAGE_FORMATS']:
                    plt.imsave(os.path.join(self.pred_output_path,
                                            image_filename + '.png'),
                               sub_image,
                               cmap='gray')

    def _post_process_intermediate_results(self, image_name, raw_results,
                                           processed_results):
        weights = raw_results['weights']
        # Store to be able to compute the mean later
        self.weights_list.append(weights)
        weights_sum = np.sum(weights)
        weights_percentages = [(weight / float(weights_sum)) * 100.0 for
                               weight in weights]
        formatted_weights = Predictor.pretty_string(weights, weights_percentages)
        print("Weights of subnetworks: {}".format(formatted_weights))
        processed_results[image_name]['weights'] = weights.tolist()

    def _post_process_final_results(self, processed_results):
        self.weights_list = np.array(self.weights_list)
        # Compute average over all weights
        weights_average = np.mean(self.weights_list, axis=0)
        weights_average_percentage = weights_average / np.sum(weights_average)
        formatted_weights = Predictor.pretty_string(weights_average,
                                                    weights_average_percentage)
        print('Average weights: {}'.format(formatted_weights))
        processed_results['average_weights'] = weights_average.tolist()
        weights_std = np.std(self.weights_list, axis=0)
        weights_std_string = ', '.join('{:.4f}'.format(std) for std in weights_std)
        print('Weights std: {}'.format(weights_std_string))
        processed_results['weights_std'] = weights_std.tolist()
