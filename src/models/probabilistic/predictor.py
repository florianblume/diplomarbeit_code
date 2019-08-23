import os
import numpy as np
import tifffile as tif
import matplotlib.pyplot as plt
import torch

from models import AbstractPredictor
from models.probabilistic import ImageProbabilisticUNet
from models.probabilistic import PixelProbabilisticUNet
from models.probabilistic import SubUNet

class Predictor(AbstractPredictor):

    def _load_net(self):
        weight_mode = self.config['WEIGHT_MODE']
        assert weight_mode in ['image', 'pixel']
        checkpoint = torch.load(self.network_path)
        if weight_mode == 'image':
            Network = ImageProbabilisticUNet
        else:
            Network = PixelProbabilisticUNet
        net = Network(self.config)
        net.load_state_dict(checkpoint['model_state_dict'])
        return net

    def _predict(self, image):
        image = self.net.predict(image)
        return image

class SubnetworkPredictor(AbstractPredictor):

    def __init__(self, config, config_path):
        self.std_list = []
        super(SubnetworkPredictor, self).__init__(config, config_path)

    def _load_net(self):
        checkpoint = torch.load(self.network_path)
        self.config['IS_INTEGRATED'] = False
        self.config['MEAN'] = checkpoint['mean']
        self.config['STD'] = checkpoint['std']
        net = SubUNet(self.config)
        net.load_state_dict(checkpoint['model_state_dict'])
        return net

    def _write_data_to_output_path(self, raw_results, pred_image_filename):
        std = raw_results['std']
        std = std.squeeze()
        if len(std.shape) == 3:
            # RGB image, grayscale otherwise
            std = np.transpose(std, axes=(1, 2, 0))
        
        # No need to save mean here as that's what the AbstractPredictor class
        # does already
        if self.config['WRITE_STD']:
            std_filename = '{}_std'.format(pred_image_filename)
            # Store it once for inspection purposes (tif) and once to
            # view it (png)
            if 'tif' in self.config['OUTPUT_IMAGE_FORMATS']:
                tif.imsave(os.path.join(self.pred_output_path,
                                        std_filename + '.tif'),
                            std.astype(np.float32))
            if 'png' in self.config['OUTPUT_IMAGE_FORMATS']:
                plt.imsave(os.path.join(self.pred_output_path,
                                        std_filename + '.png'),
                            std.astype(np.float32),
                            cmap='gray')

    def _post_process_intermediate_results(self, image_name, raw_results,
                                           processed_results):
        std = raw_results['std']
        # Store to be able to compute the mean later
        self.std_list.append(std)
        mean_std = np.mean(std)
        print("Mean of std: {:.4f}".format(mean_std))
        processed_results[image_name]['mean_std'] = mean_std

    def _post_process_final_results(self, processed_results):
        mean_std = np.mean(self.std_list)
        print('Overall mean of std: {:.4f}'.format(mean_std))
        processed_results['mean_std'] = mean_std