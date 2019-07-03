import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
import importlib
import numpy as np
import matplotlib.pyplot as plt

from data import dataloader
import util

class Predictor():

    def __init__(self, config):
        self.config_path = os.path.dirname(config)
        self.config = util.load_config(config)

    def _load_config_parameters(self):
        if 'PRED_NETWORK_PATH' not in self.config:
            raise 'No checkpoint path specified. Please specify the path to ' \
                    'a checkpoint of a model that you want to use for prediction.'
        self.experiment_base_path = self.config.get('EXPERIMENT_BASE_PATH', self.config_path)
        if self.experiment_base_path == "":
            self.experiment_base_path = self.config_path
        # don't need config path anymore
        del self.config_path
        self.network_path = os.path.join(self.experiment_base_path, self.config['PRED_NETWORK_PATH'])
        self.ps = self.config['PRED_PATCH_SIZE']
        self.overlap = self.config['OVERLAP']

    def _predict(self, image):
        raise 'This function needs to be implemented by the subclasses.'
        # Keep statement so that it is clear to IDEs that the method is going
        # to return something
        return 0

    def predict(self):
        self._load_config_parameters()
        # Load saved network
        print("Loading network from {}".format(self.network_path))
        # mean and std will be set by state dict appropriately
        checkpoint = torch.load(self.network_path)
        network = importlib.import_module(self.config['NETWORK'] + ".network")
        self.net = network.UNet(1, checkpoint['mean'], checkpoint['std'], depth=self.config['DEPTH'])
        self.net.load_state_dict(checkpoint['model_state_dict'])
        # To set dropout and batchnormalization (which we don't have but maybe in the future)
        # to inference mode.
        self.net.eval()

        self.loader = dataloader.DataLoader(self.config['DATA_BASE_PATH'])
        self.data_test, self.data_gt = self.loader.load_test_data(
            self.config['DATA_PRED_RAW_PATH'], self.config['DATA_PRED_GT_PATH'],
            self.net.mean, self.net.std)

        if self.data_gt is None:
            print('No ground-truth data provided. Images will be denoised but PSNR is not computable.')

        self.pred_output_path = os.path.join(self.experiment_base_path, self.config['PRED_OUTPUT_PATH'])
        if not os.path.exists(self.pred_output_path):
            os.mkdir(self.pred_output_path)
        
        results = {}
        num_images = self.data_test.shape[0]

        print('Predicting on {} images.'.format(num_images))
        for index in range(num_images):

            im = self.data_test[index]
            print("Predicting on image {} with shape {}:".format(index, im.shape))
            prediction = self._predict(im)

            pred_image_filename = 'pred_' + str(index).zfill(4) + '.png'
            if self.pred_output_path != "":
                # zfill(4) is enough, probably never going to pedict on more images than 9999
                plt.imsave(os.path.join(self.pred_output_path, pred_image_filename), prediction)

            im = util.denormalize(im, self.net.mean, self.net.std)

            # Can be None, if no ground-truth data has been specified
            if self.data_gt is not None:
                # X images get 1 GT image together (due to creation of data set)
                factor = int(self.data_test.shape[0] / self.data_gt.shape[0])
                l = self.data_gt[int(index / factor)]
                psnr = util.PSNR(l, prediction, 255)
                print("PSNR raw", util.PSNR(l, im, 255))
                results[pred_image_filename] = psnr
                print("PSNR denoised", psnr)  # Without info from masked pixel

        if self.data_gt is not None:
            average = np.mean(np.array(list(results.values())))
            print("Average PSNR:", average)
            with open(os.path.join(self.pred_output_path, 'results.json'), 'w') as json_output:
                results['average'] = average
                json.dump(results, json_output)