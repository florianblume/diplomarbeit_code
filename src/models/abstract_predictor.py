import os
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

import util
from data import dataloader


class AbstractPredictor():

    def __init__(self, config):
        self.config_path = os.path.dirname(config)
        self.config = util.load_config(config)

        #Initialize member variables that are getting set later
        self.net = None
        self.network_path = None
        self.experiment_base_path = None
        self.pred_output_path = None
        self.ps = None
        self.overlap = None
        self.loader = None
        self.data_gt = None
        self.data_test = None

    def _load_config_parameters(self):
        if 'PRED_NETWORK_PATH' not in self.config:
            raise 'No checkpoint path specified. Please specify the path to ' \
                'a checkpoint of a model that you want to use for prediction.'
        self.experiment_base_path = self.config.get(
            'EXPERIMENT_BASE_PATH', self.config_path)
        if self.experiment_base_path == "":
            self.experiment_base_path = self.config_path
        # don't need config path anymore
        del self.config_path
        self.network_path = os.path.join(
            self.experiment_base_path, self.config['PRED_NETWORK_PATH'])
        self.ps = self.config['PRED_PATCH_SIZE']
        self.overlap = self.config['OVERLAP']

    def _load_net(self):
        raise 'This function needs to be implemented by the subclasses.'

    def _predict(self, image):
        raise 'This function needs to be implemented by the subclasses.'
        # Keep statement so that it is clear to IDEs that the method is going
        # to return something
        return 0

    def predict(self):
        self._load_config_parameters()
        # Load saved network
        print("Loading network from {}".format(self.network_path))
        self.net = None
        self._load_net()
        # To set dropout and batchnormalization (which we don't have but maybe in the future)
        # to inference mode.
        self.net.eval()

        self.loader = dataloader.DataLoader(self.config['DATA_BASE_PATH'])
        self.data_test, self.data_gt = self.loader.load_test_data(
            self.config['DATA_PRED_RAW_PATH'], self.config['DATA_PRED_GT_PATH'],
            self.net.mean, self.net.std, self.config.get('CONVERT_DATA_TO', None))

        if self.data_gt is None:
            print(
                'No ground-truth data provided. Images will be denoised but PSNR is not computable.')

        self.pred_output_path = os.path.join(
            self.experiment_base_path, self.config['PRED_OUTPUT_PATH'])
        if not os.path.exists(self.pred_output_path):
            os.mkdir(self.pred_output_path)

        results = {}
        num_images = self.data_test.shape[0]
        # To compute standard deviation of PSNR, if available
        psnr_values = []
        running_times = []

        print('Predicting on {} images.'.format(num_images))
        for index in range(num_images):

            im = self.data_test[index]

            # This is the actual prediction
            print("Predicting on image {} with shape {}:".format(index, im.shape))
            start = time.time()
            prediction = self._predict(im)
            end = time.time()
            running_times.append(end - start)

            pred_image_filename = 'pred_' + str(index).zfill(4) + '.png'
            # If we want to store the unnoised test image we have to normalize it
            im = util.denormalize(im, self.net.mean, self.net.std)
            #im_filename = 'im_' + str(index).zfill(4) + '.png'
            if self.pred_output_path != "":
                # zfill(4) is enough, probably never going to pedict on more images than 9999
                plt.imsave(os.path.join(self.pred_output_path,
                                        pred_image_filename), prediction, cmap='gray')
                #plt.imsave(os.path.join(self.pred_output_path,
                #                       im_filename), im, cmap='gray')


            # Can be None, if no ground-truth data has been specified
            if self.data_gt is not None:
                # X images get 1 GT image together (due to creation of data set)
                factor = int(self.data_test.shape[0] / self.data_gt.shape[0])
                l = self.data_gt[int(index / factor)]
                psnr = util.PSNR(l, prediction, 255)
                psnr_values.append(psnr)
                print("PSNR raw", util.PSNR(l, im, 255))
                results[pred_image_filename] = psnr
                print("PSNR denoised", psnr)  # Without info from masked pixel

        # To show a visual break before printing averages
        print('')
        avg_runtime = np.mean(running_times)
        print("Average runtime per image:", avg_runtime)
        with open(os.path.join(self.pred_output_path, 'results.json'), 'w') as json_output:
            results['average_runtime'] = avg_runtime

            if self.data_gt is not None:
                average = np.mean(np.array(list(results.values())))
                std = np.std(psnr_values)
                print("Average PSNR:", average)
                print("Standard deviation:", std)
                results['average'] = average
                results['std'] = std
                json.dump(results, json_output)
