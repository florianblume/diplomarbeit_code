import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt

import util
from data import dataloader


class AbstractPredictor():
    """Class AbstractPredictor is the base class for all predictor classes. It
    manages data loading and network initizliation. Subclasses need to implement
    certain specific functions.
    """

    def __init__(self, config, config_path):
        self.config_path = config_path
        self.config = config
        self._load_config_parameters()
        self.loader = dataloader.DataLoader(self.config['DATA_BASE_PATH'])
        self.net = None
        # Load saved network
        print("Loading network from {}".format(self.network_path))
        self.net = self._load_net()
        # To set dropout and batchnormalization (which we don't have but maybe in the future)
        # to inference mode.
        self.net.eval()
        # Subclasses need to access this that's why we store it on the class
        self.pred_output_path = os.path.join(
            self.experiment_base_path, self.config['PRED_OUTPUT_PATH'])
        if not os.path.exists(self.pred_output_path):
            os.mkdir(self.pred_output_path)

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
        raise NotImplementedError

    def _predict(self, image):
        raise NotImplementedError

    def predict(self):
        self.data_test, self.data_gt = self.loader.load_test_data(
            self.config['DATA_PRED_RAW_PATH'], self.config['DATA_PRED_GT_PATH'],
            self.net.mean, self.net.std, self.config.get('CONVERT_DATA_TO', None))

        if self.data_gt is None:
            print(
                'No ground-truth data provided. Images will be denoised but PSNR is not computable.')

        results = {}
        num_images = self.data_test.shape[0]
        # To compute standard deviation of PSNR, if available
        psnr_values = []
        running_times = []

        print('Predicting on {} images.'.format(num_images))
        for index in range(num_images):
            im = self.data_test[index]
            # Do not move after self._predict(im), the subclasses need this info
            # Not the nicest style but works...
            self.pred_image_filename_base = 'pred_' + str(index).zfill(4)
            self.pred_image_filename = self.pred_image_filename_base + '.png'

            # This is the actual prediction
            print("\nPredicting on image {} with shape {}:".format(index, im.shape))
            start = time.time()
            prediction = self._predict(im)
            end = time.time()
            running_times.append(end - start)

            # If we want to store the unnoised test image we have to normalize it
            im = util.denormalize(im, self.net.mean, self.net.std)
            im_filename = 'im_' + str(index).zfill(4) + '.png'
            if self.pred_output_path != "":
                # zfill(4) is enough, probably never going to pedict on more images than 9999
                plt.imsave(os.path.join(self.pred_output_path,
                                        self.pred_image_filename), prediction, cmap='gray')
                # Uncomment if you want to see the ground-truth images
                #plt.imsave(os.path.join(self.pred_output_path,
                #                       im_filename), im, cmap='gray')


            # Can be None, if no ground-truth data has been specified
            if self.data_gt is not None:
                # X images get 1 GT image together (due to creation of data set)
                factor = int(self.data_test.shape[0] / self.data_gt.shape[0])
                l = self.data_gt[int(index / factor)]
                psnr = util.PSNR(l, prediction, 255)
                psnr_values.append(psnr)
                print("PSNR raw {:.4f}".format(util.PSNR(l, im, 255)))
                results[self.pred_image_filename] = psnr
                print("PSNR denoised {:.4f}".format(psnr))  # Without info from masked pixel

        # To show a visual break before printing averages
        print('')
        avg_runtime = np.mean(running_times)
        print("Average runtime per image: {:.4f}".format(avg_runtime))
        with open(os.path.join(self.pred_output_path, 'results.json'), 'w') as json_output:
            results['average_runtime'] = avg_runtime

            if self.data_gt is not None:
                average = np.mean(np.array(list(results.values())))
                std = np.std(psnr_values)
                print("Average PSNR: {:.4f}".format(average))
                print("Standard deviation: {:.4f}".format(std))
                results['average'] = average
                results['std'] = std
                json.dump(results, json_output)
