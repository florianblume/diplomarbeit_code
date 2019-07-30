import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tif

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
        pred_output_path = self.config.get('PRED_OUTPUT_PATH', None)
        if pred_output_path:
            self.pred_output_path = os.path.join(self.experiment_base_path,
                                                 pred_output_path)
        else:
            self.pred_output_path = None
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
        self.network_path = os.path.join(
            self.experiment_base_path, self.config['PRED_NETWORK_PATH'])
        self.ps = self.config['PRED_PATCH_SIZE']
        self.overlap = self.config['OVERLAP']

    def _load_net(self):
        raise NotImplementedError

    def _predict(self, image):
        raise NotImplementedError

    def _write_data_to_output_path(self, output_path, image_name_base):
        """This function writes additional data to the specified output path.
        The AbstractPredictor takes care of writing the denoised images that
        the network outputs to the output folder.
        
        Arguments:
            output_path {str} -- the folder where to store prediction
            artifacts
            image_name_base {str} -- the base of the name of the currently 
                                     processed image, e.g. 0000_pred
        """
        raise NotImplementedError

    def _store_additional_intermediate_results(self, image_name, results):
        raise NotImplementedError

    def _store_additional_results(self, results):
        raise NotImplementedError

    def predict(self):
        self.data_test, self.data_gt = self.loader.load_test_data(
            self.config['DATA_PRED_RAW_PATH'], self.config['DATA_PRED_GT_PATH'],
            self.net.mean, self.net.std, self.config.get('CONVERT_DATA_TO', None))
        print(self.data_test.shape)
        print(self.data_gt.shape)

        if self.data_gt is None:
            print(
                'No ground-truth data provided. Images will be denoised but PSNR is not computable.')

        results = {}
        num_images = self.data_test.shape[0]
        # To compute standard deviation of PSNR, if available
        psnr_values = []
        mse_values = []
        running_times = []

        print('Predicting on {} images.'.format(num_images))
        for index in range(num_images):
            im = self.data_test[index]
            # Do not move after self._predict(im), the subclasses need this info
            # Not the nicest style but works...
            pred_image_filename = '{}_pred'.format(str(index).zfill(len(str(abs(self.data_test.shape[0])))))

            # This is the actual prediction
            print("\nPredicting on image {} with shape {}:".format(index, im.shape))
            start = time.time()
            prediction = self._predict(im)
            end = time.time()
            diff = end - start
            running_times.append(diff)
            print('...took {:.4f} seconds.'.format(diff))

            # If we want to store the unnoised test image we have to normalize it
            im = util.denormalize(im, self.net.mean, self.net.std)
            #im_filename = 'im_' + str(index).zfill(4) + '.png'
            if self.pred_output_path:
                if 'tif' in self.config['OUTPUT_IMAGE_FORMATS']:
                    tif.imsave(os.path.join(self.pred_output_path,
                                            pred_image_filename + '.tif'),
                               prediction.astype(np.float32))
                if 'png' in self.config['OUTPUT_IMAGE_FORMATS']:
                    plt.imsave(os.path.join(self.pred_output_path,
                                            pred_image_filename + '.png'),
                               prediction,
                               cmap='gray')
                # Uncomment if you want to see the raw image
                #plt.imsave(os.path.join(self.pred_output_path,
                #                       im_filename), im, cmap='gray')
                self._write_data_to_output_path(self.pred_output_path,
                                                pred_image_filename)


            # Can be None, if no ground-truth data has been specified
            if self.data_gt is not None:
                # X images get 1 GT image together (due to creation of data set)
                factor = int(self.data_test.shape[0] / self.data_gt.shape[0])
                ground_truth = self.data_gt[int(index / factor)]
                psnr = util.PSNR(ground_truth, prediction, 255)
                psnr_values.append(psnr)
                mse = util.MSE(ground_truth, prediction)
                mse_values.append(mse)
                results[pred_image_filename] = {'psnr' : psnr,
                                                'mse'  : mse}
                print("PSNR raw {:.4f}".format(util.PSNR(ground_truth, im, 255)))
                print("PSNR denoised {:.4f}".format(psnr))  # Without info from masked pixel
                print('MSE {:.4f}'.format(mse))
                # Weights etc only get stored if ground-truth data is available
                # This is ok since the subclasses store the weights again as
                # numpy arrays so they do not get lost if there is no ground-truth
                self._store_additional_intermediate_results(pred_image_filename,
                                                            results)

        # To show a visual break before printing averages
        print('')
        avg_runtime = np.mean(running_times)
        print("Average runtime per image: {:.4f}".format(avg_runtime))

        if self.data_gt is not None:
            psnr_average = np.mean(np.array(psnr_values))
            mse_average = np.mean(np.array(mse_values))
            std = np.std(psnr_values)
            print("Average PSNR: {:.4f}".format(psnr_average))
            print("Average MSE: {:.4f}".format(mse_average))
            print("Standard error: {:.4f}".format(std))

            if self.pred_output_path:
                with open(os.path.join(self.pred_output_path, 
                                       'results.json'), 'w') as json_output:
                    results['average_runtime'] = avg_runtime
                    results['psnr_average'] = psnr_average
                    results['mse_average'] = mse_average
                    results['std'] = std
                    self._store_additional_results(results)
                    # We are pretty printing
                    json.dump(results, json_output, indent=4)
