import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tif

import util
from data import PredictionDataset
from data.transforms import ConvertToFormat, Normalize, ToTensor

class AbstractPredictor():
    """Class AbstractPredictor is the base class for all predictor classes. It
    manages data loading and network initizliation. Subclasses need to implement
    certain specific functions.
    """

    def __init__(self, config, config_path):
        self.config_path = config_path
        self.config = config
        self._load_config_parameters()
        # Load saved network
        print("Loading network from {}".format(self.network_path))
        self.net = self._load_net()
        self._load_data()
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

    def _load_data(self):
        data_base_dir = self.config['DATA_BASE_DIR']
        data_pred_raw_dirs = []
        for data_pred_raw_dir in self.config['DATA_PRED_RAW_DIRS']:
            data_pred_raw_dirs.append(os.path.join(data_base_dir,
                                                   data_pred_raw_dir))
        if 'DATA_PRED_GT_DIRS' in self.config:
            data_pred_gt_dirs = []
            for data_train_gt_dir in self.config['DATA_PRED_GT_DIRS']:
                data_train_gt_dir = os.path.join(data_base_dir,
                                                 data_train_gt_dir)
                data_pred_gt_dirs.append(data_train_gt_dir)
            self.with_gt = True
        else:
            data_train_gt_dir = None
            self.with_gt = False
        transforms = []
        if 'CONVERT_DATA_TO' in self.config:
            transforms.append(ConvertToFormat(self.config['CONVERT_DATA_TO']))
        transforms.append(Normalize(self.net.mean, self.net.std))
        transforms.append(ToTensor())
        print(data_pred_raw_dirs)
        self.dataset = PredictionDataset(data_pred_raw_dirs,
                                         data_pred_gt_dirs,
                                         transform=transforms)

    def _load_net(self):
        raise NotImplementedError

    def _write_data_to_output_path(self, raw_results, pred_image_filename):
        """This function writes additional data to the specified output path.
        The AbstractPredictor takes care of writing the denoised images that
        the network outputs to the output folder.
        
        Arguments:
            output_path {str} -- the folder where to store prediction
            artifacts
            image_name_base {str} -- the base of the name of the currently
                                     processed image, e.g. 0000_pred
        """
        pass

    def _post_process_intermediate_results(self, image_name, raw_results,
                                           processed_results):
        pass

    def _post_process_final_results(self, processed_results):
        pass

    def predict(self):
        processed_results = {}
        # To compute standard deviation of PSNR, if available
        psnr_values = []
        mse_values = []
        running_times = []

        print('Predicting on {} images.'.format(len(self.dataset)))
        fill = len(str(len(self.dataset)))
        for i, sample in enumerate(self.dataset):
            raw = sample['raw']
            # Do not move after self._predict(im), the subclasses need this info
            # Not the nicest style but works...
            pred_image_filename = '{}_pred'.format(str(i).zfill(fill))

            # This is the actual prediction
            # Permute because our images are already in Pytorch format [C, H, W]
            # And raw[0] because we only have one batch
            print("\nPredicting on image {} with shape {}:"
                  .format(i, list(raw[0].permute(1, 2, 0).size())))
            start = time.time()
            raw_result = self.net.predict(raw)
            end = time.time()
            diff = end - start
            running_times.append(diff)
            print('...took {:.4f} seconds.'.format(diff))
            prediction = raw_result['output']
            #im_filename = 'im_' + str(index).zfill(4) + '.png'
            if self.pred_output_path:

                if prediction.shape[-1] == 1 or len(prediction.shape) == 2:
                    # Gray-scale images
                    prediction = prediction.squeeze()
                    cmap = 'gray'
                else:
                    cmap = None

                if 'tif' in self.config['OUTPUT_IMAGE_FORMATS']:
                    tif.imsave(os.path.join(self.pred_output_path,
                                            pred_image_filename + '.tif'),
                               prediction.astype(np.float32))
                if 'png' in self.config['OUTPUT_IMAGE_FORMATS']:
                    plt.imsave(os.path.join(self.pred_output_path,
                                            pred_image_filename + '.png'),
                               prediction,
                               cmap=cmap)
                self._write_data_to_output_path(raw_result,
                                                pred_image_filename)

            if self.with_gt:
                # We get Pytorch tensors from the dataset
                ground_truth = sample['gt'].cpu().detach().numpy()
                # Ground-truth is normalized in dataset
                ground_truth = util.denormalize(ground_truth,
                                                self.net.mean,
                                                self.net.std)
                psnr = util.PSNR(ground_truth, prediction, 255)
                psnr_values.append(psnr)
                mse = util.MSE(ground_truth, prediction)
                mse_values.append(mse)
                processed_results[pred_image_filename] = {'psnr' : psnr,
                                                          'mse'  : mse}
                raw = raw.cpu().detach().numpy()
                raw = util.denormalize(raw, self.net.mean, self.net.std)
                #TODO 255 might not be correct for SimSim data
                print("PSNR raw {:.4f}".format(util.PSNR(ground_truth, raw, 255)))
                print("PSNR denoised {:.4f}".format(psnr))  # Without info from masked pixel
                print('MSE {:.4f}'.format(mse))
                # Weights etc only get stored if ground-truth data is available
                # This is ok since the subclasses store the weights again as
                # numpy arrays so they do not get lost if there is no ground-truth
                self._post_process_intermediate_results(pred_image_filename,
                                                        raw_result,
                                                        processed_results)

        # To show a visual break before printing averages
        print('')
        avg_runtime = np.mean(running_times)
        print("Average runtime per image: {:.4f}".format(avg_runtime))

        if self.with_gt:
            psnr_average = np.mean(np.array(psnr_values))
            mse_average = np.mean(np.array(mse_values))
            std = np.std(psnr_values)
            print("Average PSNR: {:.4f}".format(psnr_average))
            print("Average MSE: {:.4f}".format(mse_average))
            print("Standard error: {:.4f}".format(std))

            if self.pred_output_path:
                with open(os.path.join(self.pred_output_path, 
                                       'results.json'), 'w') as json_output:
                    processed_results['average_runtime'] = avg_runtime
                    processed_results['psnr_average'] = psnr_average
                    processed_results['mse_average'] = mse_average
                    processed_results['std'] = std
                    self._post_process_final_results(processed_results)
                    # We are pretty printing
                    json.dump(processed_results, json_output, indent=4)
