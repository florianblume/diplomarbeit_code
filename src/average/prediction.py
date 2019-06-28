import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import importlib
import json
import os

# Import it dynamically later
#import network
import dataloader
import util

def main(config):

    # Load saved network
    if 'PRED_NETWORK_PATH' not in config:
        raise 'No checkpoint path specified. Please specify the path to ' \
                'a checkpoint of a model that you want to use for prediction.'

    experiment_base_path = config['EXPERIMENT_BASE_PATH']
    network_path = os.path.join(experiment_base_path, config['PRED_NETWORK_PATH'])
    print("Loading network from {}".format(network_path))
    # mean and std will be set by state dict appropriately
    checkpoint = torch.load(network_path)
    network = importlib.import_module(config['NETWORK'] + ".network")
    net = network.UNet(1, checkpoint['mean'], checkpoint['std'], depth=config['DEPTH'])
    net.load_state_dict(checkpoint['model_state_dict'])
    # To set dropout and batchnormalization (which we don't have but maybe in the future)
    # to inference mode.
    net.eval()

    loader = dataloader.DataLoader(config['DATA_BASE_PATH'])
    data_test, data_gt = loader.load_test_data(
        config['DATA_PRED_RAW_PATH'], config['DATA_PRED_GT_PATH'],
        net.mean, net.std)

    if data_gt is None:
        print('No ground-truth data provided. Images will be denoised but PSNR is not computable.')

    ps = config['PRED_PATCH_SIZE']
    overlap = config['OVERLAP']
    pred_output_path = os.path.join(experiment_base_path, config['PRED_OUTPUT_PATH'])
    if not os.path.exists(pred_output_path):
        os.mkdir(pred_output_path)
    
    results = {}
    num_images = data_test.shape[0]

    print('Predicting on {} images.'.format(num_images))
    for index in range(num_images):

        im = data_test[index]
        print("Predicting on image {} with shape {}:".format(index, im.shape))
        means = net.predict(im, ps, overlap)

        pred_image_filename = 'pred_' + str(index).zfill(4) + '.png'
        if pred_output_path != "":
            # zfill(4) is enough, probably never going to pedict on more images than 9999
            plt.imsave(os.path.join(pred_output_path, pred_image_filename), means)

        im = util.denormalize(im, net.mean, net.std)
        #vmi = np.percentile(l, 0.05)
        #vma = np.percentile(l, 99.5)
        #print(vmi, vma)

        # Can be None, if no ground-truth data has been specified
        if data_gt is not None:
            # X images get 1 GT image together (due to creation of data set)
            factor = int(data_test.shape[0] / data_gt.shape[0])
            l = data_gt[int(index / factor)]
            psnr = util.PSNR(l, means, 255)
            print("PSNR raw", util.PSNR(l, im, 255))
            results[pred_image_filename] = psnr
            print("PSNR denoised", psnr)  # Without info from masked pixel

    if data_gt is not None:
        average = np.mean(np.array(list(results.values())))
        print("Average PSNR:", average)
        with open(os.path.join(pred_output_path, 'results.json'), 'w') as json_output:
            results['average'] = average
            json.dump(results, json_output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", help="Path to the config.")
    args = parser.parse_args()
    config = util.load_config(args.config)
    main(config)
