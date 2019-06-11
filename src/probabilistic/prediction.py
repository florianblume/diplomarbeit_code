import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import os

from network import DownConv
from network import UpConv
from network import UNet
import network
import dataloader
import util

def main(config):

    # Load saved network
    if 'PRED_NETWORK_PATH' not in config:
        raise 'No checkpoint path specified. Please specify the path to ' \
                'a checkpoint of a model that you want to use for prediction.'

    network_path = config['PRED_NETWORK_PATH']
    print("Loading network from {}".format(network_path))
    # mean and std will be set by state dict appropriately
    checkpoint = torch.load(network_path)
    net = UNet(1, checkpoint['mean'], checkpoint['std'], depth=config['DEPTH'])
    net.load_state_dict(checkpoint['model_state_dict'])
    # To set dropout and batchnormalization (which we don't have but maybe in the future)
    # to inference mode.
    net.eval()

    loader = dataloader.DataLoader(config['DATA_BASE_PATH'])
    data_test, data_gt = loader.load_test_data(
        config['DATA_PRED_RAW_PATH'], config['DATA_PRED_GT_PATH'],
        net.mean, net.std)

    ps = config['PRED_PATCH_SIZE']
    overlap = config['OVERLAP']
    pred_output_path = config['PRED_OUTPUT_PATH']
    if not os.path.exists(pred_output_path):
        os.mkdir(pred_output_path)
    
    results = []
    num_images = data_test.shape[0]

    print('Predicting on {} images.'.format(num_images))
    for index in range(num_images):

        im = data_test[index]
        print("Predicting on image {} with shape {}:".format(index, im.shape))
        means = net.predict(im, ps, overlap)

        if pred_output_path != "":
            # zfill(4) is enough, probably never going to pedict on more images than 9999
            plt.imsave(os.path.join(pred_output_path, 'pred_' + str(index).zfill(4) + '.png'), means)

        im = util.denormalize(im, net.mean, net.std)
        #vmi = np.percentile(l, 0.05)
        #vma = np.percentile(l, 99.5)
        #print(vmi, vma)

        # Can be None, if no ground-truth data has been specified
        if data_gt is not None:
            #TODO we always compare against the first GT image?
            l = data_gt[0]
            psnrPrior = util.PSNR(l, means, 255)
            print("PSNR raw", util.PSNR(l, im, 255))

        results.append(psnrPrior)
        print("PSNR denoised", psnrPrior)  # Without info from masked pixel

    print("Avg Prior:", np.mean(np.array(results)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", help="Path to the config.")
    args = parser.parse_args()
    config = util.load_config(args.config)
    main(config)
