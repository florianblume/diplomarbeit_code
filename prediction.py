import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

from network import DownConv
from network import UpConv
from network import UNet
import network
import dataloader
import util

def main(config):
    #from scipy import ndimage, misc

    results = []

    loader = dataloader.DataLoader(config['DATA_BASE_PATH'])
    data_test, data_gt = loader.load_test_data(
        config['DATA_PRED_RAW_PATH'], config['DATA_PRED_GT_PATH'])

    # Load saved network
    network_path = config['SAVED_NETWORK_PATH']
    print("Loading network from {}".format(network_path))
    net = torch.load("best.net")
    net.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(net.device)
    #net = UNet(1, 11.042194366455078, 23.338916778564453, depth=config['DEPTH'])
    #net.load_state_dict(torch.load(network_path))
    # To set dropout and batchnormalization (which we don't have but maybe in the future)
    # to inference mode.
    net.eval()

    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device comes from network if not specified otherwise in its constructor
    #estimate = torch.tensor(25.0/net.std).to(net.device)

    ps = config['PRED_PATCH_SIZE']
    overlap = config['OVERLAP']

    num_images = data_test.shape[0]
    print('Predicting on {} images.'.format(num_images))
    for index in range(num_images):

        im = data_test[index]
        l = data_gt[0]
        print("Predicting on image {}:".format(index))
        print('Raw image shape {}, ground-truth image shape {}.'.format(im.shape, l.shape))
        means = net.predict(im, ps, overlap)

        im = util.denormalize(im, 11.042194, 23.338917)
        vmi = np.percentile(l, 0.05)
        vma = np.percentile(l, 99.5)
        #print(vmi, vma)

        psnrPrior = util.PSNR(l, means, 255)
        results.append(psnrPrior)

        print("PSNR raw", util.PSNR(l, im, 255))
        print("PSNR denoised", psnrPrior)  # Without info from masked pixel

    print("Avg Prior:", np.mean(np.array(results)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", help="Path to the config.")
    args = parser.parse_args()
    config = util.load_config(args.config)
    main(config)
