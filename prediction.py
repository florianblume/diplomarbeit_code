import argparse
import torch
import numpy as np

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
    net = torch.load(network_path)

    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device comes from network if not specified otherwise in its constructor
    #estimate = torch.tensor(25.0/net.std).to(net.device)

    print('Predicting on {} images.'.format(data_test.shape[0]))
    for index in range(data_test.shape[0]):

        im = data_test[index]
        # TODO data_gt[0]? Not [index]?
        l = data_gt[0]
        print("Predicting on image {}".format(index))
        print('Raw image shape {}, ground-truth image shape {}.'.format(im.shape, l.shape))
        ps = config['PRED_PATCH_SIZE']
        overlap = config['OVERLAP']
        means = net.predict(im, ps, overlap)

        im = util.denormalize(im, net.mean, net.std)
        vmi = np.percentile(l, 0.05)
        vma = np.percentile(l, 99.5)
        #print(vmi, vma)

        psnrPrior = util.PSNR(l, means, 255)
        results.append(psnrPrior)

        print("PSNR raw", util.PSNR(l, im, 255))
        print("PSNR denoised", psnrPrior)  # Without info from masked pixel
        print("index", index)

        """
        print(np.min(means), np.max(means))

        plt.imshow(im[200:328, 200:328], cmap='gray', vmin=0, vmax=255)  # GT
        plt.show()

        plt.imshow(means[200:328, 200:328], cmap='gray')
        plt.show()
        """

    print("Avg Prior:", np.mean(np.array(results)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", help="Path to the config.")
    args = parser.parse_args()
    config = util.load_config(args.config)
    main(config)
