import argparse   
import torch
import numpy as np

import network
import dataloader
import util

def main(network_config, data_config):
    #from scipy import ndimage, misc

    results = []

    loader = dataloader.DataLoader(data_config['DATA_BASE_PATH'])
    data_test, data_gt = loader.load_test_data(data_config['DATA_RAW_PATH'], data_config['DATA_GT_PATH'])

    # Load saved network
    network_path = data_config['SAVED_NETWORK_PATH']
    print("Loading network from {}".format(network_path))
    net = torch.load(network_path)

    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device comes from network if not specified otherwise in its constructor
    #estimate = torch.tensor(25.0/net.std).to(net.device)

    print('Predicting on {} images.'.format(data_test.shape[0]))
    for index in range(data_test.shape[0]):

        im = data_test[index]
        l = data_gt[0]
        print('Raw image shape {}, ground-truth image shape {}.'.format(im.shape, l.shape))
        means = np.zeros(im.shape)
        mseEst = np.zeros(im.shape)

        # We have to use tiling because of memory constraints on the GPU
        ps = data_config['PATCH_SIZE']
        overlap = data_config['OVERLAP']
        xmin = 0
        ymin = 0
        xmax = ps
        ymax = ps
        ovLeft = 0
        while (xmin < im.shape[1]):
            ovTop = 0
            while (ymin < im.shape[0]):
                a = net.predict(im[ymin:ymax, xmin:xmax])
                means[ymin:ymax, xmin:xmax][ovTop:, ovLeft:] = a[ovTop:, ovLeft:]
                ymin = ymin-overlap+ps
                ymax = ymin+ps
                ovTop = overlap//2
            ymin = 0
            ymax = ps
            xmin = xmin-overlap+ps
            xmax = xmin+ps
            ovLeft = overlap//2

        im = util.denormalize(im, net.mean, net.std)
        vmi = np.percentile(l, 0.05)
        vma = np.percentile(l, 99.5)
        #print(vmi, vma)

        psnrPrior = util.PSNR(l, means, 255)
        results.append(psnrPrior)

        print("PSNR raw", util.PSNR(l, im, 255))
        print("PSNR prior", psnrPrior)  # Without info from masked pixel
        print("index", index)

        """
        print(np.min(means), np.max(means))

        plt.imshow(im[200:328, 200:328], cmap='gray', vmin=0, vmax=255)  # GT
        plt.show()

        # MSE estimate using the masked pixel
        plt.imshow(means[200:328, 200:328], cmap='gray')
        plt.show()
        """

    print("Avg Prior:", np.mean(np.array(results)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--network_config", "-nc", help="Path to the network config.")
    parser.add_argument("--data_config", "-dc", help="Path to the data config.")
    args = parser.parse_args()
    #TODO load config
    main(args.network_config, args.data_config)