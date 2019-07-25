import os
import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

def numpy_array_to_images(array_path, output_path):
    """To be able to visualize the numpy arrays that the networks handle this
    function can convert a dedicated numpy array file to images.

    Arguments:
        array_path {[type]} -- [description]
        output_path {[type]} -- [description]
    """
    data = np.load(array_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for i, image in enumerate(data):
        plt.imsave(os.path.join(output_path, str(i).zfill(4) + '.png'), image, cmap='gray')

def print_weight_histogram_and_scatterplot(json_path, bins=20, range=None):
    with open(json_path, 'r') as json_file:
        results_data = json.load(json_file)
        weights_list = []
        for key in results_data:
            data = results_data[key]
            if type(data) is dict:
                weights = data['weights']
                weights_list.append(weights)
        weights_list = np.array(weights_list)
        dirname = os.path.dirname(json_path)
        # scatter plot
        fig, ax = plt.subplots()
        ax.scatter(weights_list[:, 0], weights_list[:, 1], alpha=0.5)
        plt.savefig(os.path.join(dirname, "scatterplot.svg"))
        plt.clf()
        # histogram
        fig, ax = plt.subplots()
        normalized_weights = weights_list / np.expand_dims(np.sum(weights_list, axis=1), axis=-1)
        percentage_weights = normalized_weights
        ax.hist(percentage_weights[:, 0], range=range, bins=bins)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        dirname = os.path.dirname(json_path)
        plt.savefig(os.path.join(dirname, "histogram.svg"))

def print_weight_histogram_and_scatterplot_multi(json_paths,
                                                 output_path,
                                                 hist_name,
                                                 scatter_plot_name,
                                                 bins=20,
                                                 range=None):
    scatter_fig = plt.figure(0)
    scatter_axs = scatter_fig.add_subplot(111)
    hist_fig = plt.figure(1)
    hist_axs = hist_fig.add_subplot(111)
    for i, key in enumerate(json_paths):
        json_path = json_paths[key]
        with open(json_path, 'r') as json_file:
            results_data = json.load(json_file)
            weights_list = []
            for image_key in results_data:
                data = results_data[image_key]
                if type(data) is dict:
                    weights = data['weights']
                    weights_list.append(weights)
            weights_list = np.array(weights_list)
            # scatter plot
            scatter_axs.scatter(weights_list[:, 0], weights_list[:, 1], alpha=0.5, label=key)

            # histogram
            normalized_weights = weights_list / np.expand_dims(np.sum(weights_list, axis=1), axis=-1)
            percentage_weights = normalized_weights
            hist = np.histogram(percentage_weights[:, 0], bins=bins, range=range)
            hist_axs.plot(hist[1][:-1], hist[0], label=key)

    scatter_fig.legend(loc='upper right')
    scatter_fig.savefig(os.path.join(output_path, scatter_plot_name))
    scatter_fig.clf()
    hist_axs.xaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.2%}'.format(y)))
    hist_fig.legend(loc='upper right')
    hist_fig.savefig(os.path.join(output_path, hist_name))
    hist_fig.clf()

#TODO Fourier transform for images
