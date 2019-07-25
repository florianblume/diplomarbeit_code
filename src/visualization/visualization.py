import os
import numpy as np
import json
import matplotlib.pyplot as plt

def numpy_array_to_images(array_path, output_path):
    """To be able to visualize the numpy arrays that the networks handle this
    function can convert a dedicated numpy array file to images.

    Arguments:
        array_path {[type]} -- [description]
        output_path {[type]} -- [description]
    """
    import matplotlib.pyplot as plt
    import os
    data = np.load(array_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for i, image in enumerate(data):
        plt.imsave(os.path.join(output_path, str(i).zfill(4) + '.png'), image, cmap='gray')

def print_weight_histogram(json_path, output_file, bins=20, range=None):
    with open(json_path, 'r') as json_file:
        results_data = json.load(json_file)
        weights_list = []
        for key in results_data:
            data = results_data[key]
            if type(data) is dict:
                weights = data['weights']
                weights_list.append(weights)
        weights_list = np.array(weights_list)
        fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
        axs[0].hist(weights_list[:, 0], range=range, bins=bins)
        axs[1].hist(weights_list[:, 1], range=range, bins=bins)
        dirname = os.path.dirname(json_path)
        plt.savefig(os.path.join(dirname, output_file))

#TODO Fourier transform for images
