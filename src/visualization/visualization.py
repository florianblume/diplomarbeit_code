import numpy as np


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


#TODO Fourier transform for images