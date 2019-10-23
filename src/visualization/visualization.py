import os
import numpy as np
import json
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import FuncFormatter

cd_colors = ['#00305d', '#93107d', '#009de0', '#54368a', '#006ab2', '#69af22']

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

def single_dataset_scatterplot_to_csv(json_path,
                                      scatter_plot_name,
                                      img_factor=1.0):
    import csv
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
        with open(scatter_plot_name, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['x', 'y', 'label'])
            for i in range(weights_list.shape[0]):
                writer.writerow([weights_list[i][0], weights_list[i][1], i // img_factor])

def setup_matplotlib_style():
    matplotlib.rcParams.update({'figure.autolayout': True})
    matplotlib.rcParams.update({'font.size': 18})
    matplotlib.rcParams.update({'font.family': 'Open Sans'})
    matplotlib.rcParams.update({'font.weight': 'light'})
    matplotlib.rcParams.update({'axes.labelweight': 'light'})
    plt.box(True)

def print_weight_histogram_and_scatterplot(json_path,
                                           output_dir,
                                           hist_name,
                                           scatter_plot_name,
                                           img_factor=1,
                                           bins=20,
                                           range=None):
    """This functions produces a scatter plot and histogram of the weights
    contained in a single json results file.
    
    Arguments:
        json_path {str}         -- the path to the json file
        output_dir {str}        -- the directory where to store the results
        hist_name {str}         -- the filename of the histogram, the extension 
                                   determines the format
        scatter_plot_name {str} -- the filename of the scatter plot, the 
                                   extension determines the format
    
    Keyword Arguments:
        img_factor {int}        -- factor of raw to groun-truth images to determine
                                   number of distinct scenes
        bins {int}              -- the number of bins to use in the histogram
                                   (default: {20})
        range {list}            -- the range of the histogram (default: {None})
    """
    with open(json_path, 'r') as json_file:
        results_data = json.load(json_file)
        weights_list = []
        for key in results_data:
            data = results_data[key]
            if type(data) is dict:
                if 'weights' in data:
                    # Average image, Q-learning, REINFORCE
                    weights = data['weights']
                elif 'probabilities' in data:
                    # Probabilistic
                    weights = data['probabilities']
                elif 'mean_weights' in data:
                    # Average pixel
                    weights = data['mean_weights']
                else:
                    raise ValueError('Unkown weights key.')
                weights_list.append(weights)
        weights_list = np.array(weights_list)
        colors = np.arange(weights_list.shape[0] // img_factor)
        colors = np.repeat(colors, img_factor)

        setup_matplotlib_style()

        # scatter plot
        fig, ax = plt.subplots()
        ax.grid(zorder=0)
        scatter = ax.scatter(weights_list[:, 0], weights_list[:, 1], alpha=0.5, c=colors, zorder=3)
        # produce a legend with the unique colors from the scatter
        legend1 = ax.legend(*scatter.legend_elements(),
                            loc="upper right", title="Scene",
                            title_fontsize='12',
                            prop={'size': 14})
        ax.add_artist(legend1)
        ax.set_xlabel('Subnetwork 1')
        ax.set_ylabel('Subnetwork 2')
        plt.savefig(os.path.join(output_dir, scatter_plot_name))
        plt.clf()
        # histogram
        fig, ax = plt.subplots()
        normalized_weights = weights_list / np.expand_dims(np.sum(weights_list, axis=1), axis=-1)
        percentage_weights = normalized_weights
        ax.hist(percentage_weights[:, 0], range=range, bins=bins)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        plt.savefig(os.path.join(output_dir, hist_name))
        plt.clf()

def print_weight_histogram_and_scatterplot_multi(base_path,
                                                 json_paths,
                                                 output_dir,
                                                 scatter_plot_name):
    """This functions merges the weights stored in the specified results files
    into a single scatter plot and histogram.
    
    Arguments:
        json_paths {dict}       -- {key: json_path} - key is used in the legend 
                                   of the scatter plot and histogram
        output_dir {str}        -- the directory where to store the results
        scatter_plot_name {str} -- the filename of the scatter plot, the 
                                   extension determines the format
    
    Keyword Arguments:
        bins {int}              -- the number of bins to use for the histogram 
                                   (default: {20})
        range {list}            -- the range of the histogram (default: {None})
    """     

    setup_matplotlib_style()

    scatter_fig = plt.figure(0)
    scatter_axs = scatter_fig.add_subplot(111)
    scatter_axs.grid(zorder=0)
    scatter_axs.set_xlabel('Subnetwork 1')
    scatter_axs.set_ylabel('Subnetwork 2')
    xmin = 0
    xmax = 0
    ymin = 0
    ymax = 0
    for i, key in enumerate(json_paths):
        json_path = os.path.join(base_path, json_paths[key])
        with open(json_path, 'r') as json_file:
            results_data = json.load(json_file)
            weights_list = []
            sizes_list = []
            for image_key in results_data:
                data = results_data[image_key]
                if type(data) is dict:
                    if 'weights' in data:
                        # Average image, Q-learning, REINFORCE
                        weights = data['weights']
                        s = np.ones_like(data['weights'])
                    elif 'probabilities' in data:
                        # Probabilistic
                        weights = data['probabilities']
                        s = np.ones_like(data['probabilities'])
                    elif 'mean_weights' in data:
                        # Average pixel
                        weights = data['mean_weights']
                        s = data['std_mean_weights']
                    else:
                        raise ValueError('Unkown weights key.')
                    weights_list.append(weights)
                    sizes_list.append(s)
            sizes_list = np.array(sizes_list)
            weights_list = np.array(weights_list)

            #NOTE don't size the markers anymore because the variance is too high
            xmin = min(np.min(weights_list[:, 0] - 0.5 * sizes_list[:, 0]), xmin)
            xmax = max(np.max(weights_list[:, 0] + 0.5 * sizes_list[:, 0]), xmax)
            ymin = min(np.min(weights_list[:, 1] - 0.5 * sizes_list[:, 1]), ymin)
            ymax = max(np.max(weights_list[:, 1] + 0.5 * sizes_list[:, 1]), ymax)
            # scatter plot
            scatter_axs.scatter(weights_list[:, 0], weights_list[:, 1], alpha=0.5, label=key, c=cd_colors[i])

    scatter_fig.legend(loc='upper right')
    scatter_fig.savefig(os.path.join(output_dir, scatter_plot_name))
    scatter_fig.clf()

def print_weight_scatterplot_multi_3d(base_path,
                                      json_paths,
                                      output_dir,
                                      scatter_plot_name):
    """This functions merges the weights stored in the specified results files
    into a single scatter plot and histogram.
    
    Arguments:
        json_paths {dict}       -- {key: json_path} - key is used in the legend 
                                   of the scatter plot and histogram
        output_dir {str}        -- the directory where to store the results
        scatter_plot_name {str} -- the filename of the scatter plot, the 
                                   extension determines the format
    
    Keyword Arguments:
        bins {int}              -- the number of bins to use for the histogram 
                                   (default: {20})
        range {list}            -- the range of the histogram (default: {None})
    """     

    setup_matplotlib_style()

    scatter_fig = plt.figure(0)
    scatter_axs = scatter_fig.add_subplot(111, projection='3d')
    scatter_axs.grid(zorder=0)
    scatter_axs.set_xlabel('Subnetwork 1')
    scatter_axs.set_ylabel('Subnetwork 2')
    scatter_axs.set_zlabel('Subnetwork 3')

    scatter_axs.xaxis.set_tick_params(labelsize=10)
    scatter_axs.yaxis.set_tick_params(labelsize=10)
    scatter_axs.zaxis.set_tick_params(labelsize=10)

    for i, key in enumerate(json_paths):
        json_path = os.path.join(base_path, json_paths[key])
        with open(json_path, 'r') as json_file:
            results_data = json.load(json_file)
            weights_list = []
            sizes_list = []
            for image_key in results_data:
                data = results_data[image_key]
                if type(data) is dict:
                    if 'weights' in data:
                        # Average image, Q-learning, REINFORCE
                        weights = data['weights']
                    elif 'probabilities' in data:
                        # Probabilistic
                        weights = data['probabilities']
                    elif 'mean_weights' in data:
                        # Average pixel
                        weights = data['mean_weights']
                    else:
                        raise ValueError('Unkown weights key.')
                    weights_list.append(weights)
            weights_list = np.array(weights_list)
            # scatter plot
            scatter_axs.scatter(weights_list[:, 0], weights_list[:, 1], weights_list[:, 2], label=key, c=cd_colors[i])

    scatter_fig.legend(loc='upper left')
    scatter_fig.savefig(os.path.join(output_dir, scatter_plot_name))
    scatter_fig.clf()