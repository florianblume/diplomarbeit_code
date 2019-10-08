import os
import glob
import numpy as np
import torch

def cycle(iterable):
    """Make a cycle of an iterable to avoid StopIteration exceptions.
    
    Arguments:
        iterable {iter} -- the iterable object
    """
    while True:
        for x in iterable:
            yield x

def normal_dense(x, m_=0.0, std_=None):
    tmp = -((x-m_)**2)
    tmp = tmp / (2.0*std_*std_)
    tmp = torch.exp(tmp)
    tmp = tmp / torch.sqrt((2.0*np.pi)*std_*std_)
    return tmp

def img_to_tensor(img):
    import torchvision
    if len(img.shape) == 2:
        img.shape = img.shape + (1,)
    else:
        assert len(img.shape) == 3
        # RGB or gray-scale image
        assert img.shape[2] == 3 or img.shape[2] == 1
    # torch expects channels as the first dimension - this function automatically
    # permuates the dimensions correctly
    imgOut = torchvision.transforms.functional.to_tensor(img)
    return imgOut

def tile_tensor(a, dim, n_tile, device='cuda:0'):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile)
                                                   + i for i in range(init_dim)]))
    order_index = order_index.to(device)
    return torch.index_select(a, dim, order_index)

def seed_numpy(seed):
    print('Seeding numpy with {}.'.format(seed))
    np.random.seed(seed)

def joint_shuffle(inA, inB, seed=None):
    """Shuffles both numpy arrays consistently.
    This is useful to shuffle raw and ground-truth
    data together. Both arrays need to have the same
    dimensions.

    Arguments:
        inA {np.array} -- first array to shuffle
        inB {np.array} -- second array to shuffle
        seed {int}     -- if set, will be used as seed to numpy

    Returns:
        np.array, np.array -- the shuffled arrays
    """
    assert inA.shape[0] == inB.shape[0]
    indices = np.arange(inA.shape[0])
    if seed is not None:
        seed_numpy(seed)
    np.random.shuffle(indices)
    return inA[indices], inB[indices]


def shuffle(inA, seed=None):
    """Shuffles the given array. This function is here to provide a single point
    of responsibility.
    
    Arguments:
        inA {np.array} -- the array to shuffle
        seed {int}     -- if set, will be used as seed to numpy
    
    Returns:
        np.array -- the shuffled array
    """
    if seed is not None:
        seed_numpy(seed)
    indices = np.arange(inA.shape[0])
    np.random.shuffle(indices)
    return inA[indices]

def PSNR(gt, pred, range_=255.0):
    mse = MSE(gt, pred)
    return 20 * np.log10((range_)/np.sqrt(mse))


def MSE(gt, pred):
    return np.mean((gt - pred)**2)


def average_PSNR_of_stored_numpy_array(path_to_array, path_to_gt_array):
    data = np.load(path_to_array)
    gt = np.load(path_to_gt_array)
    img_factor = int(data.shape[0]/gt.shape[0])
    gt = np.repeat(gt, img_factor, axis=0)
    psnr = 0
    for index in range(data.shape[0]):
        psnr += PSNR(gt[index], data[index])
    psnr /= data.shape[0]
    return psnr


def normalize(img, mean, std):
    zero_mean = img - mean
    return zero_mean/std


def denormalize(x, mean, std):
    return x*std + mean


def load_config(config_path):
    import yaml
    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)
        # Configs duplicate a lot of things, check if we are extending another
        # config
        extended_config_path = config.get('EXTENDS_CONFIG', None)
        if extended_config_path is not None:
            if not os.path.isabs(extended_config_path):
                # In case we don't have an absolute path prepend the path to
                # the folder the config is in
                basedir = os.path.dirname(config_path)
                extended_config_path = os.path.join(basedir, extended_config_path)
            extended_config = load_config(extended_config_path)
            extended_config.update(config)
            config = extended_config
        return config
    raise "Config could not be loaded."


def add_shot_noise_to_images(images, defect_ratio):
    result = []
    for image in images:
        out = np.copy(image)
        num_salt = np.ceil(image.size * defect_ratio)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in image.shape]
        out[coords] = np.random.randint(255, size=len(coords[0]))
        result.append(out)
    return np.array(result)


def add_gauss_noise_to_images(images, mean, std):
    result = []
    for image in images:
        noisy = image + np.random.normal(mean, std, image.shape)
        noisy = np.clip(noisy, 0, 255)
        result.append(noisy)
    return np.array(result)


def load_trainer_or_predictor(type_, config, config_path):
    """Loads the trainer or predictor class specified in the config.
    
    Arguments:
        type_ {str} -- the type to load, possible choices: 'Trainer' or 'Predictor'
        config {dict} -- the config that specifies what exactly to load
        config_path {str} -- the path to the config on the filesystem
    
    Returns:
        object -- the loaded trainer or predictor
    """
    import importlib
    model = config['MODEL']
    sub_model = config.get('SUB_MODEL', None)
    model_module_name = 'models.' + model
    type_class_name = (sub_model.capitalize()
                            if sub_model is not None else "") + type_
    type_module = importlib.import_module(model_module_name)
    type_class = getattr(type_module, type_class_name)
    result = type_class(config, os.path.dirname(config_path))
    return result

def fuse_trained_networks_as_subnetworks(trained_networks: list, output_path: str):
    """This method fuses the subnetworks specified in the list and stores the
    result at the output_path.
    
    Arguments:
        trained_networks {list} -- the list of paths to the trained networks
        output_path {str} -- the path where to store the result
    """
    model_state_dict = {}
    for i, trained_network in enumerate(trained_networks):
        assert os.path.exists(trained_network)
        existing_state_dict = torch.load(trained_network)['model_state_dict']
        existing_state_dict = {f'subnets.{i}.{k}': v for k, v in existing_state_dict.items()}
        model_state_dict.update(existing_state_dict)
    checkpoint = {'model_state_dict': model_state_dict}
    torch.save(checkpoint, output_path)

def fuse_all_trained_networks_at_path_as_subnetworks(base_paths: list,
                                                     network_identifier: str,
                                                     output_path: str):
    """This function fuses all networks found at the specified path. This is
    useful for example if multiple runs were conducted. The individual paths can
    be of the form 'experiments/network/fish_' and this function will find the
    runs 'experiments/network/fish_0', 'experiments/network/fish_1' and so on.
    The networks will be fused along axis 0, i.e. the first network of the first
    provided path will be fused with the first network of the second path and
    the first network of the third path, etc.
    
    Arguments:
        base_paths {list}        -- the list of paths to the stem of the folders
                                    containing the networks
        network_identifier {str} -- e.g. best.net
        output_path {str}        -- where to store the results, new folders will
                                    be created
    """
    import natsort

    found_subnetworks = {}
    count = -1
    for base_path in base_paths:
        paths = glob.glob(base_path + '**', recursive=True)
        paths = natsort.natsorted(paths)
        networks = [os.path.join(path, network_identifier) for path in paths]
        found_subnetworks[base_path] = networks
        if count == -1:
            count = len(networks)
        else:
            assert count == len(networks), 'The number of found subnetworks needs to match.'
    for i in range(count):
        paths_to_fuse = []
        for path in found_subnetworks:
            paths_to_fuse.append(found_subnetworks[path][i])
        temp_output_path = output_path + f'_{i}'
        if not os.path.exists(temp_output_path):
            os.makedirs(temp_output_path)
        fuse_trained_networks_as_subnetworks(paths_to_fuse, os.path.join(temp_output_path, 'fused.net'))

def pretty_string_with_percentages(weights, percentages):
    # Print it in a nice format like 15.6754 (3.1203%)
    string = ', '.join('{:.4f} ({:.4f}%)'.format(*t) for
                        t in zip(weights, percentages))
    return string

def pretty_string(weights):
    # Print it in a nice format like 15.6754
    string = ', '.join('{:.4f}'.format(w) for w in weights)
    return string

def pretty_string_percentage(weights):
    # Print it in a nice format like 15.6754%
    string = ', '.join('{:.4f}%'.format(w) for w in weights)
    return string

def pretty_string_percentage_with_std(weights, stds):
    # Print it in a nice format like 15.6754% (0.0134)
    string = ', '.join('{:.4f}% ({:.4f})'.format(*t) for
                        t in zip(weights, stds))
    return string

def _psnrs_of_multiple_runs(path):
    import json

    psnrs = {}

    i = 0
    path_ = path + '_' + str(i)
    while os.path.exists(path_):
        sub_paths = glob.glob(path_ + '/prediction*')
        for sub_path in sub_paths:
            with open(os.path.join(sub_path, 'results.json'), 'r') as results_file:
                results = json.load(results_file)
                basename = os.path.basename(sub_path)
                if basename not in psnrs:
                    psnrs[basename] = []
                psnrs[basename].append(results['psnr_average'])
        i += 1
        path_ = path + '_' + str(i)
    return psnrs

def compute_mean_std_multiple_runs(path):
    psnrs = _psnrs_of_multiple_runs(path)
    for key in psnrs:
        psnr_values = psnrs[key]
        print('{}: {} (mean) - {} (std) - {} (std err)'
                    .format(key, np.mean(psnr_values), np.std(psnr_values), np.std(psnr_values) / np.sqrt(len(psnr_values))))

def compute_mean_std_multiple_runs_fuse_external(path1, path2):
    psnrs1 = _psnrs_of_multiple_runs(path1)
    psnrs2 = _psnrs_of_multiple_runs(path2)

    for key in psnrs1:
        assert key in psnrs2
        psnrs = np.array(psnrs1[key])
        psnrs = (psnrs + np.array(psnrs2[key])) / 2.0
        print('{}: {} (mean) - {} (std) - {} (std err)'.format(key, np.mean(psnrs), np.std(psnrs), np.std(psnrs)/np.sqrt(psnrs.shape[0])))

def _psnrs_for_fuse(path, fuse):
    import json
    psnrs = []

    i = 0
    path_ = path + '_' + str(i)
    while os.path.exists(path_):
        with open(os.path.join(path_, fuse, 'results.json'), 'r') as results_file:
            results = json.load(results_file)
            psnrs.append(results['psnr_average'])
        i += 1
        path_ = path + '_' + str(i)
    return psnrs

def compute_mean_std_multiple_runs_fuse_internal(path, fuse1, fuse2):
    psnrs1 = _psnrs_for_fuse(path, fuse1)
    psnrs2 = _psnrs_for_fuse(path, fuse2)
    fused_psnrs = (np.array(psnrs1) + np.array(psnrs2)) / 2
    print('{} (mean) - {} (std) - {} (std err)'.format(np.mean(fused_psnrs), np.std(fused_psnrs), np.std(fused_psnrs)/np.sqrt(fused_psnrs.shape[0])))

def psnr_of_dataset(raw_path, gt_path, range_=None):
    import tifffile as tif
    raw_image_files = glob.glob(raw_path + '/*.tif')
    gt_image_files = glob.glob(gt_path + '/*.tif')
    factor = len(raw_image_files) / len(gt_image_files)

    raw_images = []
    for raw_image_file in raw_image_files:
        raw_images.append(tif.imread(raw_image_file))
    raw_images = np.array(raw_images)

    gt_images = []
    for gt_image_file in gt_image_files:
        gt_images.append(tif.imread(gt_image_file))
    gt_images = np.array(gt_images)

    if range_ is None:
        range_max = np.max(gt_images)
        range_min = np.min(gt_images)
        range_ = range_max - range_min

    gt_images = np.repeat(gt_images, factor, axis=0)
    print('Avg. PSNR', PSNR(gt_images, raw_images, range_))

def compute_RF_numerical(net, img_shape):
    '''
    @param net: Pytorch network
    @param img_np: numpy array to use as input to the networks, it must be full of ones and with the correct
    shape.
    '''
    from torch.autograd import Variable
    import torch.functional as nn
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if hasattr(m, 'weight'):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.fill_(0)
    net.apply(weights_init)
    img_np = np.ones(img_shape)
    img_ = Variable(torch.from_numpy(img_np).float(), requires_grad=True)
    out_cnn = net(img_)
    out_shape = out_cnn.size()
    ndims = len(out_cnn.size())
    grad = torch.zeros(out_cnn.size())
    l_tmp = []
    for i in range(ndims):
        if i==0 or i ==1:#batch or channel
            l_tmp.append(0)
        else:
            l_tmp.append(out_shape[i]/2)
    l_tmp = (int(l) for l in l_tmp)
    grad[tuple(l_tmp)]=1
    out_cnn.backward(gradient=grad)
    grad_np = img_.grad[0,0].data.numpy()
    idx_nonzeros = np.where(grad_np!=0)
    RF = [np.max(idx)-np.min(idx)+1 for idx in idx_nonzeros]

    return RF

def min_image_shape_of_datasets(datasets):
    import tifffile as tif

    loaded_images = []

    for dataset in datasets:
        images = glob.glob(os.path.join(dataset, "*.tif"))
        if images:
            loaded_images.append(tif.imread(images[0]))

    shape = None
    # Each dataset has one training sample so we can use those to infer
    # the minimum image size
    for image in loaded_images:
        if shape == None:
            shape = list(image.shape[:2])
        shape[0] = min(shape[0], image.shape[0])
        shape[1] = min(shape[1], image.shape[1])
    return shape

def compute_subnetwork_utilization(path, identifier, key, with_std=False, std_key=None):
    """This function searches for all 'identifier' folders within the given path
    and loads the results files contained in folders which it finds by looking
    for configs conforming to the name scheme 'config_' as those indicate
    inference configs for individual datasets.
    
    Arguments:
        path {str}       -- the path to look for 'identifier' folders within
        identifier {str} -- the identifier, e.g. q_learning or reinforce
        key {str}        -- the key used in the results file to store average weights
    """
    import json
    import natsort

    assert with_std == (std_key != None), '\"std_key\" must not be none if \"with_std\" is set to True.'

    assert os.path.exists(path)
    sub_folders = glob.glob(os.path.join(path, '**', identifier), recursive=True)
    sub_folders = natsort.natsorted(sub_folders)

    print('')
    print('Found {} folder(s) by identifier \"{}\" in folder {}'
                    .format(len(sub_folders), identifier, path))

    configs = {}
    count = 0
    for sub_folder in sub_folders:
        _configs = glob.glob(os.path.join(sub_folder, '**', 'config_*.yml'), recursive=True)
        _configs = natsort.natsorted(_configs)
        for config in _configs:
            dirname = os.path.dirname(config)
            if dirname not in configs:
                configs[dirname] = []
            configs[dirname].append(config)
        count += len(_configs)
    
    print('Found {} configs for individual inference.'.format(count))
    print('')

    for folder in configs:
        _configs = configs[folder]
        output_paths = []
        print(folder + ':')
        for config in _configs:
            loaded_config = load_config(config)
            output_path = os.path.join(os.path.dirname(config), loaded_config['PRED_OUTPUT_PATH'])
            output_paths.append(output_path)
        for output_path in output_paths:
            results_path = os.path.join(output_path, 'results.json')
            if not os.path.exists(results_path):
                print('No \"results.json\" found at path {}.'.format(output_path))
                continue
            with open(results_path, 'r') as result_file:
                results = json.load(result_file)
            assert key in results, 'The specified key {} is not contained in the results file.'.format(key)
            average_weights = results[key]
            average_weights = np.array(average_weights)
            average_weights /= np.sum(average_weights)
            if with_std:
                assert std_key in results, 'The specified std key {} is not contained in the results file.'.format(std_key)
                stds = results[std_key]
                print('\t' + os.path.basename(output_path) + ':\t' + pretty_string_percentage_with_std(average_weights, stds))
            else:
                print('\t' + os.path.basename(output_path) + ':\t' + pretty_string_percentage(average_weights))
        print('')

def find_psnr_extremes_in_comparison(joint, individual, output_path=None):
    import json
    import csv
    with open(joint, 'r') as joint_file:
        joint_data = json.load(joint_file)
    with open(individual, 'r') as individual_file:
        individual_data = json.load(individual_file)
    psnrs = {}
    for key in joint_data.keys():
        if type(joint_data[key]) == dict:
            if key in individual_data.keys():
                psnrs[key] = individual_data[key]['psnr'] - joint_data[key]['psnr']
    minimum = min(psnrs, key=psnrs.get)
    maximum = max(psnrs, key=psnrs.get)
    print('Minimum:', minimum, psnrs[minimum])
    print('Maximum', maximum, psnrs[maximum])
    if output_path is not None:
        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for k,v in sorted(psnrs.items(), key=lambda p:p[1]):
                print('\t', v,k)
                writer.writerow([0, v])
    else:
        for k,v in sorted(psnrs.items(), key=lambda p:p[1]):
                print('\t', v,k)
    

def find_psnr_extremes_in_one_training(path):
    import json
    psnrs = {}
    with open(path, 'r') as results_file:
        results_data = json.load(results_file)
        average_psnr = results_data['psnr_average']
        for key in results_data.keys():
            if type(results_data[key]) == dict:
                image = results_data[key]
                psnrs[key] = average_psnr - results_data[key]['psnr']
    minimum = min(psnrs, key=psnrs.get)
    maximum = min(psnrs, key=psnrs.get)
    print(minimum, psnrs[minimum])
    print(maximum, psnrs[maximum])