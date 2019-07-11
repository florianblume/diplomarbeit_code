import os
import numpy as np
import sys
import shutil
import tifffile as tif
from skimage import io

# Needs to be in this order otherwise util won't get found
main_path = os.getcwd()
sys.path.append(os.path.join(main_path, 'src'))
import util

def load_add_gauss_store(src_path, dest_path, mean, std):
    data = np.load(os.path.join(main_path, src_path))
    data = util.add_gauss_noise_to_images(data, mean, std)
    path = os.path.dirname(dest_path)
    if not os.path.exists(path):
        os.makedirs(path)
    np.save(os.path.join(main_path, dest_path), data)


def zero_mean(data):
    return data - np.mean(data, axis=0)


def adjust_raw_and_scaled_shifted_gt(gts, raws):
    range_ = 255.0 / np.std(gts)
    zero_gts = zero_mean(gts) / np.std(gts)
    zero_gts = zero_mean(zero_gts)
    zero_raws = zero_mean(raws)
    std = np.sum(zero_gts * zero_raws) / (np.sum(zero_raws * zero_raws))
    zero_raws *= std
    return zero_gts, zero_raws, range_


"""
Script to create the datasets as they were used in the experiments.
"""
"""
### Remove old data

print('Removing old generated data.')
if os.path.exists('data/processed'):
    shutil.rmtree(os.path.join(main_path, 'data/processed'))

print('Processing cells dataset.')

### Create fish dataset

print('Creating fish dataset.')
fish_dataset_src_path = os.path.join(main_path, 'data/raw/fish')
fish_dataset_dest_path = os.path.join(main_path, 'data/processed/fish')

print('.....Copying raw data to \"processed\" folder.')
shutil.copytree(fish_dataset_src_path, fish_dataset_dest_path)

# Add custom Gauss noise images

print('.....Creating Gauss-noised versions.')
load_add_gauss_store('data/raw/fish/raw/test_noisy.npy',
                     'data/processed/fish/gauss15/test_noisy.npy',
                     0,
                     15)
load_add_gauss_store('data/raw/fish/raw/test_noisy.npy',
                     'data/processed/fish/gauss30/test_noisy.npy',
                     0,
                     30)
load_add_gauss_store('data/raw/fish/raw/training_big_raw.npy',
                     'data/processed/fish/gauss15/training_big_raw.npy',
                     0,
                     15)
load_add_gauss_store('data/raw/fish/raw/training_big_raw.npy',
                     'data/processed/fish/gauss30/training_big_raw.npy',
                     0,
                     30)
                                    
### Create mouse dataset

print('Creating mouse dataset.')
fish_dataset_src_path = os.path.join(main_path, 'data/raw/mouse')
fish_dataset_dest_path = os.path.join(main_path, 'data/processed/mouse')

print('.....Copying raw data to \"processed\" folder.')
shutil.copytree(fish_dataset_src_path, fish_dataset_dest_path)

# Add custom Gauss noise images

print('.....Creating Gauss-noised versions.')
load_add_gauss_store('data/raw/mouse/raw/test_noisy.npy',
                     'data/processed/mouse/gauss15/test_noisy.npy',
                     0,
                     15)
load_add_gauss_store('data/raw/mouse/raw/test_noisy.npy',
                     'data/processed/mouse/gauss30/test_noisy.npy',
                     0,
                     30)
load_add_gauss_store('data/raw/mouse/raw/training_big_raw.npy',
                     'data/processed/mouse/gauss15/training_big_raw.npy',
                     0,
                     15)
load_add_gauss_store('data/raw/mouse/raw/training_big_raw.npy',
                     'data/processed/mouse/gauss30/training_big_raw.npy',
                     0,
                     30)

### Create joined dataset

print('Creating joined datasets.')

## Fish-only
print('.....Creating fish-only joined datasets.')
# Here we take two fish datasets and merge them

# We need gt images twice
util.merge_two_npy_datasets('data/processed/fish/gt',
                            'data/processed/fish/gt',
                            'data/processed/joined/fish_only/gt')
# avg16 merged with raw without Gauss noise
print('..........Joining avg16 and raw.')
util.merge_two_npy_datasets('data/processed/fish/avg16',
                            'data/processed/fish/raw',
                            'data/processed/joined/fish_only/avg16_gauss0/noisy')
# avg16 merged with Gauss noise with std 30
print('..........Joining avg16 and Gauss 30.')
util.merge_two_npy_datasets('data/processed/fish/avg16',
                            'data/processed/fish/gauss30',
                            'data/processed/joined/fish_only/avg16_gauss30/noisy')

## Mouse-only
# We do not need any mouse-only datasets yet
#os.mkdir(os.path.join(main_path, 'data/processed/joined/mouse'))

## Fish-Mouse joined
print('.....Creating fish-mouse joined datasets.')

util.merge_two_npy_datasets('data/processed/fish/gt',
                            'data/processed/mouse/gt',
                            'data/processed/joined/fish_mouse/gt')

print('..........Joining raw images.')
util.merge_two_npy_datasets('data/processed/fish/raw',
                            'data/processed/mouse/raw',
                            'data/processed/joined/fish_mouse/raw/')

print('..........Joining fish raw and mouse Gauss 15.')
util.merge_two_npy_datasets('data/processed/fish/raw',
                            'data/processed/mouse/gauss15',
                            'data/processed/joined/fish_mouse/raw_gauss15/')

print('..........Joining Gauss 15 images.')
util.merge_two_npy_datasets('data/processed/fish/gauss15',
                            'data/processed/mouse/gauss15',
                            'data/processed/joined/fish_mouse/gauss15_gauss15/')

print('..........Joining fish raw and mouse Gauss 30.')
util.merge_two_npy_datasets('data/processed/fish/raw',
                            'data/processed/mouse/gauss30',
                            'data/processed/joined/fish_mouse/raw_gauss30/')
print('..........Joining fish avg8 and mouse Gauss 30.')
util.merge_two_npy_datasets('data/processed/fish/avg8',
                            'data/processed/mouse/gauss30',
                            'data/processed/joined/fish_mouse/avg8_gauss30/')

print('..........Joining fish avg16 and mouse Gauss 30.')
util.merge_two_npy_datasets('data/processed/fish/avg16',
                            'data/processed/mouse/gauss30',
                            'data/processed/joined/fish_mouse/avg16_gauss30/')

print('..........Joining fish Gauss 15 and mouse Gauss 30.')
util.merge_two_npy_datasets('data/processed/fish/gauss15',
                            'data/processed/mouse/gauss30',
                            'data/processed/joined/fish_mouse/gauss15_gauss30/')

print('..........Joining Gauss 30 images.')
util.merge_two_npy_datasets('data/processed/fish/gauss30',
                            'data/processed/mouse/gauss30',
                            'data/processed/joined/fish_mouse/gauss30_gauss30/')
"""
print('Processing SimSim dataset.')

print('.....Creating SimSim-only datasets.')

train_raw = tif.imread('data/raw/simsim/camsim_ccd_phot300_rn8_bgrd0.tif')
train_gt = tif.imread('data/raw/simsim/noise_free_32b.tif')
assert train_raw.shape[0] == train_raw.shape[0]

# SimSim ground-truth data is scaled and shifted in color space - we need to
# correct that to be able to mix it with datasets where this is not the case
train_gt, train_raw, range_ = adjust_raw_and_scaled_shifted_gt(
                                            train_gt, train_raw)

factor = int(train_raw.shape[0] / 3)
"""
os.makedirs('data/processed/simsim/raw/')
os.makedirs('data/processed/simsim/gt/')
os.makedirs('data/processed/joined/simsim/all/raw')
os.makedirs('data/processed/joined/simsim/all/gt')
"""
print('..........Creating dataset containing all parts.')

np.save('data/processed/joined/simsim/all/raw/train.npy', train_raw)
np.save('data/processed/joined/simsim/all/gt/train.npy', train_gt)

train_raws = []
train_gts = []

sub_indices = [(0, 1), (1, 2), (0, 2)]

print('..........Creating dataset consisting of parts only.')

for i in range(3):
    train_raws.append(train_raw[i * factor:(i + 1) * factor])
    train_gts.append(train_gt[i * factor:(i + 1) * factor])
    np.save(
        'data/processed/simsim/raw/train_part_{}.npy'.format(str(i)), train_raws[i])
    np.save(
        'data/processed/simsim/gt/train_part_{}.npy'.format(str(i)), train_gts[i])

# Create combination of subsets
for sub_index in sub_indices:
    """
    os.makedirs('data/processed/joined/simsim/part_{}_{}/raw'.format(
                                                    sub_index[0], sub_index[1]))
    os.makedirs('data/processed/joined/simsim/part_{}_{}/gt'.format(
                                                    sub_index[0], sub_index[1]))
    """
    np.save(
        'data/processed/joined/simsim/part_{}_{}/raw/train.npy'.format(sub_index[0], sub_index[1]),
        np.concatenate([train_raws[sub_index[0]], train_raws[sub_index[1]]], axis=0))
    np.save(
        'data/processed/joined/simsim/part_{}_{}/gt/train.npy'.format(sub_index[0], sub_index[1]),
        np.concatenate([train_gts[sub_index[0]], train_gts[sub_index[1]]], axis=0))

# Create the test set

# Complete set
test_raw = []
test_gt = []

# Smaller sets
test_raws = []
test_gts = []

print('..........Creating artificial ground-truth by rotating images by 180 degrees.')

for i in range(train_raw.shape[0]):
    # We rotate all images by 180 degrees to obtain our test set
    # The stripes of the light are in the same direction but the
    # image is different
    test_raw.append(np.rot90(train_raw[i], k=2))
    test_gt.append(np.rot90(train_gt[i], k=2))

test_raw = np.array(test_raw)
test_gt = np.array(test_gt)

for i in range(3):
    test_raws.append(test_raw[i * factor:(i + 1) * factor])
    test_gts.append(test_gt[i * factor:(i + 1) * factor])
    np.save(
        'data/processed/simsim/raw/test_part_{}.npy'.format(str(i)), test_raws[i])
    np.save('data/processed/simsim/gt/test_part_{}.npy'.format(str(i)),
            test_gts[i])

np.save('data/processed/joined/simsim/all/raw/test.npy', test_raw)
np.save('data/processed/joined/simsim/all/gt/test.npy', test_gt)

# Create combination of subsets
for sub_index in sub_indices:
    np.save(
        'data/processed/joined/simsim/part_{}_{}/raw/test.npy'.format(sub_index[0], sub_index[1]),
        np.concatenate([test_raws[sub_index[0]], test_raws[sub_index[1]]], axis=0))
    np.save(
        'data/processed/joined/simsim/part_{}_{}/gt/test.npy'.format(sub_index[0], sub_index[1]),
        np.concatenate([test_gts[sub_index[0]], test_gts[sub_index[1]]], axis=0))

# Create mixed datasets

print('Creating fused datasets.')

print('.....Creating fused dataset of fish and SimSim.')

paths = ['raw_all/raw', 'raw_all/gt', 'raw_part0/raw', 'raw_part0/gt',
         'raw_part1/raw', 'raw_part1/gt', 'raw_part2/raw', 'raw_part2/gt']

#for path in paths:
    #os.makedirs(os.path.join('data/processed/joined/fish_simsim', path))

def fuse_fish_and_simsim(data):
    # The network automatically repeats the ground-truth images so that they
    # fit the number of raw images. This can be down because a ground-truth
    # image is usually obtained by averaging multiple images. When we fuse
    # these two different datasets then this is not possible anymore, as the
    # factors do not match. That's why we repeat the images here before.
    repeat_fish = data[3]

    fish_path = data[0]
    fish_data = np.load(fish_path)
    sim_im_size = train_raw[0].shape
    fish_im_size = fish_data[0].shape
    rect_origin = (int((fish_im_size[0] - sim_im_size[0]) / 2), 
                int((fish_im_size[1] - sim_im_size[1]) / 2))

    # Image sizes of fish and SimSim do not match, this is why we cut out a
    # rectangle of the size of the SimSim images in the center of the fish images

    fish_data_cut = []
    for i in range(fish_data.shape[0]):
        fish_data_cut.append(
            fish_data[i][rect_origin[0]:rect_origin[0] + sim_im_size[0],
                        rect_origin[1]:rect_origin[1] + sim_im_size[1]])
    fish_data_cut = np.array(fish_data_cut)
    fish_data_cut = np.repeat(fish_data_cut, repeat_fish, axis=0)

    np.save(
        'data/processed/joined/fish_simsim/raw_all/' + data[1] + '/' + data[2] + '.npy',
        np.concatenate([fish_data_cut, train_raw], axis=0))
    np.save(
        'data/processed/joined/fish_simsim/raw_part0/' + data[1] + '/' + data[2] + '.npy',
        np.concatenate([fish_data_cut, train_raws[0]], axis=0))
    np.save(
        'data/processed/joined/fish_simsim/raw_part1/' + data[1] + '/' + data[2] + '.npy',
        np.concatenate([fish_data_cut, train_raws[1]], axis=0))
    np.save(
        'data/processed/joined/fish_simsim/raw_part2/' + data[1] + '/' + data[2] + '.npy',
        np.concatenate([fish_data_cut, train_raws[2]], axis=0))
    
# We know the factors of the gt to raw from experience
# We only need to repeat gt data as those are fewer images
data_to_fuse = [('data/raw/fish/raw/training_big_raw.npy', 'raw', 'train', 1),
                ('data/raw/fish/gt/training_big_GT.npy', 'gt', 'train', 50),
                ('data/raw/fish/raw/test_noisy.npy', 'raw', 'test', 1),
                ('data/raw/fish/gt/test_gt.npy', 'gt', 'test', 50)]

for data in data_to_fuse:
    fuse_fish_and_simsim(data)

print('Done.')
