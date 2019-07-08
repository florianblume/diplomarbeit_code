import os
import numpy as np
import sys
import shutil
import tifffile as tif
from skimage import io

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

"""
Script to create the datasets as they were used in the experiments.
"""

### Remove old data

print('Removing old generated data.')
shutil.rmtree(os.path.join(main_path, 'data/processed'))

print('Processing cells dataset...')

### Create fish dataset

print('Creating fish dataset...')
fish_dataset_src_path = os.path.join(main_path, 'data/raw/fish')
fish_dataset_dest_path = os.path.join(main_path, 'data/processed/fish')

print('Copying raw data to \"processed\" folder.')
shutil.copytree(fish_dataset_src_path, fish_dataset_dest_path)

# Add custom Gauss noise images

print('Creating Gauss-noised versions.')
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

print('Creating mouse dataset...')
fish_dataset_src_path = os.path.join(main_path, 'data/raw/mouse')
fish_dataset_dest_path = os.path.join(main_path, 'data/processed/mouse')

print('Copying raw data to \"processed\" folder.')
shutil.copytree(fish_dataset_src_path, fish_dataset_dest_path)

# Add custom Gauss noise images

print('Creating Gauss-noised versions.')
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
print('Creating fish-only joined datasets...')
# Here we take two fish datasets and merge them

# We need gt images twice
util.merge_two_npy_datasets('data/processed/fish/gt',
                            'data/processed/fish/gt',
                            'data/processed/joined/fish_only/gt')
# avg16 merged with raw without Gauss noise
print('Joining avg16 and raw.')
util.merge_two_npy_datasets('data/processed/fish/avg16',
                            'data/processed/fish/raw',
                            'data/processed/joined/fish_only/avg16_gauss0/noisy')
# avg16 merged with Gauss noise with std 30
print('Joining avg16 and Gauss 30.')
util.merge_two_npy_datasets('data/processed/fish/avg16',
                            'data/processed/fish/gauss30',
                            'data/processed/joined/fish_only/avg16_gauss30/noisy')

## Mouse-only
# We do not need any mouse-only datasets yet
#os.mkdir(os.path.join(main_path, 'data/processed/joined/mouse'))

## Fish-Mouse joined
print('Creating fish-mouse joined datasets...')

util.merge_two_npy_datasets('data/processed/fish/gt',
                            'data/processed/mouse/gt',
                            'data/processed/joined/fish_mouse/gt')

print('Joining raw images.')
util.merge_two_npy_datasets('data/processed/fish/raw',
                            'data/processed/mouse/raw',
                            'data/processed/joined/fish_mouse/raw/')

print('Joining fish raw and mouse Gauss 15.')
util.merge_two_npy_datasets('data/processed/fish/raw',
                            'data/processed/mouse/gauss15',
                            'data/processed/joined/fish_mouse/raw_gauss15/')

print('Joining Gauss 15 images.')
util.merge_two_npy_datasets('data/processed/fish/gauss15',
                            'data/processed/mouse/gauss15',
                            'data/processed/joined/fish_mouse/gauss15_gauss15/')

print('Joining fish raw and mouse Gauss 30.')
util.merge_two_npy_datasets('data/processed/fish/raw',
                            'data/processed/mouse/gauss30',
                            'data/processed/joined/fish_mouse/raw_gauss30/')
print('Joining fish avg8 and mouse Gauss 30.')
util.merge_two_npy_datasets('data/processed/fish/avg8',
                            'data/processed/mouse/gauss30',
                            'data/processed/joined/fish_mouse/avg8_gauss30/')

print('Joining fish avg16 and mouse Gauss 30.')
util.merge_two_npy_datasets('data/processed/fish/avg16',
                            'data/processed/mouse/gauss30',
                            'data/processed/joined/fish_mouse/avg16_gauss30/')

print('Joining fish Gauss 15 and mouse Gauss 30.')
util.merge_two_npy_datasets('data/processed/fish/gauss15',
                            'data/processed/mouse/gauss30',
                            'data/processed/joined/fish_mouse/gauss15_gauss30/')

print('Joining Gauss 30 images.')
util.merge_two_npy_datasets('data/processed/fish/gauss30',
                            'data/processed/mouse/gauss30',
                            'data/processed/joined/fish_mouse/gauss30_gauss30/')



print('Processing SimSim dataset...')

train_raw = tif.imread('data/raw/simsim/camsim_ccd_phot300_rn8_bgrd0.tif')
train_gt = tif.imread('data/raw/simsim/noise_free_32b.tif')
assert train_raw.shape[0] == train_raw.shape[0]

factor = int(train_raw.shape[0] / 3)

os.makedirs('data/processed/simsim/raw/')
os.makedirs('data/processed/simsim/gt/')
os.makedirs('data/processed/joined/simsim/raw')
os.makedirs('data/processed/joined/simsim/gt')

for i in range(3):
    np.save('data/processed/simsim/raw/train_part_{}.npy'.format(str(i)), train_raw[i * factor:(i + 1) * factor])
    np.save('data/processed/simsim/gt/train_part_{}.npy'.format(str(i)), train_gt[i * factor:(i + 1) * factor])

np.save('data/processed/joined/simsim/raw/train_raw.npy', train_raw)
np.save('data/processed/joined/simsim/gt/train_gt.npy', train_gt)

# Create the test set

test_raw = []
test_gt = []

for i in range(train_raw.shape[0]):
    # We rotate all images by 180 degrees to obtain our test set
    # The stripes of the light are in the same direction but the
    # image is different
    test_raw.append(np.rot90(train_raw[i], k=2))
    test_gt.append(np.rot90(train_gt[i], k=2))

for i in range(3):
    np.save('data/processed/simsim/raw/test_part_{}.npy'.format(str(i)), test_raw[i * factor:(i + 1) * factor])
    np.save('data/processed/simsim/gt/test_part_{}.npy'.format(str(i)), test_gt[i * factor:(i + 1) * factor])

np.save('data/processed/joined/simsim/raw/test_raw.npy', test_raw)
np.save('data/processed/joined/simsim/gt/test_gt.npy', test_gt)

print('Done.')