import os
import tempfile
import pytest
import numpy as np
import tifffile as tif

# Dataset with factor 1 - 2 gt to raw images
dataset_1_raw_dir = tempfile.TemporaryDirectory()
dataset_1_raw_same_1_dir = tempfile.TemporaryDirectory()
dataset_1_raw_same_2_dir = tempfile.TemporaryDirectory()
dataset_1_gt_dir = tempfile.TemporaryDirectory()

# Dataset with factor 1 - 1 gt to raw images
dataset_2_raw_dir = tempfile.TemporaryDirectory()
dataset_2_gt_dir = tempfile.TemporaryDirectory()

# Dataset with factor 1 - 3 gt to raw images
dataset_3_raw_dir = tempfile.TemporaryDirectory()
dataset_3_gt_dir = tempfile.TemporaryDirectory()

def dataset_1_raw_images():
    raw_image_1 = np.arange(0, 90, 0.1).reshape(30, 30)
    raw_image_2 = np.arange(90, 2 * 90, 0.1).reshape(30, 30)
    raw_image_3 = np.arange(2 * 90, 3 * 90, 0.1).reshape(30, 30)
    raw_image_4 = np.arange(3 * 90, 4 * 90, 0.1).reshape(30, 30)
    return [raw_image_1, raw_image_2, raw_image_3, raw_image_4]

# To test if even probability works
def dataset_1_raw_images_same_1():
    raw_image_1 = np.arange(0, 90, 0.1).reshape(30, 30)
    raw_image_2 = np.arange(90, 2 * 90, 0.1).reshape(30, 30)
    raw_image_3 = np.arange(2 * 90, 3 * 90, 0.1).reshape(30, 30)
    raw_image_4 = np.arange(3 * 90, 4 * 90, 0.1).reshape(30, 30)
    return [raw_image_1, raw_image_2, raw_image_3, raw_image_4]

# To test if even probability works
def dataset_1_raw_images_same_2():
    raw_image_1 = np.arange(0, 90, 0.1).reshape(30, 30)
    raw_image_2 = np.arange(90, 2 * 90, 0.1).reshape(30, 30)
    raw_image_3 = np.arange(2 * 90, 3 * 90, 0.1).reshape(30, 30)
    raw_image_4 = np.arange(3 * 90, 4 * 90, 0.1).reshape(30, 30)
    return [raw_image_1, raw_image_2, raw_image_3, raw_image_4]

def dataset_1_gt_images():
    gt_image_1 = np.arange(4 * 90, 5 * 90, 0.1).reshape(30, 30)
    gt_image_2 = np.arange(5 * 90, 6 * 90, 0.1).reshape(30, 30)
    return [gt_image_1, gt_image_2]

def dataset_2_raw_images():
    raw_image_1 = np.arange(0, 9, 0.01).reshape(30, 30)
    raw_image_2 = np.arange(9, 18, 0.01).reshape(30, 30)
    return [raw_image_1, raw_image_2]

def dataset_2_gt_images():
    gt_image_1 = np.arange(18, 27, 0.01).reshape(30, 30)
    gt_image_2 = np.arange(27, 36, 0.01).reshape(30, 30)
    return [gt_image_1, gt_image_2]

def dataset_3_raw_images():
    raw_image_1 = np.arange(0, 900).reshape(30, 30)
    raw_image_2 = np.arange(900, 2 * 900).reshape(30, 30)
    raw_image_3 = np.arange(2 * 900, 3 * 900).reshape(30, 30)
    return [raw_image_1, raw_image_2, raw_image_3]

def dataset_3_gt_images():
    gt_image_1 = np.arange(3 * 900, 4 * 900).reshape(30, 30)
    return [gt_image_1]

@pytest.fixture(scope='session', autouse=True)
def setup_session():
    dataset_dirs = [(dataset_1_raw_dir, dataset_1_gt_dir),
                    (dataset_2_raw_dir, dataset_2_gt_dir),
                    (dataset_3_raw_dir, dataset_3_gt_dir)]
    dataset_images = [(dataset_1_raw_images(), dataset_1_gt_images()),
                      (dataset_2_raw_images(), dataset_2_gt_images()),
                      (dataset_3_raw_images(), dataset_3_gt_images())]
    for i, dataset_dir_tuple in enumerate(dataset_dirs):
        raw_dir = dataset_dir_tuple[0]
        for j, raw_image in enumerate(dataset_images[i][0]):
            tif.imsave(os.path.join(raw_dir.name, 'raw{}.tif'.format(j)), raw_image)
        gt_dir = dataset_dir_tuple[1]
        for j, gt_image in enumerate(dataset_images[i][1]):
            tif.imsave(os.path.join(gt_dir.name, 'gt{}.tif'.format(j)), gt_image)
