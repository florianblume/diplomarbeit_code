import os
import shutil
import math
import pytest
import numpy as np
from torchvision.transforms import Compose

from data import TrainingDataset
from data.transforms import *
import constants

tmp_raw_dir = 'tmp/testdata/raw'
tmp_gt_dir = 'tmp/testdata/gt'

def create_raw_images():
    raw_image_1 = np.arange(16).reshape(4, 4)
    raw_image_2 = np.arange(16, 32).reshape(4, 4)
    raw_image_3 = np.arange(32, 48).reshape(4, 4)
    return [raw_image_1, raw_image_2, raw_image_3]

def create_gt_images():
    gt_image_1 = np.arange(64, 80).reshape(4, 4)
    gt_image_2 = np.arange(80, 96).reshape(4, 4)
    gt_image_3 = np.arange(96, 112).reshape(4, 4)
    return [gt_image_1, gt_image_2, gt_image_3]

@pytest.fixture
def setup_module(test_training_dataset):
    os.makedirs(tmp_raw_dir)
    os.makedirs(tmp_gt_dir)
    raw_images = create_raw_images()
    gt_images = create_gt_images()
    for i, raw_image in enumerate(raw_images):
        np.save(tmp_raw_dir + '/raw1.npy', raw_image)
        np.save(tmp_gt_dir + '/gt1.npy', gt_images[i])
    
@pytest.fixture
def teardown_module(test_training_dataset):
    shutil.rmtree('tmp')

def test_raw_num_images_with_val():
    dataset = TrainingDataset('tmp/testdata/raw', val_ratio=0.3)
    assert len(dataset) == 2
    assert dataset.get_validation_samples().shape[0] == 1

def test_raw_num_images_without_val():
    dataset = TrainingDataset('tmp/testdata/raw', val_ratio=0)
    assert len(dataset) == 3
    assert dataset.get_validation_samples().shape[0] == 0

def test_raw_mean_std():
    dataset = TrainingDataset('tmp/testdata/raw', val_ratio=0)
    raw_images = create_raw_images()
    mean = np.mean(raw_images)
    std = dataset.std(raw_images)
    assert mean == dataset.mean()
    assert std == dataset.std()

def test_raw_transforms():
    raw_images = create_raw_images()
    crop_width = 100
    crop_height = 100
    composite = Compose([RandomCrop(crop_width, crop_height),
                            RandomFlip(),
                            RandomRotation(),
                            ConvertToFormat('float64'),
                            ToTensor()])
    # Setting the seed is actually not necessary because it gets set to this
    # constant by the dataset already on init
    dataset = TrainingDataset('tmp/testdata/raw', 
                              transform=composite,
                              seed=constants.NP_RANDOM_SEED)
    # We seed numpy so we can predict the outcome
    np.random.seed(2)
    # For random crop
    x = 0
    y = 1
    flip = True
    rot = 1

    for image in raw_images:
        image = image[y:y+crop_height, x:x+crop_width]
        image = np.flip(image)
        image = np.rot90(image, rot)

    # That's what the dataset does internally
    np.random.seed(constants.NP_RANDOM_SEED)
    np.random.shuffle(raw_images)

    converted_images = []
    np.random.seed(2)
    for i in len(dataset):
        converted_image = dataset[i]
        # Squeeze to get rid of unnecessary torch dimensions
        converted_images.append(converted_image.numpy().squeeze())
        np.random.seed(2)

    converted_images = np.array(converted_images)
    assert np.array_equal(raw_images, converted_images)
