import os
import shutil
import pytest
import numpy as np

from data import TrainingDataset

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

def test_num_images():
    dataset = TrainingDataset('tmp/testdata/raw', val_ratio=0.3)
    assert len(dataset) == 2
    assert dataset.get_validation_images().shape[0] == 1
