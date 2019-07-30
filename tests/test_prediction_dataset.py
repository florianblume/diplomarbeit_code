import shutil
import tempfile
import pytest
import numpy as np
import tifffile as tif

from tests import base_test
from tests import conftest

from data import PredictionDataset

def test_raw_images():
    """This test case performs a simple test without transformations and checks
    whether raw images are returned correctly when specifing
    a validation ratio of 0.3 and Noise2Void training.
    """
    dataset = PredictionDataset([conftest.tmp_raw_dir.name])
    assert len(dataset) == 3
    raw_images = np.array(conftest.create_raw_images())
    for i in range(len(dataset)):
        dataset_image = dataset[i]
        raw_image = dataset_image['raw']
        assert np.array_equal(raw_image, raw_images[i])

def test_gt_images():
    """This test case performs a simple test without transformations and checks
    whether raw images are returned correctly when specifing
    a validation ratio of 0.3 and Noise2Void training.
    """
    dataset = PredictionDataset([conftest.tmp_raw_dir.name], 
                                [conftest.tmp_gt_dir.name])
    assert len(dataset) == 3
    raw_images = np.array(conftest.create_raw_images())
    gt_images = np.array(conftest.create_gt_images())
    for i in range(len(dataset)):
        dataset_image = dataset[i]
        raw_image = dataset_image['raw']
        gt_image = dataset_image['gt']
        assert np.array_equal(raw_image, raw_images[i])
        assert np.array_equal(gt_image, gt_images[i])

def test_multi_path():
    dataset = PredictionDataset([conftest.tmp_raw_dir.name, conftest.tmp_raw_dir_2.name],
                                [conftest.tmp_gt_dir.name, conftest.tmp_gt_dir_2.name])
    raw = conftest.create_raw_images()
    raw2 = conftest.create_raw_images_2()
    raw.extend(raw2)
    raw = np.array(raw)
    gt = conftest.create_gt_images()
    gt2 = conftest.create_gt_images_2()
    gt.extend(gt2)
    gt = np.array(gt)
    for i, sample in enumerate(dataset):
        raw_ = sample['raw']
        gt_ = sample['gt']
        assert np.array_equal(raw[i], raw_)
        assert np.array_equal(gt[i], gt_)