import shutil
import tempfile
import pytest
import numpy as np
from torchvision.transforms import Compose

from tests import base_test

from data import PredictionDataset
import constants

tmp_raw_dir = tempfile.TemporaryDirectory()
tmp_gt_dir = tempfile.TemporaryDirectory()

def create_raw_images():
    raw_image_1 = np.arange(900).reshape(30, 30)
    raw_image_2 = np.arange(900, 2 * 900).reshape(30, 30)
    raw_image_3 = np.arange(2 * 900, 3 * 900).reshape(30, 30)
    return [raw_image_1, raw_image_2, raw_image_3]

def create_gt_images():
    gt_image_1 = np.arange(3 * 900, 4 * 900).reshape(30, 30)
    gt_image_2 = np.arange(4 * 900, 5 * 900).reshape(30, 30)
    gt_image_3 = np.arange(5 * 900, 6 * 900).reshape(30, 30)
    return [gt_image_1, gt_image_2, gt_image_3]

@pytest.fixture(scope='module', autouse=True)
def setup_module():
    raw_images = create_raw_images()
    gt_images = create_gt_images()
    for i, raw_image in enumerate(raw_images):
        np.save(tmp_raw_dir.name + '/raw{}.npy'.format(i), raw_image)
        np.save(tmp_gt_dir.name + '/gt{}.npy'.format(i), gt_images[i])
    def fin():
        shutil.rmtree(tmp_raw_dir.name)
        shutil.rmtree(tmp_gt_dir.name)

def test_raw_images():
    """This test case performs a simple test without transformations and checks
    whether raw images are returned correctly when specifing
    a validation ratio of 0.3 and Noise2Void training.
    """
    dataset = PredictionDataset(tmp_raw_dir.name)
    assert len(dataset) == 3
    raw_images = np.array(create_raw_images())
    for i in range(len(dataset)):
        dataset_image = dataset[i]
        raw_image = dataset_image['raw']
        assert np.array_equal(raw_image, raw_images[i])

def test_gt_images():
    """This test case performs a simple test without transformations and checks
    whether raw images are returned correctly when specifing
    a validation ratio of 0.3 and Noise2Void training.
    """
    dataset = PredictionDataset(tmp_raw_dir.name, tmp_gt_dir.name)
    assert len(dataset) == 3
    raw_images = np.array(create_raw_images())
    gt_images = np.array(create_gt_images())
    for i in range(len(dataset)):
        dataset_image = dataset[i]
        raw_image = dataset_image['raw']
        gt_image = dataset_image['gt']
        assert np.array_equal(raw_image, raw_images[i])
        assert np.array_equal(gt_image, gt_images[i])