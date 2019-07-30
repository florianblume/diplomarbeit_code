import os
import shutil
import tempfile
import pytest
import numpy as np
from torchvision.transforms import Compose

from tests import base_test

import util
from data import TrainingDataset
from data.transforms import RandomCrop, RandomFlip, RandomRotation,\
                            ConvertToFormat, ToTensor
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

def test_raw_images_with_val():
    """This test case performs a simple test without transformations and checks
    whether raw images are returned correctly when specifing
    a validation ratio of 0.3 and Noise2Void training.
    """
    dataset = TrainingDataset(tmp_raw_dir.name, val_ratio=0.3,
                              add_normalization_transform=False)
    assert len(dataset) == 2
    assert dataset.get_validation_samples().shape[0] == 1
    raw_images = np.array(create_raw_images())
    np.random.seed(constants.NP_RANDOM_SEED)
    raw_images = util.shuffle(raw_images)
    for i in range(len(dataset)):
        dataset_image = dataset[i]
        raw_image = dataset_image['raw']
        # N2V i.e. some pixels have randomly been switched, we don't test for those
        mask = dataset_image['mask'].astype(np.bool)
        mask = np.invert(mask)
        assert np.array_equal(raw_image[mask], raw_images[i][mask])

def test_raw_images_without_val():
    """This test case performs a simple test without transformations and checks
    whether raw images are returned correctly when specifing
    a validation ratio of 0 and Noise2Void training.
    """
    dataset = TrainingDataset(tmp_raw_dir.name, val_ratio=0,
                              add_normalization_transform=False)
    assert len(dataset) == 3
    assert dataset.get_validation_samples().shape[0] == 0
    raw_images = np.array(create_raw_images())
    np.random.seed(constants.NP_RANDOM_SEED)
    raw_images = util.shuffle(raw_images)
    for i in range(len(dataset)):
        dataset_image = dataset[i]
        raw_image = dataset_image['raw']
        # N2V i.e. some pixels have randomly been switched, we don't test for those
        mask = dataset_image['mask'].astype(np.bool)
        mask = np.invert(mask)
        assert np.array_equal(raw_image[mask], raw_images[i][mask])

def test_raw_gt_images_with_val():
    """This test case performs a simple test without transformations and checks
    whether raw and ground-truth images are returned correctly when specifing
    a validation ratio of 0.3.
    """
    dataset = TrainingDataset(tmp_raw_dir.name, tmp_gt_dir.name,
                              val_ratio=0.3, add_normalization_transform=False)
    assert len(dataset) == 2
    assert dataset.get_validation_samples().shape[0] == 1
    raw_images = np.array(create_raw_images())
    gt_images = np.array(create_gt_images())
    # Same seed that the dataset uses
    np.random.seed(constants.NP_RANDOM_SEED)
    raw_images, gt_images = util.joint_shuffle(raw_images,
                                               gt_images,
                                               constants.NP_RANDOM_SEED)
    for i in range(len(dataset)):
        dataset_image = dataset[i]
        # No squeeze necessary here as we have no ToTensor transform which
        # adds a dimension for pytroch
        raw_image = dataset_image['raw']
        gt_image = dataset_image['gt']
        assert np.array_equal(raw_image, raw_images[i])
        assert np.array_equal(gt_image, gt_images[i])

def test_raw_gt_num_images_without_val():
    """This test case performs a simple test without transformations and checks
    whether raw and ground-truth images are returned correctly when specifing
    a validation ratio of 0.
    """
    dataset = TrainingDataset(tmp_raw_dir.name, tmp_gt_dir.name,
                              val_ratio=0, add_normalization_transform=False)
    assert len(dataset) == 3
    assert dataset.get_validation_samples().shape[0] == 0
    raw_images = np.array(create_raw_images())
    gt_images = np.array(create_gt_images())
    # Same seed that the dataset uses
    np.random.seed(constants.NP_RANDOM_SEED)
    raw_images, gt_images = util.joint_shuffle(raw_images,
                                               gt_images,
                                               constants.NP_RANDOM_SEED)
    for i in range(len(dataset)):
        dataset_image = dataset[i]
        # No squeeze necessary here as we have no ToTensor transform which
        # adds a dimension for pytroch
        raw_image = dataset_image['raw']
        gt_image = dataset_image['gt']
        assert np.array_equal(raw_image, raw_images[i])
        assert np.array_equal(gt_image, gt_images[i])

def test_raw_mean_std():
    """This test case tests whether mean and standard deviation of the data
    is computed correctly by the dataset.
    """
    dataset = TrainingDataset(tmp_raw_dir.name, val_ratio=0)
    raw_images = create_raw_images()
    mean = np.mean(raw_images)
    std = np.std(raw_images)
    assert mean == dataset.mean()
    assert std == dataset.std()

def test_raw_transforms():
    """This test case tests whether the training dataset correctly applies the
    specified transformations to the data if we want to perform N2V training,
    i.e. we don't specify gt images and the hot pixels have to be replaced by
    their neighbors. The actual hot pixel replacement is not tested because
    it depends on many random draws.
    """
    raw_images = create_raw_images()
    crop_width = 20
    crop_height = 20
    transforms = [RandomCrop(crop_width, crop_height),
                  RandomFlip(),
                  RandomRotation(),
                  ToTensor()]
    # Setting the seed is actually not necessary because it gets set to this
    # constant by the dataset already on init
    dataset = TrainingDataset(tmp_raw_dir.name,
                              transforms=transforms,
                              add_normalization_transform=True,
                              num_pixels=16,
                              seed=constants.NP_RANDOM_SEED)
    # The following numbers are the outcome within the transforms when setting
    # the numpy seed to 2
    np.random.seed(2)
    # For random crop
    x = 8
    y = 8
    flip = True
    rot = 3

    for i, image in enumerate(raw_images):
        image = image[y:y+crop_height, x:x+crop_width]
        image = np.flip(image)
        image = np.rot90(image, rot)
        image = util.normalize(image, dataset.mean(), dataset.std())
        raw_images[i] = image

    # Dataset shuffles the data, too
    raw_images = util.shuffle(np.array(raw_images), constants.NP_RANDOM_SEED)

    np.random.seed(2)
    for i, converted_image in enumerate(dataset):
        raw_image = converted_image['raw'].numpy()
        # Squeeze to get rid of unnecessary torch dimensions
        raw_image = np.squeeze(raw_image)
        mask = converted_image['mask'].astype(np.bool)
        mask = np.invert(mask)
        # This statement is not true as we obtain data in a Noise2Void manner, i.e.
        # hot pixels are replaced by their neighbors. We checked the results
        # visually as this was the cheapest solution.
        assert np.array_equal(raw_images[i][mask], raw_image[mask])
        # This seed is important here because we iterate over the dataset
        # and in its __getitem__ method we need this seed to be set again
        np.random.seed(2)

def test_raw_gt_transforms():
    """This test case tests whether the training dataset correctly applies the
    specified transformations to both the raw and ground-truth data.
    """
    raw_images = create_raw_images()
    gt_images = create_gt_images()
    crop_width = 20
    crop_height = 20
    transforms = [RandomCrop(crop_width, crop_height),
                  RandomFlip(),
                  RandomRotation(),
                  ToTensor()]
    # Setting the seed is actually not necessary because it gets set to this
    # constant by the dataset already on init
    dataset = TrainingDataset(tmp_raw_dir.name,
                              tmp_gt_dir.name,
                              transforms=transforms,
                              add_normalization_transform=True,
                              num_pixels=16,
                              seed=constants.NP_RANDOM_SEED)
    # The following numbers are the outcome within the transforms when setting
    # the numpy seed to 2
    np.random.seed(2)
    # For random crop
    x = 8
    y = 8
    flip = True
    rot = 3

    for i, raw_image in enumerate(raw_images):
        raw_image = raw_image[y:y+crop_height, x:x+crop_width]
        raw_image = np.flip(raw_image)
        raw_image = np.rot90(raw_image, rot)
        raw_image = util.normalize(raw_image, dataset.mean(), dataset.std())
        raw_images[i] = raw_image

        gt_image = gt_images[i]
        gt_image = gt_image[y:y+crop_height, x:x+crop_width]
        gt_image = np.flip(gt_image)
        gt_image = np.rot90(gt_image, rot)
        gt_image = util.normalize(gt_image, dataset.mean(), dataset.std())
        gt_images[i] = gt_image


    # Dataset shuffles the data, too
    raw_images, gt_images = util.joint_shuffle(np.array(raw_images),
                                               np.array(gt_images),
                                               constants.NP_RANDOM_SEED)

    np.random.seed(2)
    for i, converted_image in enumerate(dataset):
        raw_image = converted_image['raw'].numpy()
        gt_image = converted_image['gt'].numpy()
        # Squeeze to get rid of unnecessary torch dimensions
        raw_image = np.squeeze(raw_image)
        gt_image = np.squeeze(gt_image)
        mask = converted_image['mask'].astype(np.bool)
        mask = np.invert(mask)
        # This statement is not true as we obtain data in a Noise2Void manner, i.e.
        # hot pixels are replaced by their neighbors. We checked the results
        # visually as this was the cheapest solution.
        assert np.array_equal(raw_images[i][mask], raw_image[mask])
        assert np.array_equal(gt_images[i][mask], gt_image[mask])
        # This seed is important here because we iterate over the dataset
        # and in its __getitem__ method we need this seed to be set again
        np.random.seed(2)