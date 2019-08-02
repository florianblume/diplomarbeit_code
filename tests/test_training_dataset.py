import numpy as np

from tests import base_test
from tests import conftest

import util
from data import TrainingDataset
from data.transforms import RandomCrop, RandomFlip, RandomRotation, ToTensor
import constants

def abstract_raw_images_test(all_size, val_ratio, train_size, val_size,
                             keep_in_memory):
    dataset = TrainingDataset(conftest.tmp_raw_dir.name, val_ratio=val_ratio,
                              add_normalization_transform=False,
                              keep_in_memory=keep_in_memory)
    assert len(dataset) == all_size
    assert len(dataset.train_indices) == train_size
    assert len(dataset.val_indices) == val_size
    raw_images = np.array(conftest.create_raw_images())
    for i in range(len(dataset)):
        dataset_image = dataset[i]
        raw_image = dataset_image['raw']
        gt_image = dataset_image['gt']
        # N2V i.e. some pixels have randomly been switched, we don't test for those
        mask = dataset_image['mask'].astype(np.bool)
        mask = np.invert(mask)
        np.save('gt.npy', gt_image)
        np.save('raw.npy', raw_images[i])
        assert np.array_equal(raw_image[mask], raw_images[i][mask])
        # Check that noise 2 void has been successfully prepared
        assert not np.array_equal(raw_image, raw_images[i])
        assert np.array_equal(gt_image, raw_images[i])

def test_raw_images_with_val():
    """This test case performs a simple test without transformations and checks
    whether raw images are returned correctly when specifing
    a validation ratio of 0.3 and Noise2Void training.
    """
    abstract_raw_images_test(3, 0.4, 2, 1, True)

def test_raw_images_without_val():
    """This test case performs a simple test without transformations and checks
    whether raw images are returned correctly when specifing
    a validation ratio of 0 and Noise2Void training.
    """
    abstract_raw_images_test(3, 0.0, 3, 0, True)

def test_raw_images_with_val_on_demand():
    """This test case performs a simple test without transformations and checks
    whether raw images are returned correctly when specifing
    a validation ratio of 0.3 and Noise2Void training.
    """
    abstract_raw_images_test(3, 0.4, 2, 1, False)

def test_raw_images_without_val_on_demand():
    """This test case performs a simple test without transformations and checks
    whether raw images are returned correctly when specifing
    a validation ratio of 0 and Noise2Void training.
    """
    abstract_raw_images_test(3, 0.0, 3, 0, False)

def abstract_raw_gt_images_test(all_size, val_ratio, train_size, val_size,
                             keep_in_memory):
    dataset = TrainingDataset(conftest.tmp_raw_dir.name,
                              conftest.tmp_gt_dir.name,
                              val_ratio=val_ratio,
                              add_normalization_transform=False,
                              keep_in_memory=keep_in_memory)
    assert len(dataset) == all_size
    assert len(dataset.train_indices) == train_size
    assert len(dataset.val_indices) == val_size
    raw_images = np.array(conftest.create_raw_images())
    gt_images = np.array(conftest.create_gt_images())
    for i in range(len(dataset)):
        dataset_image = dataset[i]
        # No squeeze necessary here as we have no ToTensor transform which
        # adds a dimension for pytroch
        raw_image = dataset_image['raw']
        gt_image = dataset_image['gt']
        assert np.array_equal(raw_image, raw_images[i])
        assert np.array_equal(gt_image, gt_images[i])

def test_raw_gt_images_with_val():
    """This test case performs a simple test without transformations and checks
    whether raw and ground-truth images are returned correctly when specifing
    a validation ratio of 0.3.
    """
    abstract_raw_gt_images_test(3, 0.4, 2, 1, True)

def test_raw_gt_num_images_without_val():
    """This test case performs a simple test without transformations and checks
    whether raw and ground-truth images are returned correctly when specifing
    a validation ratio of 0.
    """
    abstract_raw_gt_images_test(3, 0.0, 3, 0, True)

def test_raw_gt_images_with_val_on_demand():
    """This test case performs a simple test without transformations and checks
    whether raw and ground-truth images are returned correctly when specifing
    a validation ratio of 0.3.
    """
    abstract_raw_gt_images_test(3, 0.4, 2, 1, False)

def test_raw_gt_num_images_without_val_on_demand():
    """This test case performs a simple test without transformations and checks
    whether raw and ground-truth images are returned correctly when specifing
    a validation ratio of 0.
    """
    abstract_raw_gt_images_test(3, 0.0, 3, 0, False)

def test_raw_mean_std():
    """This test case tests whether mean and standard deviation of the data
    is computed correctly by the dataset.
    """
    dataset = TrainingDataset(conftest.tmp_raw_dir.name, val_ratio=0)
    raw_images = conftest.create_raw_images()
    mean = np.mean(raw_images)
    std = np.std(raw_images)
    assert mean == dataset.mean
    assert std == dataset.std

def test_raw_transforms():
    """This test case tests whether the training dataset correctly applies the
    specified transformations to the data if we want to perform N2V training,
    i.e. we don't specify gt images and the hot pixels have to be replaced by
    their neighbors. The actual hot pixel replacement is not tested because
    it depends on many random draws.
    """
    raw_images = conftest.create_raw_images()
    crop_width = 20
    crop_height = 20
    transforms = [RandomCrop(crop_width, crop_height),
                  RandomFlip(),
                  RandomRotation(),
                  ToTensor()]
    # Setting the seed is actually not necessary because it gets set to this
    # constant by the dataset already on init
    dataset = TrainingDataset(conftest.tmp_raw_dir.name,
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
        image = util.normalize(image, dataset.mean, dataset.std)
        raw_images[i] = image

    np.random.seed(2)
    for i, converted_image in enumerate(dataset):
        raw_image = converted_image['raw'].numpy()
        # Squeeze to get rid of unnecessary torch dimensions
        raw_image = np.squeeze(raw_image)
        mask = converted_image['mask'].numpy().squeeze().astype(np.bool)
        mask = np.invert(mask)
        # We only check equality where the mask is false because the other pixels
        # are the hot pixels that have been replaced
        assert np.array_equal(raw_images[i][mask], raw_image[mask])
        # Also check that the hot pixels have been replaced somehow
        assert not np.array_equal(raw_images[i], raw_image)
        # This seed is important here because we iterate over the dataset
        # and in its __getitem__ method we need this seed to be set again
        np.random.seed(2)

def test_raw_gt_transforms():
    """This test case tests whether the training dataset correctly applies the
    specified transformations to both the raw and ground-truth data.
    """
    raw_images = conftest.create_raw_images()
    gt_images = conftest.create_gt_images()
    crop_width = 20
    crop_height = 20
    transforms = [RandomCrop(crop_width, crop_height),
                  RandomFlip(),
                  RandomRotation(),
                  ToTensor()]
    # Setting the seed is actually not necessary because it gets set to this
    # constant by the dataset already on init
    dataset = TrainingDataset(conftest.tmp_raw_dir.name,
                              conftest.tmp_gt_dir.name,
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
        raw_image = util.normalize(raw_image, dataset.mean, dataset.std)
        raw_images[i] = raw_image

        gt_image = gt_images[i]
        gt_image = gt_image[y:y+crop_height, x:x+crop_width]
        gt_image = np.flip(gt_image)
        gt_image = np.rot90(gt_image, rot)
        gt_image = util.normalize(gt_image, dataset.mean, dataset.std)
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
        mask = converted_image['mask'].numpy().squeeze().astype(np.bool)
        mask_test = np.ones_like(mask)
        assert np.array_equal(mask, mask_test)
        mask = np.invert(mask)
        # No need to check for the inequality like in the test case above
        # because we gave gt data (i.e. clean targets) and thus the mask should
        # be 1 everywhere
        assert np.array_equal(raw_images[i][mask], raw_image[mask])
        assert np.array_equal(gt_images[i][mask], gt_image[mask])
        # This seed is important here because we iterate over the dataset
        # and in its __getitem__ method we need this seed to be set again
        np.random.seed(2)
    
def abstract_format_to_test(keep_in_memory):
    dataset = TrainingDataset(conftest.tmp_raw_dir.name,
                              conftest.tmp_gt_dir.name,
                              add_normalization_transform=False,
                              convert_to_format='uint8',
                              keep_in_memory=keep_in_memory,
                              num_pixels=16,
                              seed=constants.NP_RANDOM_SEED)
    for _, sample in enumerate(dataset):
        assert sample['raw'].dtype == np.uint8
        assert sample['gt'].dtype == np.uint8
    
def test_format_to_in_memory():
    abstract_format_to_test(True)

def test_format_to_on_demand():
    abstract_format_to_test(False)

def abstract_image_factor_test(keep_in_memory):
    dataset = TrainingDataset(conftest.tmp_raw_dir.name,
                              conftest.tmp_single_gt_dir.name,
                              val_ratio=0,
                              add_normalization_transform=False,
                              keep_in_memory=keep_in_memory)
    assert len(dataset.raw_image_paths) == 3
    assert len(dataset.gt_image_paths) == 1
    raw_images = np.array(conftest.create_raw_images())
    gt_image = conftest.create_single_gt_image()
    for i in range(len(dataset)):
        dataset_image = dataset[i]
        # No squeeze necessary here as we have no ToTensor transform which
        # adds a dimension for pytroch
        raw_image = dataset_image['raw']
        gt_image = dataset_image['gt']
        assert np.array_equal(raw_image, raw_images[i])
        assert np.array_equal(gt_image, gt_image)

def test_image_factor():
    abstract_image_factor_test(True)

def test_image_factor_on_demand():
    abstract_image_factor_test(False)
