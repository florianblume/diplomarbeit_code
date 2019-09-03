import copy
import pytest
import torch
import numpy as np
from data.transforms import *

@pytest.fixture
def raw():
    return {'raw' : np.arange(16).reshape((4, 4))}

@pytest.fixture
def raw_3():
    return {'raw' : np.arange(48).reshape((4, 4, 3))}

@pytest.fixture
def raw_gt():
    raw = np.arange(16).reshape((4, 4))
    gt = np.arange(16, 32).reshape((4, 4))
    return {'raw' : raw, 'gt' : gt}

@pytest.fixture
def raw_gt_3():
    raw = np.arange(48).reshape((4, 4, 3))
    gt = np.arange(48, 96).reshape((4, 4, 3))
    return {'raw' : raw, 'gt' : gt}

# Tests for class RandomCrop

def test_random_crop(raw_gt):
    raw_image = raw_gt['raw'].copy()
    gt_image = raw_gt['gt'].copy()
    np.random.seed(2)
    width = 2
    height = 2
    # resulting x from the width for seed 2
    x = 0
    # resulting y from the height for seed 2
    y = 1
    raw_cropped = raw_image[y:y+height, x:x+width]
    gt_cropped = gt_image[y:y+height, x:x+width]
    result = RandomCrop(width, height)(raw_gt)
    result_raw = result['raw']
    result_gt = result['gt']
    print(result_gt)
    print(gt_cropped)
    assert np.array_equal(raw_cropped, result_raw)
    assert np.array_equal(gt_cropped, result_gt)

# Test for class RandomRotation

def test_random_rotation_raw_gt(raw_gt):
    raw_image = raw_gt['raw'].copy()
    gt_image = raw_gt['gt'].copy()
    # np.random.randint(0, 4)=2 for seed 3, but calling np.random.randint(0, 4)
    # again gives 0 -> sometimes calling two times gives the same number for a
    # seed which would mean we would not detect a problem of rotating the raw
    # image but not the ground-truth one
    raw_image_rotated = np.rot90(raw_image, 2)
    gt_image_rotated = np.rot90(gt_image, 2)
    np.random.seed(3)
    result = RandomRotation()(raw_gt)
    result_raw_rot = result['raw']
    result_gt_rot = result['gt']
    assert np.array_equal(raw_image_rotated, result_raw_rot)
    assert np.array_equal(gt_image_rotated, result_gt_rot)

# Tests for class RandomFlip

def test_random_flip_raw_gt(raw_gt):
    raw_image = copy.deepcopy(raw_gt['raw'])
    raw_image_flipped = np.array(np.flip(raw_image))
    gt_image = copy.deepcopy(raw_gt['gt'])
    gt_image_flipped = np.array(np.flip(gt_image))
    # Seed such that the flip is true
    np.random.seed(0)
    result = RandomFlip()(copy.deepcopy(raw_gt))
    result_raw_flipped = result['raw']
    result_gt_flipped = result['gt']
    assert np.array_equal(raw_image_flipped, result_raw_flipped)
    assert np.array_equal(gt_image_flipped, result_gt_flipped)
    # Seed such that the flip is false
    np.random.seed(1)
    result = RandomFlip()(raw_gt)
    result_raw = result['raw']
    result_gt = result['gt']
    assert np.array_equal(raw_image, result_raw)
    assert np.array_equal(gt_image, result_gt)

# Tests for class ConvertToFormat

def test_convert_to_format_raw(raw):
    result = ConvertToFormat('float64')(raw)['raw']
    assert result.dtype == np.float64

def test_convert_to_format_raw_gt(raw_gt):
    raw = ConvertToFormat('float64')(raw_gt)['raw']
    gt = ConvertToFormat('float64')(raw_gt)['gt']
    assert raw.dtype == np.float64
    assert gt.dtype == np.float64

# Tests for class ToTensor

def abstract_to_tensor_raw_test(raw, expected, shape):
    result = ToTensor()(raw)['raw']
    assert isinstance(result, torch.Tensor)
    assert result.shape == shape
    # Remove unneccessary dimension if image is grayscale
    result = result.numpy().squeeze()
    assert np.array_equal(expected, result)

def test_to_tensor_raw(raw):
    abstract_to_tensor_raw_test(raw, raw['raw'].copy(), (1, 4, 4))

def test_to_tensor_raw_3(raw_3):
    raw_image = raw_3['raw'].copy()
    # Pytorch switches from [H, W, C] to [C, H, W]
    raw_image = raw_image.transpose((2, 0, 1))
    abstract_to_tensor_raw_test(raw_3, raw_image, (3, 4, 4))

def abstract_to_tensor_raw_gt_test(raw_gt, expected_raw, expected_gt, shape):
    result = ToTensor()(raw_gt)
    raw = result['raw']
    gt = result['gt']
    assert isinstance(raw, torch.Tensor)
    assert raw.shape == shape
    raw = raw.squeeze()
    assert np.array_equal(raw, expected_raw)
    assert isinstance(gt, torch.Tensor)
    assert gt.shape == shape
    gt = gt.squeeze()
    assert np.array_equal(gt, expected_gt)

def test_to_tensor_raw_gt(raw_gt):
    abstract_to_tensor_raw_gt_test(raw_gt,
                                   raw_gt['raw'].copy(),
                                   raw_gt['gt'].copy(),
                                   (1, 4, 4))
    
def test_to_tensor_raw_gt_3(raw_gt_3):
    raw = raw_gt_3['raw'].copy().transpose(2, 0, 1)
    gt = raw_gt_3['gt'].copy().transpose(2, 0, 1)
    abstract_to_tensor_raw_gt_test(raw_gt_3,
                                   raw,
                                   gt,
                                   (3, 4, 4))
