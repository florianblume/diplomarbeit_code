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

# Test for class RandomRotation

# Tests for class RandomFlip

def test_random_flip_raw(raw):
    raw_image = raw['raw'].copy()
    raw_image_flipped = np.array(np.flip(raw_image))
    # Seed such that the flip is true
    np.random.seed(0)
    result = RandomFlip()(raw)['raw']
    assert np.array_equal(raw_image_flipped, result)
    # Seed such that the flip is false
    np.random.seed(1)
    result = RandomFlip()(raw)['raw']
    assert np.array_equal(raw_image, result)

def test_random_flip_raw_gt(raw_gt):
    raw_image = raw_gt['raw'].copy()
    raw_image_flipped = np.array(np.flip(raw_image))
    gt_image = raw_gt['gt'].copy()
    gt_image_flipped = np.array(np.flip(gt_image))
    # Seed such that the flip is true
    np.random.seed(0)
    result = RandomFlip()(raw_gt)
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
