import numpy as np

from tests import base_test
from tests import conftest

from data import PredictionDataset

def test_single_dataset_raw_images():
    dataset = PredictionDataset([conftest.dataset_1_raw_dir.name])
    assert len(dataset) == 4
    raw_images = np.array(conftest.dataset_1_raw_images())
    for i, dataset_image in enumerate(dataset):
        raw_image = dataset_image['raw']
        assert np.array_equal(raw_image, raw_images[i])

def test_single_dataset_gt_images():
    dataset = PredictionDataset([conftest.dataset_1_raw_dir.name],
                                [conftest.dataset_1_gt_dir.name])
    assert len(dataset) == 4
    raw_images = np.array(conftest.dataset_1_raw_images())
    gt_images = np.array(conftest.dataset_1_gt_images())
    factor = int(len(raw_images) / len(gt_images))
    for i, dataset_image in enumerate(dataset):
        raw_image = dataset_image['raw']
        gt_image = dataset_image['gt']
        assert np.array_equal(raw_image, raw_images[i])
        assert np.array_equal(gt_image, gt_images[int(i / factor)])

def test_multi_dataset_raw_images():
    dataset = PredictionDataset([conftest.dataset_1_raw_dir.name, 
                                 conftest.dataset_2_raw_dir.name])
    raw = conftest.dataset_1_raw_images()
    raw2 = conftest.dataset_2_raw_images()
    raw.extend(raw2)
    raw = np.array(raw)
    for i, sample in enumerate(dataset):
        raw_ = sample['raw']
        assert np.array_equal(raw[i], raw_)

def test_multi_dataset_gt_images():
    dataset = PredictionDataset([conftest.dataset_1_raw_dir.name,
                                 conftest.dataset_2_raw_dir.name,
                                 conftest.dataset_3_raw_dir.name],
                                [conftest.dataset_1_gt_dir.name,
                                 conftest.dataset_2_gt_dir.name,
                                 conftest.dataset_3_gt_dir.name])
    raw = conftest.dataset_1_raw_images()
    raw2 = conftest.dataset_2_raw_images()
    raw3 = conftest.dataset_3_raw_images()
    gt = conftest.dataset_1_gt_images()
    gt2 = conftest.dataset_2_gt_images()
    gt3 = conftest.dataset_3_gt_images()
    factors = [int(len(raw) / len(gt)),
               int(len(raw2) / len(gt2)),
               int(len(raw3) / len(gt3))]
    raw.extend(raw2)
    raw.extend(raw3)
    raw = np.array(raw)
    for i, sample in enumerate(dataset):
        if i < 4:
            factor = factors[0]
            current_gt = gt
            gt_idx = i
        elif i < 6:
            factor = factors[1]
            current_gt = gt2
            # because first dataset has 4 entries
            gt_idx = i - 4
        else:
            factor = factors[2]
            current_gt = gt3
            # because second dataset has an additional 2 entries
            gt_idx = i - 6
        raw_ = sample['raw']
        gt_ = sample['gt']
        assert np.array_equal(raw[i], raw_)
        assert np.array_equal(current_gt[int(gt_idx / factor)], gt_)