import numpy as np
import pytest
import torch

from tests import base_test
from tests import conftest

import util
from data import TrainingDataset
from data.transforms import Crop, RandomCrop, ToTensor

def joint_shuffle(inA, inB):
    return inA, inB

def shuffle(inA, seed):
    return inA

def _example_index(length):
    return 0

def _stratified_coord_x(max):
    return np.random.randint(max)

def _stratified_coord_y(max):
    return np.random.randint(max)

def _hot_pixel_replacement_index(length):
    return np.random.randint(length)

def _train_indices_permutation(indices):
    return indices

def _val_indices_permutation(indices):
    return indices

def _dataset_index_even_single_dataset(num_datasets):
    return 0

def _dataset_index_proportional_single_dataset(num_indices):
    # Return 0 here as we only have one dataset and want to use up the indices
    return 0

def _dataset_index_proportional(dataset_sizes):
    probabilities = dataset_sizes / np.sum(dataset_sizes)
    return np.random.choice(len(dataset_sizes), 1, p=probabilities)[0]

# Classes for multi dataset dataset-index generation

class IndexEvenGenerator():

    counter = 0

    @staticmethod
    def index(num_datasets):
        index = IndexEvenGenerator.counter
        IndexEvenGenerator.counter += 1
        IndexEvenGenerator.counter %= len(num_datasets)
        return index

class IndexProportionalGenerator():

    counter = 0

    @staticmethod
    def index_without_val(num_indices):
        # Very ugly but I couldn't get it to work otherwise...
        if IndexProportionalGenerator.counter < 4:
            result = 0
        elif IndexProportionalGenerator.counter < 6:
            # now all indices of first dataset have been used up and refilled, i.e.
            # we need to continued at len(dataset_1)
            result = 1
        elif IndexProportionalGenerator.counter < 9:
            # now all indices of the second dataset have been used up and refilled,
            # i.e. we need to continue at len(dataset_1) + len(dataset_2)
            result = 2
        IndexProportionalGenerator.counter += 1
        print('res', result)
        if IndexProportionalGenerator.counter == 9:
            IndexProportionalGenerator.counter = 0
        return result

    @staticmethod
    def index_with_val(num_indices):
        # Very ugly but I couldn't get it to work otherwise...
        if IndexProportionalGenerator.counter < 2:
            result = 0
        elif IndexProportionalGenerator.counter < 3:
            # now all indices of first dataset have been used up and refilled, i.e.
            # we need to continued at len(dataset_1)
            result = 1
        elif IndexProportionalGenerator.counter < 5:
            # now all indices of the second dataset have been used up and refilled,
            # i.e. we need to continue at len(dataset_1) + len(dataset_2)
            result = 2
        IndexProportionalGenerator.counter += 1
        if IndexProportionalGenerator.counter == 5:
            IndexProportionalGenerator.counter = 0
        return result

@pytest.fixture(scope='module', autouse=True)
def setup_module():
    """We need to remove the randomness of the dataset in order to test it
    properly.
    """
    util.joint_shuffle = joint_shuffle
    util.shuffle = shuffle
    TrainingDataset._example_index = _example_index
    TrainingDataset._stratified_coord_x = _stratified_coord_x
    TrainingDataset._stratified_coord_y = _stratified_coord_y
    TrainingDataset._hot_pixel_replacement_index = _hot_pixel_replacement_index
    TrainingDataset._train_indices_permutation = _train_indices_permutation
    TrainingDataset._val_indices_permutation = _val_indices_permutation

##################################
############# Single dataset tests
##################################

##### Probabilities proportional to dataset size tests

def abstract_test_single_dataset_mean_std(in_memory):
    # Dataset 1 has 4 raw images and 2 gt images
    dataset_1_raw = conftest.dataset_1_raw_images()
    dataset = TrainingDataset([conftest.dataset_1_raw_dir.name], batch_size=2,
                              val_ratio=0.5, add_normalization_transform=False,
                              keep_in_memory=in_memory)
    assert dataset.mean == np.mean(dataset_1_raw)
    assert dataset.std == np.std(dataset_1_raw)

def test_single_dataset_mean_std_in_memory():
    abstract_test_single_dataset_mean_std(True)

def test_single_dataset_mean_std_on_demand():
    abstract_test_single_dataset_mean_std(False)

def abstract_test_single_dataset_iteration_raw_only_with_val(in_memory):
    # Dataset 1 has 4 raw images and 2 gt images
    dataset_1_raw = conftest.dataset_1_raw_images()
    # Batch size doesn't matter in this case
    dataset = TrainingDataset([conftest.dataset_1_raw_dir.name],
                              val_ratio=0.5, add_normalization_transform=False,
                              keep_in_memory=in_memory)
    TrainingDataset._dataset_index_proportional = _dataset_index_proportional_single_dataset
    assert len(dataset) == 4
    # indices[0] because we only have one dataset
    assert len(dataset.val_indices[0]) == 2
    # We get back a batch of size 4 with all images in order
    for i in range(len(dataset)):
        sample = dataset[i]
        raw = sample['raw'].squeeze()
        gt = sample['gt'].squeeze()
        mask = sample['mask'].astype(np.bool)
        assert not mask.all()
        mask = ~mask
        # Iterating over the dataset uses all images, regardless of whether
        # they are for training or for validation
        assert np.array_equal(raw[mask], dataset_1_raw[i][mask])
        assert not np.array_equal(raw, dataset_1_raw[i])
        assert np.array_equal(sample['gt'], dataset_1_raw[i])
    
def test_single_dataset_iteration_raw_only_with_val_in_memory():
    abstract_test_single_dataset_iteration_raw_only_with_val(True)

def test_single_dataset_iteration_raw_only_with_val_on_demand():
    abstract_test_single_dataset_iteration_raw_only_with_val(False)

def abstract_test_single_dataset_raw_only_with_val(in_memory):
    # Dataset 1 has 4 raw images and 2 gt images
    dataset_1_raw = conftest.dataset_1_raw_images()
    dataset = TrainingDataset([conftest.dataset_1_raw_dir.name], batch_size=2,
                              val_ratio=0.5, add_normalization_transform=False,
                              keep_in_memory=in_memory)
    TrainingDataset._dataset_index_proportional = _dataset_index_proportional_single_dataset
    assert len(dataset) == 4
    # indices[0] because we only have one dataset
    assert len(dataset.train_indices[0]) == 2
    assert len(dataset.val_indices[0]) == 2
    # We get back a batch of size 4 with all images in order
    sample = next(iter(dataset))
    for i, raw in enumerate(sample['raw']):
        mask = sample['mask'][i].astype(np.bool)
        assert not mask.all()
        mask = ~mask
        # Second half are training indices in TrainingDataset
        assert np.array_equal(raw[mask], dataset_1_raw[i + 2][mask])
        assert not np.array_equal(raw, dataset_1_raw[i + 2])
        assert np.array_equal(sample['gt'][i], dataset_1_raw[i + 2])
    for val in dataset.validation_samples():
        for i, raw in enumerate(val['raw']):
            val_raw = raw.squeeze()
            val_gt = val['gt'][i].squeeze()
            mask = val['mask'][i].squeeze().astype(np.bool)
            assert not mask.all()
            mask = ~mask
            assert np.array_equal(val_raw[mask], dataset_1_raw[i][mask])
            assert np.array_equal(val_gt, dataset_1_raw[i])

def test_single_dataset_raw_only_with_val_in_memory():
    abstract_test_single_dataset_raw_only_with_val(True)

def test_single_dataset_raw_only_with_val_on_demand():
    abstract_test_single_dataset_raw_only_with_val(False)

def abstract_test_single_dataset_raw_only(in_memory):
    # Dataset 1 has 4 raw images and 2 gt images
    dataset_1_raw = conftest.dataset_1_raw_images()
    dataset = TrainingDataset([conftest.dataset_1_raw_dir.name], batch_size=4,
                              val_ratio=0, add_normalization_transform=False,
                              keep_in_memory=in_memory)
    TrainingDataset._dataset_index_proportional = _dataset_index_proportional_single_dataset
    assert len(dataset) == 4
    # indices[0] because we only have one dataset
    assert len(dataset.train_indices[0]) == 4
    assert len(dataset.val_indices[0]) == 0
    # We get back a batch of size 4 with all images in order
    sample = next(iter(dataset))
    for i, raw in enumerate(sample['raw']):
        mask = sample['mask'][i].astype(np.bool)
        assert not mask.all()
        mask = ~mask
        # Second half are training indices in TrainingDataset
        assert np.array_equal(raw[mask], dataset_1_raw[i][mask])
        assert not np.array_equal(raw, dataset_1_raw[i])
        assert np.array_equal(sample['gt'][i], dataset_1_raw[i])

def test_single_dataset_raw_only_in_memory():
    abstract_test_single_dataset_raw_only(True)

def test_single_dataset_raw_only_on_demand():
    abstract_test_single_dataset_raw_only(False)

def abstract_test_single_dataset_raw_only_batch_size(in_memory):
    # Dataset 1 has 4 raw images and 2 gt images
    dataset_1_raw = conftest.dataset_1_raw_images()
    # Every image gets used 4 times in this setting
    dataset = TrainingDataset([conftest.dataset_1_raw_dir.name], batch_size=16,
                              val_ratio=0, add_normalization_transform=False,
                              keep_in_memory=in_memory)
    TrainingDataset._dataset_index_proportional = _dataset_index_proportional_single_dataset
    assert len(dataset) == 4
    # indices[0] because we only have one dataset
    assert len(dataset.train_indices[0]) == 4
    assert len(dataset.val_indices[0]) == 0
    # We get back a batch of size 4 with all images in order
    sample = next(iter(dataset))
    for i, raw in enumerate(sample['raw']):
        mask = sample['mask'][i].astype(np.bool)
        assert not mask.all()
        mask = ~mask
        # Second half are training indices in TrainingDataset
        #assert np.array_equal(raw[mask], dataset_1_raw[i][mask])
        assert not np.array_equal(raw, dataset_1_raw[i%4])
        assert np.array_equal(sample['gt'][i], dataset_1_raw[i%4])

def test_single_dataset_raw_only_batch_size_in_memory():
    abstract_test_single_dataset_raw_only_batch_size(True)

def test_single_dataset_raw_only_batch_size_on_demand():
    abstract_test_single_dataset_raw_only_batch_size(False)

def abstract_test_single_dataset_raw_only_with_transforms(in_memory):
    # Dataset 1 has 4 raw images and 2 gt images
    dataset_1_raw = conftest.dataset_1_raw_images()
    train_transforms = [RandomCrop(20, 20), ToTensor()]
    eval_transforms = [Crop(0, 0, 20, 20), ToTensor()]
    dataset = TrainingDataset([conftest.dataset_1_raw_dir.name], batch_size=4,
                              val_ratio=0, add_normalization_transform=True,
                              train_transforms=train_transforms,
                              eval_transforms=eval_transforms,
                              keep_in_memory=in_memory)
    assert len(dataset) == 4
    # indices[0] because we only have one dataset
    assert len(dataset.train_indices[0]) == 4
    assert not dataset.val_indices[0]
    # We get back a batch of size 4 with all images in order
    sample = next(iter(dataset))
    raw = sample['raw']
    assert isinstance(raw, torch.Tensor)
    assert raw.shape == (4, 1, 20, 20)
    gt = sample['gt']
    assert isinstance(gt, torch.Tensor)
    assert gt.shape == (4, 1, 20, 20)

def test_single_dataset_raw_only_with_transforms_in_memory():
    abstract_test_single_dataset_raw_only_with_transforms(True)

def test_single_dataset_raw_only_with_transforms_on_demand():
    abstract_test_single_dataset_raw_only_with_transforms(False)

def abstract_test_single_dataset_raw_gt_with_val(in_memory):
    # Dataset 1 has 4 raw images and 2 gt images
    dataset_1_raw = conftest.dataset_1_raw_images()
    dataset_1_gt = conftest.dataset_1_gt_images()
    factor = 2
    dataset = TrainingDataset([conftest.dataset_1_raw_dir.name],
                              [conftest.dataset_1_gt_dir.name],
                              batch_size=2, val_ratio=0.5,
                              add_normalization_transform=False,
                              keep_in_memory=in_memory)
    TrainingDataset._dataset_index_proportional = _dataset_index_proportional_single_dataset
    assert len(dataset) == 4
    # indices[0] because we only have one dataset
    assert len(dataset.train_indices[0]) == 2
    assert len(dataset.val_indices[0]) == 2
    # We get back a batch of size 4 with all images in order
    sample = next(iter(dataset))
    for i, raw in enumerate(sample['raw']):
        mask = sample['mask'][i].astype(np.bool)
        assert mask.all()
        # Second half are training indices in TrainingDataset
        assert np.array_equal(raw, dataset_1_raw[i + 2])
        assert np.array_equal(sample['gt'][i], dataset_1_gt[int((i + 2) / factor)])
    for val in dataset.validation_samples():
        for i, raw in enumerate(val['raw']):
            val_raw = raw.squeeze()
            val_gt = val['gt'][i].squeeze()
            mask = val['mask'][i].squeeze().astype(np.bool)
            assert mask.all()
            assert np.array_equal(val_raw, dataset_1_raw[i])
            assert np.array_equal(val_gt, dataset_1_gt[int(i / factor)])

def test_single_dataset_raw_gt_with_val_in_memory():
    abstract_test_single_dataset_raw_gt_with_val(True)

def test_single_dataset_raw_gt_with_val_on_demand():
    abstract_test_single_dataset_raw_gt_with_val(False)

##### Even probabilities tests

def abstract_test_single_dataset_raw_only_with_val_even(in_memory):
    # Dataset 1 has 4 raw images and 2 gt images
    dataset_1_raw = conftest.dataset_1_raw_images()
    dataset = TrainingDataset([conftest.dataset_1_raw_dir.name], batch_size=2,
                              val_ratio=0.5, add_normalization_transform=False,
                              distribution_mode='even', keep_in_memory=in_memory)
    TrainingDataset._dataset_index_even = _dataset_index_even_single_dataset
    assert len(dataset) == 4
    # indices[0] because we only have one dataset
    assert len(dataset.train_indices[0]) == 2
    assert len(dataset.val_indices[0]) == 2
    # We get back a batch of size 4 with all images in order
    sample = next(iter(dataset))
    for i, raw in enumerate(sample['raw']):
        mask = sample['mask'][i].astype(np.bool)
        assert not mask.all()
        # Invert for masking non-replaced pixels
        mask = ~mask
        # Second half are training indices in TrainingDataset
        assert np.array_equal(raw[mask], dataset_1_raw[i + 2][mask])
        assert not np.array_equal(raw, dataset_1_raw[i + 2])
        assert np.array_equal(sample['gt'][i], dataset_1_raw[i + 2])
    for val in dataset.validation_samples():
        for i, raw in enumerate(val['raw']):
            val_raw = raw.squeeze()
            val_gt = val['gt'][i].squeeze()
            mask = ~val['mask'][i].squeeze().astype(np.bool)
            assert not mask.all()
            assert np.array_equal(val_raw[mask], dataset_1_raw[i][mask])
            assert np.array_equal(val_gt, dataset_1_raw[i])
    
def test_single_dataset_raw_only_with_val_even_in_memory():
    abstract_test_single_dataset_raw_only_with_val_even(True)
    
def test_single_dataset_raw_only_with_val_even_on_demand():
    abstract_test_single_dataset_raw_only_with_val_even(False)

##################################
############# Multi dataset tests
##################################

def abstract_test_multi_dataset_raw_only(in_memory):
    dataset_1_raw = conftest.dataset_1_raw_images()
    dataset_2_raw = conftest.dataset_2_raw_images()
    dataset_3_raw = conftest.dataset_3_raw_images()
    datasets = [dataset_1_raw, dataset_2_raw, dataset_3_raw]
    dataset = TrainingDataset([conftest.dataset_1_raw_dir.name,
                               conftest.dataset_2_raw_dir.name,
                               conftest.dataset_3_raw_dir.name],
                              batch_size=9, val_ratio=0,
                              add_normalization_transform=False,
                              keep_in_memory=in_memory)
    TrainingDataset._dataset_index_proportional =\
                IndexProportionalGenerator.index_without_val
    assert len(dataset) == 9
    # indices[0] because we only have one dataset
    assert len(dataset.train_indices[0]) == 4
    assert len(dataset.train_indices[1]) == 2
    assert len(dataset.train_indices[2]) == 3
    assert len([idx for sub in dataset.val_indices for idx in sub]) == 0
    dataset_indices = [0, 0, 0, 0, 1, 1, 2, 2, 2]
    raw_indices = [0, 1, 2, 3, 0, 1, 0, 1, 2]
    # We get back a batch of size 4 with all images in order
    sample = next(iter(dataset))
    for i, raw in enumerate(sample['raw']):
        mask = ~sample['mask'][i].astype(np.bool)
        assert not mask.all()
        gt = sample['gt'][i]
        dataset_index = dataset_indices[i]
        raw_index = raw_indices[i]
        # Second half are training indices in TrainingDataset
        test_dataset = datasets[dataset_index]
        assert np.array_equal(raw[mask], test_dataset[raw_index][mask])
        assert not np.array_equal(raw, test_dataset[raw_index])
        assert np.array_equal(gt, test_dataset[raw_index])

def test_multi_dataset_raw_only_in_memory():
    abstract_test_multi_dataset_raw_only(True)

def test_multi_dataset_raw_only_on_demand():
    abstract_test_multi_dataset_raw_only(False)

def abstract_test_multi_dataset_raw_only_with_val(in_memory):
    # Dataset 1 has 4 raw images and 2 gt images
    dataset_1_raw = conftest.dataset_1_raw_images()
    dataset_2_raw = conftest.dataset_2_raw_images()
    dataset_3_raw = conftest.dataset_3_raw_images()
    datasets = [dataset_1_raw, dataset_2_raw, dataset_3_raw]
    dataset = TrainingDataset([conftest.dataset_1_raw_dir.name,
                               conftest.dataset_2_raw_dir.name,
                               conftest.dataset_3_raw_dir.name],
                               batch_size=5, val_ratio=0.5,
                               add_normalization_transform=False,
                               keep_in_memory=in_memory)
    IndexProportionalGenerator.counter = 0
    TrainingDataset._dataset_index_proportional =\
                IndexProportionalGenerator.index_with_val
    assert len([idx for sub in dataset.train_indices for idx in sub]) == 5
    assert len([idx for sub in dataset.val_indices for idx in sub]) == 4
    dataset_indices   = [0, 0, 1, 2, 2]
    raw_train_indices = [2, 3, 1, 1, 2]
    # We get back a batch of size 4 with all images in order
    sample = next(iter(dataset))
    for i, raw in enumerate(sample['raw']):
        mask = ~sample['mask'][i].astype(np.bool)
        # Since N2V training
        assert not mask.all()
        gt = sample['gt'][i]
        dataset_index = dataset_indices[i]
        raw_index = raw_train_indices[i]
        test_dataset = datasets[dataset_index]
        assert np.array_equal(raw[mask], test_dataset[raw_index][mask])
        assert not np.array_equal(raw, test_dataset[raw_index])
        assert np.array_equal(gt, test_dataset[raw_index])
    raw_val_indices = [0, 1, 0, 0]
    for val in dataset.validation_samples():
        for i, raw in enumerate(val['raw']):
            dataset_index = dataset_indices[i]
            raw_index = raw_val_indices[i]
            val_raw = raw.squeeze()
            val_gt = val['gt'][i].squeeze()
            mask = ~val['mask'][i].squeeze().astype(np.bool)
            test_dataset = datasets[dataset_index]
            assert np.array_equal(val_raw[mask], test_dataset[raw_index][mask])
            assert np.array_equal(val_gt, test_dataset[raw_index])

def test_multi_dataset_raw_only_with_val_in_memory():
    abstract_test_multi_dataset_raw_only_with_val(True)

def test_multi_dataset_raw_only_with_val_on_demand():
    abstract_test_multi_dataset_raw_only_with_val(False)

def abstract_test_multi_dataset_raw_gt(in_memory):
    # Dataset 1 has 4 raw images and 2 gt images
    raw_datasets = [conftest.dataset_1_raw_images(),
                    conftest.dataset_2_raw_images(),
                    conftest.dataset_3_raw_images()]
    gt_datasets = [conftest.dataset_1_gt_images(),
                   conftest.dataset_2_gt_images(),
                   conftest.dataset_3_gt_images()]
    dataset = TrainingDataset([conftest.dataset_1_raw_dir.name,
                               conftest.dataset_2_raw_dir.name,
                               conftest.dataset_3_raw_dir.name],
                              [conftest.dataset_1_gt_dir.name,
                               conftest.dataset_2_gt_dir.name,
                               conftest.dataset_3_gt_dir.name],
                              batch_size=9, val_ratio=0,
                              add_normalization_transform=False,
                              keep_in_memory=in_memory)
    IndexProportionalGenerator.counter = 0
    TrainingDataset._dataset_index_proportional =\
                IndexProportionalGenerator.index_without_val
    # indices[0] because we only have one dataset
    assert len(dataset) == 9
    assert len([idx for sub in dataset.train_indices for idx in sub]) == 9
    assert len([idx for sub in dataset.val_indices for idx in sub]) == 0
    dataset_indices = [0, 0, 0, 0, 1, 1, 2, 2, 2]
    raw_indices     = [0, 1, 2, 3, 0, 1, 0, 1, 2]
    gt_indices      = [0, 0, 1, 1, 0, 1, 0, 0, 0]
    # We get back a batch of size 4 with all images in order
    sample = next(iter(dataset))
    for i, raw in enumerate(sample['raw']):
        mask = sample['mask'][i].astype(np.bool)
        assert mask.all()
        gt = sample['gt'][i]
        dataset_index = dataset_indices[i]
        raw_index = raw_indices[i]
        gt_index = gt_indices[i]
        # Second half are training indices in TrainingDataset
        raw_dataset = raw_datasets[dataset_index]
        gt_dataset = gt_datasets[dataset_index]
        assert np.array_equal(raw, raw_dataset[raw_index])
        assert np.array_equal(gt, gt_dataset[gt_index])

def test_multi_dataset_raw_gt_in_memory():
    abstract_test_multi_dataset_raw_gt(True)

def test_multi_dataset_raw_gt_on_demand():
    abstract_test_multi_dataset_raw_gt(False)

def abstract_test_multi_dataset_raw_gt_with_val(in_memory):
    # Dataset 1 has 4 raw images and 2 gt images
    raw_datasets = [conftest.dataset_1_raw_images(),
                    conftest.dataset_2_raw_images(),
                    conftest.dataset_3_raw_images()]
    gt_datasets = [conftest.dataset_1_gt_images(),
                   conftest.dataset_2_gt_images(),
                   conftest.dataset_3_gt_images()]
    dataset = TrainingDataset([conftest.dataset_1_raw_dir.name,
                               conftest.dataset_2_raw_dir.name,
                               conftest.dataset_3_raw_dir.name],
                              [conftest.dataset_1_gt_dir.name,
                               conftest.dataset_2_gt_dir.name,
                               conftest.dataset_3_gt_dir.name],
                              batch_size=5, val_ratio=0.5,
                              add_normalization_transform=False,
                              keep_in_memory=in_memory)
    IndexProportionalGenerator.counter = 0
    TrainingDataset._dataset_index_proportional =\
                IndexProportionalGenerator.index_with_val
    # indices[0] because we only have one dataset
    assert len(dataset) == 9
    assert len([idx for sub in dataset.train_indices for idx in sub]) == 5
    assert len([idx for sub in dataset.val_indices for idx in sub]) == 4
    dataset_indices   = [0, 0, 1, 2, 2]
    raw_train_indices = [2, 3, 1, 1, 2]
    gt_train_indices  = [1, 1, 1, 0, 0]
    # We get back a batch of size 4 with all images in order
    sample = next(iter(dataset))
    for i, raw in enumerate(sample['raw']):
        mask = sample['mask'][i].astype(np.bool)
        assert mask.all()
        gt = sample['gt'][i]
        dataset_index = dataset_indices[i]
        raw_index = raw_train_indices[i]
        gt_index = gt_train_indices[i]
        # Second half are training indices in TrainingDataset
        raw_dataset = raw_datasets[dataset_index]
        gt_dataset = gt_datasets[dataset_index]
        assert np.array_equal(raw, raw_dataset[raw_index])
        assert np.array_equal(gt, gt_dataset[gt_index])
    raw_val_indices = [0, 1, 0, 0]
    gt_val_indices  = [0, 0, 0, 0]
    for val in dataset.validation_samples():
        for i, raw in enumerate(val['raw']):
            mask = val['mask'][i].squeeze().astype(np.bool)
            assert mask.all()
            gt_index = 0
            dataset_index = dataset_indices[i]
            raw_index = raw_val_indices[i]
            gt_index = gt_val_indices[i]
            val_raw = raw.squeeze()
            val_gt = val['gt'][i].squeeze()
            raw_dataset = raw_datasets[dataset_index]
            gt_dataset = gt_datasets[dataset_index]
            assert np.array_equal(val_raw, raw_dataset[raw_index])
            assert np.array_equal(val_gt, gt_dataset[gt_index])

def test_multi_dataset_raw_gt_with_val_in_memory():
    abstract_test_multi_dataset_raw_gt_with_val(True)

def test_multi_dataset_raw_gt_with_val_on_demand():
    abstract_test_multi_dataset_raw_gt_with_val(False)

# Testing even distribution

def abstract_test_multi_dataset_raw_only_even(in_memory):
    raw_datasets = [conftest.dataset_1_raw_images(),
                    conftest.dataset_2_raw_images(),
                    conftest.dataset_3_raw_images()]
    dataset = TrainingDataset([conftest.dataset_1_raw_dir.name,
                               conftest.dataset_2_raw_dir.name,
                               conftest.dataset_3_raw_dir.name],
                              batch_size=10, val_ratio=0,
                              add_normalization_transform=False,
                              keep_in_memory=in_memory)
    IndexEvenGenerator.counter = 0
    TrainingDataset._dataset_index_proportional = IndexEvenGenerator.index
    # indices[0] because we only have one dataset
    assert len(dataset) == 9
    assert len([idx for sub in dataset.train_indices for idx in sub]) == 9
    assert len([idx for sub in dataset.val_indices for idx in sub]) == 0
    # datasets     0  1  2  0  1  2  0  1  2  0
    raw_indices = [0, 0, 0, 1, 1, 1, 2, 0, 2, 3]
    # We get back a batch of size 4 with all images in order
    sample = next(iter(dataset))
    for i, raw in enumerate(sample['raw']):
        mask = ~sample['mask'][i].astype(np.bool)
        assert not mask.all()
        gt = sample['gt'][i]
        dataset_index = i % 3
        raw_dataset = raw_datasets[dataset_index]
        raw_index = raw_indices[i]
        assert np.array_equal(raw[mask], raw_dataset[raw_index][mask])
        assert not np.array_equal(raw, raw_dataset[raw_index])
        assert np.array_equal(gt, raw_dataset[raw_index])

def test_multi_dataset_raw_only_even_in_memory():
    abstract_test_multi_dataset_raw_only_even(True)

def test_multi_dataset_raw_only_even_on_demand():
    abstract_test_multi_dataset_raw_only_even(False)

def abstract_test_multi_dataset_raw_only_with_val_even(in_memory):
    # Dataset 1 has 4 raw images and 2 gt images
    raw_datasets = [conftest.dataset_1_raw_images(),
                    conftest.dataset_2_raw_images(),
                    conftest.dataset_3_raw_images()]
    dataset = TrainingDataset([conftest.dataset_1_raw_dir.name,
                               conftest.dataset_2_raw_dir.name,
                               conftest.dataset_3_raw_dir.name],
                              batch_size=6, val_ratio=0.5,
                              add_normalization_transform=False,
                              keep_in_memory=in_memory)
    IndexEvenGenerator.counter = 0
    TrainingDataset._dataset_index_proportional = IndexEvenGenerator.index
    # Order of indices because we cut off a part for validation
    # dataset      0  1  2  0  1  2
    raw_indices = [2, 1, 1, 3, 1, 2]
    # indices[0] because we only have one dataset
    assert len(dataset) == 9
    assert len([idx for sub in dataset.train_indices for idx in sub]) == 5
    assert len([idx for sub in dataset.val_indices for idx in sub]) == 4
    # We get back a batch of size 4 with all images in order
    sample = next(iter(dataset))
    for i, raw in enumerate(sample['raw']):
        mask = ~sample['mask'][i].astype(np.bool)
        assert not mask.all()
        gt = sample['gt'][i]
        # We are looping through the datasets evenly
        dataset_index = i % 3
        raw_dataset = raw_datasets[dataset_index]
        gt_dataset = gt_datasets[dataset_index]
        raw_index = raw_indices[i]
        assert np.array_equal(raw[mask], raw_dataset[raw_index][mask])
        assert not np.array_equal(raw, raw_dataset[raw_index])
        assert np.array_equal(gt, raw_dataset[raw_index])
    # datasets         0  1  2  0
    val_raw_indices = [0, 0, 0, 1]
    for i, val in enumerate(dataset.validation_samples()):
        mask = ~val['mask'][i].astype(np.bool)
        assert not mask.all()
        gt = val['gt'][i]
        # We are looping through the datasets evenly
        dataset_index = i % 3
        raw_dataset = raw_datasets[dataset_index]
        raw_index = val_raw_indices[i]
        raw = raw_dataset[raw_index]
        assert np.array_equal(raw[mask], raw_dataset[raw_index][mask])
        assert not np.array_equal(raw, raw_dataset[raw_index])
        assert np.array_equal(gt, gt_dataset[gt_index])


def test_multi_dataset_raw_only_with_val_even_in_memory():
    abstract_test_multi_dataset_raw_gt_with_val_even(True)

def test_multi_dataset_raw_only_with_val_even_on_demand():
    abstract_test_multi_dataset_raw_gt_with_val_even(False)

def abstract_test_multi_dataset_raw_gt_even(in_memory):
    # Dataset 1 has 4 raw images and 2 gt images
    raw_datasets = [conftest.dataset_1_raw_images(),
                    conftest.dataset_2_raw_images(),
                    conftest.dataset_3_raw_images()]
    gt_datasets = [conftest.dataset_1_gt_images(),
                   conftest.dataset_2_gt_images(),
                   conftest.dataset_3_gt_images()]
    dataset = TrainingDataset([conftest.dataset_1_raw_dir.name,
                               conftest.dataset_2_raw_dir.name,
                               conftest.dataset_3_raw_dir.name],
                              [conftest.dataset_1_gt_dir.name,
                               conftest.dataset_2_gt_dir.name,
                               conftest.dataset_3_gt_dir.name],
                              batch_size=10, val_ratio=0,
                              add_normalization_transform=False,
                              keep_in_memory=in_memory)
    IndexEvenGenerator.counter = 0
    TrainingDataset._dataset_index_proportional = IndexEvenGenerator.index
    # indices[0] because we only have one dataset
    assert len(dataset) == 9
    assert len([idx for sub in dataset.train_indices for idx in sub]) == 9
    assert len([idx for sub in dataset.val_indices for idx in sub]) == 0
    # datasets     0  1  2  0  1  2  0  1  2  0
    raw_indices = [0, 0, 0, 1, 1, 1, 2, 0, 2, 3]
    gt_indices  = [0, 0, 0, 0, 1, 0, 1, 0, 0, 1]
    # We get back a batch of size 4 with all images in order
    sample = next(iter(dataset))
    for i, raw in enumerate(sample['raw']):
        mask = sample['mask'][i].astype(np.bool)
        assert mask.all()
        gt = sample['gt'][i]
        # We are looping through the datasets evenly
        dataset_index = i % 3
        raw_dataset = raw_datasets[dataset_index]
        gt_dataset = gt_datasets[dataset_index]
        raw_index = raw_indices[i]
        gt_index = gt_indices[i]
        assert np.array_equal(raw, raw_dataset[raw_index])
        assert np.array_equal(gt, gt_dataset[gt_index])

def test_multi_dataset_raw_gt_even_in_memory():
    abstract_test_multi_dataset_raw_gt_even(True)

def test_multi_dataset_raw_gt_even_on_demand():
    abstract_test_multi_dataset_raw_gt_even(False)

def abstract_test_multi_dataset_raw_gt_with_val_even(in_memory):
    # Dataset 1 has 4 raw images and 2 gt images
    raw_datasets = [conftest.dataset_1_raw_images(),
                    conftest.dataset_2_raw_images(),
                    conftest.dataset_3_raw_images()]
    gt_datasets = [conftest.dataset_1_gt_images(),
                   conftest.dataset_2_gt_images(),
                   conftest.dataset_3_gt_images()]
    dataset = TrainingDataset([conftest.dataset_1_raw_dir.name,
                               conftest.dataset_2_raw_dir.name,
                               conftest.dataset_3_raw_dir.name],
                              [conftest.dataset_1_gt_dir.name,
                               conftest.dataset_2_gt_dir.name,
                               conftest.dataset_3_gt_dir.name],
                              batch_size=6, val_ratio=0.5,
                              add_normalization_transform=False,
                              keep_in_memory=in_memory)
    IndexEvenGenerator.counter = 0
    TrainingDataset._dataset_index_proportional = IndexEvenGenerator.index
    # Order of indices because we cut off a part for validation
    # dataset      0  1  2  0  1  2
    raw_indices = [2, 1, 1, 3, 1, 2]
    gt_indices  = [1, 1, 0, 1, 1, 0]
    # indices[0] because we only have one dataset
    assert len(dataset) == 9
    assert len([idx for sub in dataset.train_indices for idx in sub]) == 5
    assert len([idx for sub in dataset.val_indices for idx in sub]) == 4
    # We get back a batch of size 4 with all images in order
    sample = next(iter(dataset))
    for i, raw in enumerate(sample['raw']):
        mask = sample['mask'][i].astype(np.bool)
        assert mask.all()
        gt = sample['gt'][i]
        # We are looping through the datasets evenly
        dataset_index = i % 3
        raw_dataset = raw_datasets[dataset_index]
        gt_dataset = gt_datasets[dataset_index]
        raw_index = raw_indices[i]
        gt_index = gt_indices[i]
        assert np.array_equal(raw, raw_dataset[raw_index])
        assert np.array_equal(gt, gt_dataset[gt_index])
    # datasets         0  1  2  0
    val_raw_indices = [0, 0, 0, 1]
    val_gt_indices  = [0, 0, 0, 0]
    for i, val in enumerate(dataset.validation_samples()):
        mask = val['mask'][i].astype(np.bool)
        assert mask.all()
        gt = val['gt'][i]
        # We are looping through the datasets evenly
        dataset_index = i % 3
        raw_dataset = raw_datasets[dataset_index]
        raw_index = val_raw_indices[i]
        raw = raw_dataset[raw_index]
        gt_dataset = gt_datasets[dataset_index]
        gt_index = val_gt_indices[i]
        gt = gt_dataset[gt_index]
        assert np.array_equal(raw, raw_dataset[raw_index])
        assert np.array_equal(gt, gt_dataset[gt_index])


def test_multi_dataset_raw_gt_with_val_even_in_memory():
    abstract_test_multi_dataset_raw_gt_with_val_even(True)

def test_multi_dataset_raw_gt_with_val_even_on_demand():
    abstract_test_multi_dataset_raw_gt_with_val_even(False)

# Test for selection probability

def abstract_test_multi_dataset_raw_only_proportional_probability(in_memory):
    # Dataset 1 has 4 raw images and 2 gt images
    dataset_1_raw = conftest.dataset_1_raw_images()
    dataset_2_raw = conftest.dataset_2_raw_images()
    datasets = [dataset_1_raw, dataset_2_raw]
    dataset = TrainingDataset([conftest.dataset_1_raw_dir.name,
                               conftest.dataset_2_raw_dir.name],
                               # Batch size 12 to ensure that we draw enough samples
                               batch_size=24, val_ratio=0,
                               add_normalization_transform=False,
                               keep_in_memory=in_memory)
    TrainingDataset._dataset_index_proportional = _dataset_index_proportional
    dataset_1_count = 0
    dataset_2_count = 0
    sample = next(iter(dataset))
    # Compare GT against raw because we didn't specify GT thus have N2V training
    for gt in sample['gt']:
        for raw_1 in dataset_1_raw:
            if np.array_equal(gt, raw_1):
                dataset_1_count += 1
        for raw_2 in dataset_2_raw:
            if np.array_equal(gt, raw_2):
                dataset_2_count += 1
    assert dataset_1_count > dataset_2_count

def test_multi_dataset_raw_only_proportional_probability_in_memory():
    abstract_test_multi_dataset_raw_only_proportional_probability(True)

    def test_multi_dataset_raw_only_proportional_probability_on_demand():
        abstract_test_multi_dataset_raw_only_proportional_probability(False)
