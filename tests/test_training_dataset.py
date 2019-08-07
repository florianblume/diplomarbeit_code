import numpy as np
import pytest
import torch

from tests import base_test
from tests import conftest

import util
from data import TrainingDataset
from data.transforms import RandomCrop, ToTensor

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

# Classes for multi dataset dataset-index generation

class IndexEvenGenerator():

    counter = 0

    @staticmethod
    def _dataset_index_even_multi_dataset(num_datasets):
        index = IndexEvenGenerator.counter
        IndexEvenGenerator.counter += 1
        IndexEvenGenerator.counter %= num_datasets
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
    assert len(dataset) == 2
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
        # Second half are training indices in TrainingDataset that's why we do
        # i + 2
        assert np.array_equal(raw[mask], dataset_1_raw[i + 2][mask])
        assert not np.array_equal(raw, dataset_1_raw[i + 2])
        assert np.array_equal(sample['gt'], dataset_1_raw[i + 2])
    
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
    assert len(dataset) == 2
    # indices[0] because we only have one dataset
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
    # indices[0] because we only have one dataset
    assert len(dataset) == 4
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
    transforms = [RandomCrop(20, 20), ToTensor()]
    dataset = TrainingDataset([conftest.dataset_1_raw_dir.name], batch_size=4,
                              val_ratio=0, add_normalization_transform=True,
                              transforms=transforms,
                              keep_in_memory=in_memory)
    assert len(dataset) == 4
    # indices[0] because we only have one dataset
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
    assert len(dataset) == 2
    # indices[0] because we only have one dataset
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
    assert len(dataset) == 2
    # indices[0] because we only have one dataset
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
    # Dataset 1 has 4 raw images and 2 gt images
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
    # indices[0] because we only have one dataset
    assert len(dataset) == 9
    assert len([idx for sub in dataset.val_indices for idx in sub]) == 0
    # We get back a batch of size 4 with all images in order
    sample = next(iter(dataset))
    for i, raw in enumerate(sample['raw']):
        mask = ~sample['mask'][i].astype(np.bool)
        assert not mask.all()
        gt = sample['gt'][i]
        if i < 4:
            dataset_index = 0
            image_index = i
        elif i < 6:
            dataset_index = 1
            image_index = i - 4
        else:
            dataset_index = 2
            image_index = i - 6
        # Second half are training indices in TrainingDataset
        test_dataset = datasets[dataset_index]
        assert np.array_equal(raw[mask], test_dataset[image_index][mask])
        assert not np.array_equal(raw, test_dataset[image_index])
        assert np.array_equal(gt, test_dataset[image_index])

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
    # indices[0] because we only have one dataset
    assert len(dataset) == 5
    assert len([idx for sub in dataset.val_indices for idx in sub]) == 4
    # We get back a batch of size 4 with all images in order
    sample = next(iter(dataset))
    for i, raw in enumerate(sample['raw']):
        mask = ~sample['mask'][i].astype(np.bool)
        # Since N2V training
        assert not mask.all()
        gt = sample['gt'][i]
        if i < 2:
            dataset_index = 0
            # Since validation indices come first, the indices of the two raw
            # images in our dataset are 2 and 3
            image_index = [2, 3][i]
        elif i < 3:
            dataset_index = 1
            # Only one image in the second dataset because the other is used
            # for validaiton
            image_index = 1
        else:
            dataset_index = 2
            # Same as for dataset 1
            image_index = [1, 2][i - 3]
        # Second half are training indices in TrainingDataset
        test_dataset = datasets[dataset_index]
        assert np.array_equal(raw[mask], test_dataset[image_index][mask])
        assert not np.array_equal(raw, test_dataset[image_index])
        assert np.array_equal(gt, test_dataset[image_index])
    for val in dataset.validation_samples():
        for i, raw in enumerate(val['raw']):
            if i < 2:
                dataset_index = 0
                img_index = i
            elif i < 3:
                dataset_index = 1
                img_index = 0
            elif i < 4:
                dataset_index = 2
                img_index = 0
            val_raw = raw.squeeze()
            val_gt = val['gt'][i].squeeze()
            mask = ~val['mask'][i].squeeze().astype(np.bool)
            test_dataset = datasets[dataset_index]
            assert np.array_equal(val_raw[mask], test_dataset[img_index][mask])
            assert np.array_equal(val_gt, test_dataset[img_index])

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
    assert len([idx for sub in dataset.val_indices for idx in sub]) == 0
    # We get back a batch of size 4 with all images in order
    sample = next(iter(dataset))
    for i, raw in enumerate(sample['raw']):
        mask = sample['mask'][i].astype(np.bool)
        assert mask.all()
        gt = sample['gt'][i]
        if i < 4:
            dataset_index = 0
            raw_index = i
            # Dataset 1 has factor 1 - 2 raw to gt
            gt_index = [0, 1][int(i / 2)]
        elif i < 6:
            dataset_index = 1
            raw_index = i - 4
            gt_index = i - 4
        else:
            dataset_index = 2
            raw_index = i - 6
            gt_index = 0
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
    print(dataset.train_indices)
    print(dataset.val_indices)
    # indices[0] because we only have one dataset
    assert len(dataset) == 5
    assert len([idx for sub in dataset.val_indices for idx in sub]) == 4
    # We get back a batch of size 4 with all images in order
    sample = next(iter(dataset))
    for i, raw in enumerate(sample['raw']):
        mask = sample['mask'][i].astype(np.bool)
        assert mask.all()
        gt = sample['gt'][i]
        if i < 2:
            dataset_index = 0
            # Since validation indices come first, the indices of the two raw
            # images in our dataset are 2 and 3
            raw_index = [2, 3][i]
            gt_index = 1
        elif i < 3:
            dataset_index = 1
            # Only one image in the second dataset because the other is used
            # for validaiton
            raw_index = 1
            gt_index = 1
        else:
            dataset_index = 2
            # Same as for dataset 1
            raw_index = [1, 2][i - 3]
            gt_index = 0
        # Second half are training indices in TrainingDataset
        raw_dataset = raw_datasets[dataset_index]
        gt_dataset = gt_datasets[dataset_index]
        assert np.array_equal(raw, raw_dataset[raw_index])
        assert np.array_equal(gt, gt_dataset[gt_index])
    for val in dataset.validation_samples():
        for i, raw in enumerate(val['raw']):
            mask = val['mask'][i].squeeze().astype(np.bool)
            assert mask.all()
            gt_index = 0
            if i < 2:
                dataset_index = 0
                raw_index = i
            elif i < 3:
                dataset_index = 1
                raw_index = 0
            elif i < 4:
                dataset_index = 2
                raw_index = 0
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
