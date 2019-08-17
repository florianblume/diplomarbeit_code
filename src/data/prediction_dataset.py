import os
import glob
import numpy as np
import tifffile as tif
import natsort
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from data.transforms import ToTensor

class PredictionDataset(Dataset):

    def __init__(self, raw_images_dirs: list, gt_images_dirs=None,
                 transform=None):
        self._factors = []
        self._indices = []
        self._raw_images_dirs = raw_images_dirs
        self._raw_images = []
        with_ground_truth = gt_images_dirs is not None
        if with_ground_truth:
            self._gt_images = []
            self._gt_images_dirs = gt_images_dirs

        self._test_mode = None
        for i, raw_images_dir in enumerate(raw_images_dirs):
            assert os.path.exists(raw_images_dir)
            assert os.path.isdir(raw_images_dir)
            raw_images = glob.glob(os.path.join(raw_images_dir, "*.tif"))
            # Sort the unsorted files
            raw_images = natsort.natsorted(raw_images)
            self._raw_images.append(raw_images)
            # For faster retrieval of the corresponding dataset
            indices = [(i, im_index) for im_index in range(len(raw_images))]
            self._indices.extend(indices)

            # If there are no ground-truth images we learn the network
            # Noise2Void style, otherwise we train it Noise2Clean
            if with_ground_truth:
                if self._test_mode is not None and self._test_mode == 'void':
                    raise ValueError('Cannot predict clean and void style '+
                                     ' at the same time.')
                gt_images_dir = gt_images_dirs[i]
                assert os.path.exists(gt_images_dir)
                assert os.path.isdir(gt_images_dir)
                gt_images = glob.glob(os.path.join(gt_images_dir, "*.tif"))
                # Sort the unsorted files
                gt_images = natsort.natsorted(gt_images)
                self._gt_images.append(gt_images)
                # Same number of raw and ground-truth images
                factor = len(raw_images) / len(gt_images)
                assert factor.is_integer(), 'Number of raw images needs to be '+\
                                            'divisible by the number of ground-'+\
                                            'truth images.'
                self._factors.append(factor)
                self._test_mode = 'clean'
            else:
                if self._test_mode is not None and self._test_mode == 'clean':
                    raise ValueError('Cannot predict clean and void style '+
                                     ' at the same time.')
                # If we want to train N2V style
                self._test_mode = 'void'

        self._raw_images = np.array(self._raw_images)
        if gt_images_dirs is not None:
            self._gt_images = np.array(self._gt_images)
        if transform is not None:
            self._transform = Compose(transform)
        else:
            self._transform = None

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, idx):
        dataset, im_index = self._indices[idx][0], self._indices[idx][1]
        raw_image = tif.imread(self._raw_images[dataset][im_index])
        if self._test_mode == 'clean':
            gt_image = tif.imread(self._gt_images[dataset]\
                                       [int(im_index / self._factors[dataset])])
            sample = {'raw' : raw_image, 'gt' : gt_image}
        else:
            sample = {'raw' : raw_image}

        if self._transform is not None:
            sample = self._transform(sample)
            if isinstance(self._transform.transforms[-1], ToTensor):
                # Assemble "batch", maybe we have actual batching in the future
                sample['raw'] = torch.stack([sample['raw']]).float()
                sample['gt'] = torch.stack([sample['gt']])

        return sample
