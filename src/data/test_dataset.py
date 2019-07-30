import os
import glob
import numpy as np
import natsort
from torch.utils.data import Dataset

class TrainingDataset(Dataset):

    def __init__(self, raw_images_dir, gt_images_dir=None,
                 transform=None):
        assert os.path.exists(raw_images_dir)
        assert os.path.isdir(raw_images_dir)

        self._raw_images_dir = raw_images_dir
        raw_images = glob.glob(os.path.join(raw_images_dir, "*.npy"))
        # Sort the unsorted files
        raw_images = natsort.natsorted(raw_images)
        self._raw_images = np.array(raw_images)

        # If there are no ground-truth images we learn the network
        # Noise2Void style, otherwise we train it Noise2Clean
        if gt_images_dir is not None:
            assert os.path.isdir(gt_images_dir)
            self._test_mode = 'clean'
            self._gt_images_dir = gt_images_dir
            gt_images = glob.glob(os.path.join(gt_images_dir, "*.npy"))
            # Sort the unsorted files
            gt_images = natsort.natsorted(gt_images)
            self._gt_images = np.array(gt_images)
            # Same number of raw and ground-truth images
            assert len(self._raw_images) == len(self._gt_images)
        else:
            # If we want to train N2V style
            self._test_mode = 'void'

        self._transform = transform

    def __len__(self):
        return len(self._raw_images)

    def __getitem__(self, idx):
        raw_image = np.load(os.path.join(self._raw_images_dir,
                                         self._raw_images[idx]))
        gt_image = np.load(os.path.join(self._gt_images_dir,
                                        self._gt_images[idx]))
        sample = {'raw' : raw_image, 'gt' : gt_image}
        if self._transform is not None:
            sample = self._transform(sample)

        return sample
