import os
import glob
import numpy as np
from torch.utils.data import Dataset

class TrainingDataset(Dataset):

    def __init__(self, raw_images_dir, gt_images_dir=None, 
                 val_ratio=0.1, transform=None):
        assert os.path.exists(raw_images_dir)
        assert os.path.isdir(raw_images_dir)

        self.raw_images_dir = raw_images_dir
        self.raw_images = glob.glob(os.path.join(raw_images_dir, "*.npy"))

        # If there are no ground-truth images we learn the network
        # Noise2Void style, otherwise we train it Noise2Clean
        if gt_images_dir is not None:
            assert os.path.isdir(gt_images_dir)
            self.gt_images_dir = gt_images_dir
            self.gt_images = glob.glob(os.path.join(gt_images_dir, "*.npy"))
            # Same number of raw and ground-truth images
            assert len(self.raw_images) == len(self.gt_images)
            self.train_mode = 'clean'
        else:
            # If we want to train N2V style 
            self.gt_images_dir = raw_images_dir
            self.gt_images = glob.glob(os.path.join(gt_images_dir, "*.npy"))
            self.train_mode = 'void'

        self._compute_mean_and_std()

        self.val_ratio = val_ratio
        self.transform = transform

    def _compute_mean_and_std(self):
        pass

    def __len__(self):
        return len(self.raw_images)

    def __getitem__(self, idx):
        raw_image = np.load(os.path.join(self.raw_images_dir,
                                         self.raw_images[idx]))
        gt_image = np.load(os.path.join(self.gt_images_dir,
                                        self.gt_images[idx]))
        mask = np.ones(raw_image.shape)

        sample = {}

        if self.transform:
            sample = self.transform(sample)

        if self.train_mode == 'void':
            # N2V - we need to set the hot pixels
            c = 1

        return sample
        