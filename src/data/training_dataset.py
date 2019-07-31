import os
import glob
import numpy as np
import tifffile as tif
import natsort
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision.transforms import Compose
import time

import util
from data.transforms import ToTensor, Normalize
import constants

class TrainingDataset(Dataset):

    def __init__(self, raw_images_dirs: list, gt_images_dirs=None,
                 val_ratio=0.1, transforms=None, 
                 add_normalization_transform=True,
                 num_pixels=32.0, seed=constants.NP_RANDOM_SEED):
        assert 0 <= val_ratio <= 1
        if gt_images_dirs is not None:
            assert len(raw_images_dirs) == len(gt_images_dirs)
        if transforms is not None:
            assert isinstance(transforms, list),\
                    'Expected list of transforms but got {} instead.'\
                        .format(type(transforms))

        self._raw_images_dirs = raw_images_dirs
        print('Adding raw images from: {}.'.format(raw_images_dirs))
        self._raw_images = []
        if gt_images_dirs is not None:
            self._gt_images = []
            self._gt_images_dirs = gt_images_dirs
            self._train_mode = 'clean'
            print('Adding gt images from: {}.'.format(gt_images_dirs))
        else:
            self._train_mode = 'void'


        for i, raw_images_dir in enumerate(raw_images_dirs):
            assert os.path.exists(raw_images_dir)
            assert os.path.isdir(raw_images_dir)

            raw_images = glob.glob(os.path.join(raw_images_dir, "*.tif"))
            # Sort the unsorted files
            raw_images = natsort.natsorted(raw_images)
            # We need the full paths because we can't find out the base dir
            # later during image loading
            self._raw_images.extend(raw_images)

            # If there are no ground-truth images we learn the network
            # Noise2Void style, otherwise we train it Noise2Clean
            if gt_images_dirs is not None:
                gt_images_dir = gt_images_dirs[i]
                assert os.path.isdir(gt_images_dir)
                gt_images = glob.glob(os.path.join(gt_images_dir, "*.tif"))
                # Sort the unsorted files
                gt_images = natsort.natsorted(gt_images)
                self._gt_images.extend(gt_images)
                # Same number of raw and ground-truth images
                assert len(self._raw_images) == len(self._gt_images)
            else:
                # If we want to train N2V style
                self._gt_images_dirs = raw_images_dirs
                self._gt_images = self._raw_images

        if self._train_mode == 'void':
            print('Using {} raw images and no gt images for training.'
                        .format(len(self._raw_images)))
        else:
            print('Using {} raw and {} gt images for training.'
                        .format(len(self._raw_images), len(self._gt_images)))
        self._raw_images = np.array(self._raw_images)
        self._gt_images = np.array(self._gt_images)

        self._mean, self._std = self._compute_mean_and_std()
        print('Dataset has mean {} and standard deviation {}.'\
                                .format(self._mean, self._std))

        # If requested we append a normalization transform, this allows us to
        # use the freshly computed mean and std
        if add_normalization_transform:
            self._add_normalize_transform(transforms)
        elif transforms is not None:
            self._transform = Compose(transforms)
        else:
            self._transform = None

        self._num_pixels = num_pixels

        # Seeding is done in the util function
        self._raw_images, self._gt_images = util.joint_shuffle(self._raw_images,
                                                               self._gt_images,
                                                               seed)
        # One training example that is the same for all experiments
        np.random.seed(seed)
        example_index = np.random.randint(len(self))
        self._training_example = {'raw' : self._raw_images[example_index],
                                  'gt'  : self._gt_images[example_index]}

        # Create training and validation indices
        self._val_ratio = val_ratio
        dataset_size = self._raw_images.shape[0]
        indices = list(range(dataset_size))
        split = int(val_ratio * dataset_size)
        np.random.seed(seed)
        np.random.shuffle(indices)
        self._train_indices, self._val_indices = indices[split:], indices[:split]

    def _compute_mean_and_std(self):
        """
        This works approximately and can be used for large datasets that never
        fit in memory.

        means = []
        std = 0
        for raw_image in self._raw_images:
            image = tif.imread(os.path.join(self._raw_images_dir, raw_image))
            means.append(np.mean(image))
        mean = np.mean(means)
        for raw_image in self._raw_images:
            image = tif.imread(os.path.join(self._raw_images_dir, raw_image))
            tmp = np.sum((image - mean)**2)
            std += tmp / float(self._raw_images.shape[0] * image.shape[0] * image.shape[1] - 1)
        """
        images = []
        for raw_image in self._raw_images:
            image = tif.imread(raw_image)
            images.append(image)

        return np.mean(images), np.std(images)

    def _add_normalize_transform(self, transforms):
        normalize = Normalize(self._mean, self._std)
        if transforms is None:
            transforms = [normalize]
        elif isinstance(transforms[-1], ToTensor):
            # Last transform is ToTensor but Normalize expects numpy arrays so
            # we have to add it before the ToTensor transform
            transforms.insert(-1, normalize)
        else:
            # Last one isn't ToTensor i.e. we can just add the transform
            transforms.append(normalize)
        self._transform = Compose(transforms)

    def set_transform(self, transform):
        self._transform = transform

    def mean(self):
        return self._mean

    def std(self):
        return self._std

    def train_indices(self):
        return self._train_indices

    def val_indices(self):
        return self._val_indices

    def __len__(self):
        return len(self._raw_images)

    @staticmethod
    def get_stratified_coords_2D(box_size, shape):
        """This function computes the indices of the hot pixels within the
        specified shape. The box size is used as a box around each hot pixel
        where no other hot pixel can reside in.
        
        Arguments:
            box_size {int} -- box size around each hot pixel where there can be
                              no other hot pixel in
            shape {tuple} -- the shape of the area to compute the hot pixels for
        
        Returns:
            np.array -- the indices of the hot pixels
        """
        coords = []
        # Number of hot pixels in x and y direction
        box_count_y = int(np.ceil(shape[0] / box_size))
        box_count_x = int(np.ceil(shape[1] / box_size))
        for i in range(box_count_y):
            for j in range(box_count_x):
                # Construct random hot pixel within box
                y = np.random.randint(0, box_size)
                x = np.random.randint(0, box_size)
                y = int(i * box_size + y)
                x = int(j * box_size + x)
                if (y < shape[0] and x < shape[1]):
                    coords.append((y, x))
        return coords

    def __getitem__(self, idx):
        raw_image = tif.imread(self._raw_images[idx])
        gt_image = tif.imread(self._gt_images[idx])
        sample = {'raw' : raw_image, 'gt' : gt_image}
        if self._transform is not None:
            sample = self._transform(sample)

        # Retrieve the transformed image
        transformed_raw_image = sample['raw']
        image_shape_len = len(transformed_raw_image.shape)
        if image_shape_len < 2 or image_shape_len > 3:
            # Only gray-scale or RGB images are supported as of now
            raise ValueError('Unkown image shape.')

        if image_shape_len == 2:
            # Image has not been transformed into tensor already, thus shape is
            # still [H, W]
            transformed_shape = transformed_raw_image.shape
            image_mode = 'hw'
        elif transformed_raw_image.shape[0] == (1 or 3):
            # Image has been transformed into tensor already, thus shape is
            # now [C, H, W]
            transformed_shape = transformed_raw_image.shape[1:]
            image_mode = 'chw'
        elif transformed_raw_image.shape[2] == (1 or 3):
            # Image has not been transformed into tensor already, thus shape is
            # still [H, W, C]
            transformed_shape = transformed_raw_image.shape[:-1]
            image_mode = 'hwc'
        else:
            raise ValueError('Unkown shape format. Neither [C, H, W] nor [H, W, C].')

        if self._train_mode == 'void':
            # Noise2Void style training, we have to replace the masked pixel
            # with one of its neighbors so that the network can't learn the
            # identity function
            mask = np.zeros(transformed_shape)
            max_x = transformed_shape[1] - 1
            max_y = transformed_shape[0] - 1
            # The size of the box around each hot pixel which we don't take
            # another hot pixel from
            box_size = np.round(np.sqrt(self._num_pixels)).astype(np.int)
            hot_pixels = TrainingDataset.get_stratified_coords_2D(box_size,
                                                                  transformed_shape)
            for hot_pixel in hot_pixels:
                # We want to replace the hot pixel with one of its neighbors
                x, y = hot_pixel[1], hot_pixel[0]

                # Thus we define a ROI around it with which to replace the
                # hot pixel
                roi_min_x = max(x - 2, 0)
                roi_max_x = min(x + 3, max_x)
                roi_min_y = max(y - 2, 0)
                roi_max_y = min(y + 3, max_y)

                roi_width = roi_max_x - roi_min_x
                roi_height = roi_max_y - roi_min_y
                                                
                # Construct set of indices in the ROI excluding the hot pixel
                # roi.shape[0] are the channels
                x_indices, y_indices = np.mgrid[0:roi_width,
                                                0:roi_height]
                indices = np.vstack((x_indices.flatten(), y_indices.flatten())).T
                indices_list = indices.tolist()
                # [2, 2] are the hot pixel's coordinates, we don't want to
                # sample it thus remove them from the list (for very small
                # ROIs [2, 2] is not in the list, thus the check)
                if [2, 2] in indices_list:
                    indices_list.remove([2, 2])
                # Obtain random pixel to replace hot pixel with
                x_, y_ = indices_list[np.random.randint(len(indices_list))]

                if image_mode == 'hw':
                    # Cut out the ROI from the original image
                    roi = transformed_raw_image[roi_min_y:roi_max_y,
                                                roi_min_x:roi_max_x]
                    # Obtain neighbor to replace hot pixel with
                    repl = roi[y_, x_]
                    # Replace content of hot pixel
                    transformed_raw_image[y, x] = repl
                elif image_mode == 'chw':
                    roi = transformed_raw_image[:,
                                                roi_min_y:roi_max_y,
                                                roi_min_x:roi_max_x]
                    repl = roi[:, y_, x_]
                    transformed_raw_image[:, y, x] = repl
                elif image_mode == 'hwc':
                    roi = transformed_raw_image[roi_min_y:roi_max_y,
                                                roi_min_x:roi_max_x,
                                                :]
                    repl = roi[y_, x_, :]
                    transformed_raw_image[y, x, :] = repl

                # Set hot pixel to active
                mask[y, x] = 1.0
                # No need to update the image in the dict as we hold a 
                # reference to it
        else:
            # Noise2Clean style training, we have ground-truth data available
            # and thus mask all pixels to be active
            mask = np.ones(transformed_shape)

        # Need to convert to float32 as all other tensors are in float32
        mask = mask.astype(np.float32)
        if self._transform is not None\
                and isinstance(self._transform.transforms[-1], ToTensor):
            # In case that we convert the raw and gt image to a torch tensor
            # then we have to convert the mask as well because the user wants
            # it to be performed automatically
            mask.shape = (mask.shape[0], mask.shape[1], 1)
            mask = torchvision.transforms.functional.to_tensor(mask)
        sample['mask'] = mask
        return sample
        
    def const_training_example(self):
        raw_image = tif.imread(self._training_example['raw'])
        gt_image = tif.imread(self._training_example['gt'])
        return {'raw' : raw_image,
                'gt'  : gt_image}