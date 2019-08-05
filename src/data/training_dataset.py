import os
import glob
import numpy as np
import tifffile as tif
import natsort
from torch.utils.data import Dataset
from torchvision.transforms import Compose, functional

import util
from data.transforms import ToTensor, Normalize, ConvertToFormat
import constants

class TrainingDataset(Dataset):

    def __init__(self, raw_images_dir, gt_images_dir=None, batch_size=24,
                 val_ratio=0.1, transforms=None, convert_to_format=None,
                 add_normalization_transform=True, keep_in_memory=True,
                 num_pixels=32.0, seed=constants.NP_RANDOM_SEED):
        assert 0.0 <= val_ratio <= 1.0
        if transforms is not None:
            assert isinstance(transforms, list),\
                    'Expected list of transforms but got {} instead.'\
                        .format(type(transforms))

        self._keep_in_memory = keep_in_memory
        self.num_pixels = num_pixels
        self.batch_size = batch_size
        # This is important as we want to keep all images in memory and some
        # don't need float32 precision because they were uint8 originally
        self.convert_to_format = convert_to_format
        # If set to train, transforms are applied to data when loading, if set
        # to 'eval' not
        self._mode = 'train'

        self._load_raw_images(raw_images_dir)
        self._train_mode = self._load_gt_images(gt_images_dir)
        # Not sure if we keep in memory thus use the number of paths
        self._factor = len(self.raw_image_paths) / len(self.gt_image_paths)
        assert self._factor.is_integer(), 'Number of raw images needs to be ' +\
                                          'divisible by the number of ground-' +\
                                          'truth images.'

        if self._train_mode == 'void':
            print('Using {} raw images and no gt images for training.'
                   .format(len(self.raw_image_paths)))
        else:
            print('Using {} raw and {} gt images for training.'
                  .format(len(self.raw_image_paths), len(self.gt_image_paths)))

        self.mean, self.std = self._compute_mean_and_std()
        print('Dataset has mean {} and standard deviation {}.'\
                                .format(self.mean, self.std))

        self._init_transform(transforms, add_normalization_transform,
                             convert_to_format)

        # One training example that is the same for all experiments
        np.random.seed(seed)
        example_index = np.random.randint(len(self))
        self._training_example = {'raw' : self.raw_image_paths[example_index],
                                  'gt'  : self.gt_image_paths[example_index]}

        # Create training and validation indices
        self._val_ratio = val_ratio
        dataset_size = len(self.raw_image_paths)
        indices = np.array(range(dataset_size))
        split = int(val_ratio * dataset_size)
        # Seed numpy so that train and validation set are always the same for
        # all experiments
        indices = util.shuffle(indices, seed)
        self.train_indices, self.val_indices = indices[split:], indices[:split]

        self._init_indices()

    def _load_raw_images(self, raw_images_dir):
        assert os.path.exists(raw_images_dir)
        assert os.path.isdir(raw_images_dir)
        print('Adding raw images from: {}.'.format(raw_images_dir))
        self.raw_images_dir = raw_images_dir
        raw_image_paths = glob.glob(os.path.join(raw_images_dir, "*.tif"))
        # Sort the unsorted files
        raw_image_paths = natsort.natsorted(raw_image_paths)
        # We need the full paths because we can't find out the base dir
        # later during image loading
        self.raw_image_paths = np.array(raw_image_paths)
        if self._keep_in_memory:
            raw_images = [tif.imread(raw_image_path) for raw_image_path
                          in raw_image_paths]
            self.raw_images = np.array(raw_images)
            if self.convert_to_format is not None:
                # We convert the data here as this saves memory if the data
                # does not need to be stored in float32
                self.raw_images = self.raw_images.astype(self.convert_to_format)

    def _load_gt_images(self, gt_images_dir):
        # If there are no ground-truth images we learn the network
        # Noise2Void style, otherwise we train it Noise2Clean
        train_mode = ""
        # We also check if the user set raw and ground-truth dir to
        # the same path to achieve N2V
        if gt_images_dir is not None and gt_images_dir != self.raw_images_dir:
            assert os.path.isdir(gt_images_dir)
            print('Adding ground-truth images from: {}.'.format(gt_images_dir))
            self.gt_images_dir = gt_images_dir
            gt_image_paths = glob.glob(os.path.join(gt_images_dir, "*.tif"))
            # Sort the unsorted files
            gt_image_paths = natsort.natsorted(gt_image_paths)
            self.gt_image_paths = np.array(gt_image_paths)
            if self._keep_in_memory:
                gt_images = [tif.imread(gt_image_path) for gt_image_path
                          in gt_image_paths]
                self.gt_images = np.array(gt_images)
                if self.convert_to_format is not None:
                    # We convert the data here as this saves memory if the data
                    # does not need to be stored in float32
                    self.raw_images = self.raw_images.astype(
                                                        self.convert_to_format)
            train_mode = 'clean'
        else:
            # If we want to train N2V style
            self.gt_images_dir = self.raw_images_dir
            self.gt_image_paths = self.raw_image_paths
            if self._keep_in_memory:
                # Need to copy because pixels in raw data get replaced
                # in Noise2Void
                self.gt_images = self.raw_images.copy()
            train_mode = 'void'
        print('Performing Noise2{} training.'.format(train_mode.capitalize())) 
        return train_mode

    def _init_transform(self, requested_transforms, add_normalization_transform,
                        data_format):
        transforms = requested_transforms
        # We need a different set of transforms for evaluation as we do not
        # want to randomly crop, rotate and flip images
        eval_transforms = []

        # We have to go this special way for conversion because if the user
        # specifies to keep the data in memory then the data is converted to
        # the desired format during loading to save memory (the sole purpose of
        # the conversion)
        if data_format is not None:
            transforms = self._add_convert_to_format_transform(requested_transforms,
                                                                    data_format)
            eval_transforms.append(ConvertToFormat(data_format))

        # From now on we can be sure that self.transform is a Compose or None
        # If requested we append a normalization transform, this allows us to
        # use the freshly computed mean and std
        if add_normalization_transform:
            # We need to ensure that the Normalize operation is before ToTensor
            if transforms and isinstance(transforms[-1], ToTensor):
                transforms.insert(-1, Normalize(self.mean, self.std))
            else:
                if not transforms:
                    transforms = []
                transforms.append(Normalize(self.mean, self.std))
            eval_transforms.append(Normalize(self.mean, self.std))

        if transforms is not None:
            self.transforms = Compose(transforms)
        else:
            self.transforms = None
        if eval_transforms:
            # No need to check if self.transforms is None as it gets only set
            # if self.transforms is not None
            if isinstance(self.transforms.transforms[-1], ToTensor):
                # If we have a ToTensor transform for train mode we need the
                # same for eval mode
                eval_transforms.append(ToTensor())
            self.eval_transforms = Compose(eval_transforms)
        else:
            self.eval_transforms = None

    def _add_convert_to_format_transform(self, transforms, data_format):
        convert_to_format = ConvertToFormat(data_format)
        if transforms is None:
            transforms = [convert_to_format]
        elif isinstance(transforms[-1], ToTensor):
            # Last transform is ToTensor but Normalize expects numpy arrays so
            # we have to add it before the ToTensor transform
            transforms.insert(-1, convert_to_format)
        else:
            # Last one isn't ToTensor i.e. we can just add the transform
            transforms.append(convert_to_format)
        return transforms

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
        if self._keep_in_memory:
            return np.mean(self.raw_images), np.std(self.raw_images)
        else:
            images = []
            for raw_image_path in self.raw_image_paths:
                image = tif.imread(raw_image_path)
                images.append(image)

            mean, std = np.mean(images), np.std(images)
            # Make sure the images get deleted right away
            del images
            return mean, std

    def _init_indices(self):
        self.current_train_indices = self.train_indices.copy()
        self.current_val_indices = self.val_indices.copy()

    def __len__(self):
        # Use paths because in non-memory mode raw_images are not loaded
        return len(self.raw_image_paths)

    def train(self):
        self._mode = 'train'

    def eval(self):
        self._mode = 'eval'

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

    @staticmethod
    def _noise_to_void_preparation(num_pixels, transformed_raw_image, 
                                   transformed_shape, image_mode):
        # Noise2Void style training, we have to replace the masked pixel
        # with one of its neighbors so that the network can't learn the
        # identity function
        mask = np.zeros(transformed_shape)
        max_x = transformed_shape[1] - 1
        max_y = transformed_shape[0] - 1
        # The size of the box around each hot pixel which we don't take
        # another hot pixel from
        box_size = np.round(np.sqrt(num_pixels)).astype(np.int)
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
        # No need to return the image itself as we get it by reference and
        # it thus gets updated directly
        return mask

    def __getitem__(self, idx):
        if self._keep_in_memory:
            raw_image = self.raw_images[idx]
            gt_image = self.gt_images[int(idx / self._factor)]
        else:
            raw_image = tif.imread(self.raw_image_paths[idx])
            gt_image = tif.imread(self.gt_image_paths[int(idx / self._factor)])

        sample = {'raw' : raw_image, 'gt' : gt_image}
        if self.transforms is not None and self._mode == 'train':
            sample = self.transforms(sample)
        elif self._mode == 'eval':
            sample = self.eval_transforms(sample)

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
        elif len(transformed_raw_image) == (1 or 3):
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
            mask = TrainingDataset._noise_to_void_preparation(self.num_pixels,
                                                              transformed_raw_image,
                                                              transformed_shape,
                                                              image_mode)
        else:
            # Noise2Clean style training, we have ground-truth data available
            # and thus mask all pixels to be active
            mask = np.ones(transformed_shape)

        # Need to convert to float32 as all other tensors are in float32
        mask = mask.astype(np.float32)
        if self.transforms is not None\
                and isinstance(self.transforms.transforms[-1], ToTensor):
            # In case that we convert the raw and gt image to a torch tensor
            # then we have to convert the mask as well because the user wants
            # it to be performed automatically
            mask.shape = (mask.shape[0], mask.shape[1], 1)
            mask = functional.to_tensor(mask)
        sample['mask'] = mask
        return sample

    def __iter__(self):
        return self
    
    def __next__(self):
        pass
        
    def training_example(self):
        raw_image = tif.imread(self._training_example['raw'])
        gt_image = tif.imread(self._training_example['gt'])
        sample = {'raw' : raw_image,
                  'gt'  : gt_image}
        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample