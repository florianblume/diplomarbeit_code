import os
import glob
import copy
import numpy as np
import tifffile as tif
import natsort
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, functional

import util
from data.transforms import ToTensor, Normalize, ConvertToFormat
import constants

class TrainingDataset(Dataset):

    def _init_attributes(self):
        # How samples are drawn from the the specified datasets (i.e. image dirs)
        # 'even' means even and 'proportional' according to dataset size
        self.distribution_modes = ['even', 'proportional']

        self._num_datasets = 0
        # list of list of loaded raw images, if keep in memory is true
        self.raw_images = []
        self.gt_images = []
        # list of folders with raw images
        self.raw_images_dirs = []
        # list of lists of full paths to raw images
        self.raw_image_paths = []
        # analogously
        self.gt_images_dirs = []
        self.gt_image_paths = []
        # individual raw to gt factors
        self._factors = []
        # list of lists of indices
        self.train_indices = []
        # currently remaining train indices, gets filled up when all have been
        # returned once
        self._current_train_indices = []
        self.val_indices = []
        # we store one exapmle per dataset
        self._training_examples = []

    # Random functions extenalized such that tests can inject random functions
    # with known outcome
    @staticmethod
    def _example_index(length):
        return np.random.randint(length)

    @staticmethod
    def _shuffle_raw_indices(indices, seed):
        return util.shuffle(indices, seed)

    @staticmethod
    def _stratified_coord_x(max):
        return np.random.randint(max)

    @staticmethod
    def _stratified_coord_y(max):
        return np.random.randint(max)

    @staticmethod
    def _hot_pixel_replacement_index(length):
        return np.random.randint(length)

    @staticmethod
    def _train_indices_permutation(indices):
        return np.random.permutation(indices).tolist()

    @staticmethod
    def _val_indices_permutation(indices):
        return np.random.permutation(indices).tolist()

    @staticmethod
    def _dataset_index_even(length):
        return np.random.randint(length)

    @staticmethod
    def _dataset_index_proportional(dataset_sizes):
        probabilities = dataset_sizes / np.sum(dataset_sizes)
        return np.random.choice(len(dataset_sizes), 1, p=probabilities)[0]

    def __init__(self, raw_images_dirs, gt_images_dirs=None, batch_size=24,
                 distribution_mode='proportional', val_ratio=0.1,
                 transforms=None, convert_to_format=None,
                 add_normalization_transform=True, keep_in_memory=True,
                 num_pixels=32.0, seed=constants.NP_RANDOM_SEED):
        self._init_attributes()
        assert isinstance(raw_images_dirs, list),\
                'Expected list of raw image dirs but got {} instead.'\
                    .format(type(transforms))
        if gt_images_dirs is not None:
            assert isinstance(gt_images_dirs, list),\
                    'Expected list of gt image dirs but got {} instead.'\
                        .format(type(transforms))
            self._train_mode = 'clean'
        else:
            self._train_mode = 'void'
        print('Performing Noise2{} training.'.format(self._train_mode.capitalize()))
        if transforms is not None:
            assert isinstance(transforms, list),\
                    'Expected list of transforms but got {} instead.'\
                        .format(type(transforms))
        assert batch_size > 0
        if distribution_mode not in self.distribution_modes:
            raise ValueError('Illegal distribution mode \"{}\". Possible \
                             choices are{}'.format(distribution_mode,
                                                   self.distribution_modes))
        self.distribution_mode = distribution_mode
        print('Using dataset distribution mode \"{}\".'.format(distribution_mode))
        assert 0.0 <= val_ratio <= 1.0
        assert num_pixels > 0

        self._keep_in_memory = keep_in_memory
        self.num_pixels = num_pixels
        self.batch_size = batch_size
        # This is important as we want to keep all images in memory and some
        # don't need float32 precision because they were uint8 originally
        self.convert_to_format = convert_to_format
        # If set to train, transforms are applied to data when loading, if set
        # to 'eval' not
        self.val_ratio = val_ratio
        self.raw_images_dirs = raw_images_dirs
        self._num_datasets = len(raw_images_dirs)
        self.gt_images_dirs = gt_images_dirs

        self._load_datasets(seed)

        flattend_raw_image_paths = [image_path for sublist in\
                                self.raw_image_paths for image_path in sublist]
        raws_size = len(flattend_raw_image_paths)
        if self._train_mode == 'void':
            print('Using {} raw images and no gt images for training.'
                   .format(raws_size))
        else:
            flattend_gt_image_paths = [image_path for sublist in\
                                self.gt_image_paths for image_path in sublist]
            print('Using {} raw and {} gt images for training.'
                  .format(raws_size,
                          len(flattend_gt_image_paths)))

        self.mean, self.std = self._compute_mean_and_std()
        print('Dataset has mean {} and standard deviation {}.'\
                                .format(self.mean, self.std))

        self._init_transform(transforms, add_normalization_transform,
                             convert_to_format)
        self.image_shape = self._get_sample(0, 0)['raw'].shape

    def _load_datasets(self, seed):
        for i, raw_images_dir in enumerate(self.raw_images_dirs):
            raw_image_paths, raw_images = self._load_raw_images(raw_images_dir)
            gt_images_dir = self.gt_images_dirs[i] if self.gt_images_dirs is\
                                not None else None
            gt_images_dir, gt_image_paths, gt_images =\
                    self._load_gt_images(gt_images_dir, raw_images_dir,
                                         raw_image_paths, raw_images)
            if gt_image_paths is not None:
                factor = len(raw_image_paths) / len(gt_image_paths)
                assert factor.is_integer(), 'Number of raw images needs to be '+\
                                            'divisible by the number of ground-'+\
                                            'truth images.'
            else:
                factor = 1.0
            self._factors.append(factor)

            self.raw_image_paths.append(raw_image_paths)
            if raw_images is not None:
                self.raw_images.append(raw_images)
            self.gt_image_paths.append(gt_image_paths)
            if gt_images is not None:
                self.gt_images.append(gt_images)

            # One training example that is the same for all experiments
            np.random.seed(seed)
            example_index = TrainingDataset._example_index(len(raw_image_paths))
            raw_example = raw_image_paths[example_index]
            gt_example = gt_image_paths[int(example_index / factor)]
            training_example = {'raw' : raw_example,
                                'gt'  : gt_example}
            self._training_examples.append(training_example)

            # Create training and validation indices
            dataset_size = len(raw_image_paths)
            indices = np.array(range(dataset_size))
            split = int(self.val_ratio * dataset_size)
            # Seed numpy so that train and validation set are always the same for
            # all experiments
            indices = TrainingDataset._shuffle_raw_indices(indices, seed)
            train_indices, val_indices = indices[split:], indices[:split]
            self.train_indices.append(train_indices.tolist())
            self.val_indices.append(val_indices.tolist())
            # We call tolist() here because the current indices get altered
            # a lot during sample retrieval which is faster for plain
            # Python lists
        self._current_train_indices = copy.deepcopy(self.train_indices)

    def _load_raw_images(self, raw_images_dir):
        assert os.path.exists(raw_images_dir)
        assert os.path.isdir(raw_images_dir)
        print('Adding raw images from: {}.'.format(raw_images_dir))
        raw_image_paths = glob.glob(os.path.join(raw_images_dir, "*.tif"))
        # Sort the unsorted files
        raw_image_paths = natsort.natsorted(raw_image_paths)
        # We need the full paths because we can't find out the base dir
        # later during image loading
        raw_image_paths = np.array(raw_image_paths)
        if self._keep_in_memory:
            raw_images = [tif.imread(raw_image_path) for raw_image_path
                          in raw_image_paths]
            raw_images = np.array(raw_images)
            if self.convert_to_format is not None:
                # We convert the data here as this saves memory if the data
                # does not need to be stored in float32
                raw_images = raw_images.astype(self.convert_to_format)
        else:
            raw_images = None
        return raw_image_paths, raw_images

    def _load_gt_images(self, gt_images_dir, corresponding_raw_images_dir,
                        corresponding_raw_image_paths, corresponding_raw_images):
        gt_images = None
        # We check if the user set raw and ground-truth dir to
        # the same path to achieve N2V
        if gt_images_dir is not None and gt_images_dir != corresponding_raw_images_dir:
            assert os.path.isdir(gt_images_dir)
            print('Adding ground-truth images from: {}.'.format(gt_images_dir))
            gt_image_paths = glob.glob(os.path.join(gt_images_dir, "*.tif"))
            # Sort the unsorted files
            gt_image_paths = natsort.natsorted(gt_image_paths)
            gt_image_paths = np.array(gt_image_paths)
            if self._keep_in_memory:
                gt_images = [tif.imread(gt_image_path) for gt_image_path
                          in gt_image_paths]
                gt_images = np.array(gt_images)
                if self.convert_to_format is not None:
                    # We convert the data here as this saves memory if the data
                    # does not need to be stored in float32
                    gt_images = gt_images.astype(self.convert_to_format)
        else:
            # If we want to train N2V style
            gt_images_dir = corresponding_raw_images_dir
            gt_image_paths = corresponding_raw_image_paths
            if self._keep_in_memory:
                # Need to copy because pixels in raw data get replaced
                # in Noise2Void
                gt_images = corresponding_raw_images.copy()
        return gt_images_dir, gt_image_paths, gt_images

    def _init_transform(self, requested_transforms, add_normalization_transform,
                        data_format):
        transforms = requested_transforms
        # Store this for later use
        self._last_op_is_to_tensor = isinstance(transforms[-1], ToTensor) if\
                                        transforms is not None and\
                                        len(transforms) > 0 else False


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
            if self._last_op_is_to_tensor:
                transforms.insert(-1, Normalize(self.mean, self.std))
            else:
                if not transforms:
                    transforms = []
                transforms.append(Normalize(self.mean, self.std))
            eval_transforms.append(Normalize(self.mean, self.std))

        # transforms can be None at this point if the user of this Dataset
        # passed None as the transforms argument, i.e. does not want any
        # transforms
        if transforms is not None:
            self.transforms = Compose(transforms)
        else:
            self.transforms = None

        if eval_transforms:
            # No need to check if self.transforms is None as it gets set only
            # if transforms is not None
            if self._last_op_is_to_tensor:
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
        elif self._last_op_is_to_tensor:
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
            # TODO not a nice solution since we store all images again
            raws_flattened = [raw for sub in self.raw_images for raw in sub]
            return np.mean(raws_flattened), np.std(raws_flattened)
        else:
            images = []
            for raw_image_paths in self.raw_image_paths:
                for raw_image_path in raw_image_paths:
                    image = tif.imread(raw_image_path)
                    images.append(image)

            try:
                mean, std = np.mean(images), np.std(images)
            except ValueError:
                raise ValueError('Are you using images of different shapes?')

            # Make sure the images get deleted right away
            del images
            return mean, std

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
                y = TrainingDataset._stratified_coord_y(box_size)
                x = TrainingDataset._stratified_coord_x(box_size)
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
            x_, y_ = indices_list[TrainingDataset\
                                    ._hot_pixel_replacement_index(
                                        len(indices_list))]

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

    def _get_sample(self, dataset_index, sample_index):
        gt_index = int(sample_index / self._factors[dataset_index])
        if self._keep_in_memory:
            # Important that we copy the image otherwise we are editing the
            # original
            raw_image = self.raw_images[dataset_index][sample_index].copy()
            images = self.gt_images[dataset_index]
            gt_image = self.gt_images[dataset_index][gt_index]
        else:
            raw_image = tif.imread(self.raw_image_paths[dataset_index][sample_index])
            gt_image = tif.imread(self.gt_image_paths[dataset_index][gt_index])

        sample = {'raw' : raw_image, 'gt' : gt_image}
        if self.transforms is not None:
            sample = self.transforms(sample)

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
        if self._last_op_is_to_tensor:
            # In case that we convert the raw and gt image to a torch tensor
            # then we have to convert the mask as well because the user wants
            # it to be performed automatically
            mask.shape = (mask.shape[0], mask.shape[1], 1)
            mask = functional.to_tensor(mask)
        sample['mask'] = mask
        return sample

    def __len__(self):
        """Returns the length of this dataset. All images of all specified
        datasets are counted.
        
        Returns:
            int -- the length of this dataset
        """
        flattend_indices = [index for sublist in self.train_indices for index in sublist]
        return len(flattend_indices)

    def __getitem__(self, idx):
        """Returns the image of the specified index. The Dataset class
        automatically retrieves the corresponding dataset of the given index.
        
        Arguments:
            idx {int} -- index of the desired sample
        
        Returns:
            dict -- containing the keys 'raw', 'gt' and 'mask'
                    'raw' and 'gt' are of the shape
                        [batch_size, channels, height, width]
                    'mask' is of the shape [batch_size, heigh, width]
        """
        dataset_index = 0
        # This is the actual index that the user requested
        while idx > len(self.train_indices[dataset_index]):
            dataset_index += 1
            idx -= len(self.train_indices[dataset_index])
        idx = self.train_indices[dataset_index][idx]
        return self._get_sample(dataset_index, idx)

    def __iter__(self):
        return self

    def _init_empty_batch(self, batch_size):
        if self._last_op_is_to_tensor:
            module = torch
        else:
            module = np
        raw = module.zeros((batch_size,) + self.image_shape)
        ground_truth = module.zeros((batch_size,) + self.image_shape)
        mask = module.zeros((batch_size,) + self.image_shape)
        return raw, ground_truth, mask

    def _refill_indices_if_needed(self, dataset_index):
        if not self._current_train_indices[dataset_index]:
            # We used up all the indices, thus refill the array
            train_indices = copy.deepcopy(self.train_indices[dataset_index])
            train_indices = TrainingDataset._train_indices_permutation(train_indices)
            self._current_train_indices[dataset_index] = train_indices
    
    def __next__(self):
        """This function assembles a batch of the earlier specified batch size
        using the provided datasets. Depending on the distribution mode either
        a dataset is randomly drawn and afterwards a random image is drawn from
        that dataset or an image is drawn with probability proportional to the
        size of the dataset it is from. These images are then used to fill the
        batch. Indices of images that are used to fill a batch are removed from
        the index list. After using up all indices of a dataset its index list
        is refilled again.
        
        Returns:
            dict -- containing the keys 'raw', 'gt' and 'mask'
                    'raw' and 'gt' are of the shape
                        [batch_size, channels, height, width]
                    'mask' is of the shape [batch_size, heigh, width]
        """
        raw, ground_truth, mask = self._init_empty_batch(self.batch_size)
        for i in range(self.batch_size):
            if self.distribution_mode == 'even':
                dataset_index = TrainingDataset._dataset_index_even(
                                                        self._num_datasets)
            else:
                dataset_sizes = [len(dataset) for dataset in self.train_indices]
                dataset_index = TrainingDataset._dataset_index_proportional(
                                                                dataset_sizes)
            # Shuffled already, we just take the first one
            sample_index = self._current_train_indices[dataset_index][0]
            self._current_train_indices[dataset_index].remove(sample_index)
            self._refill_indices_if_needed(dataset_index)
            sample = self._get_sample(dataset_index, sample_index)
            raw[i] = sample['raw']
            ground_truth[i] = sample['gt']
            mask[i] = sample['mask']
        return {'raw' : raw,
                'gt'  : ground_truth,
                'mask': mask}

    def validation_samples(self):
        batches = []
        total_item_count = 0
        total_indices = len([index for sub in self.val_indices for index in sub])
        # In case that we already have less validation images than requested
        # batch size
        batch_size = min(total_indices - total_item_count, self.batch_size)
        current_raw, current_gt, current_mask = self._init_empty_batch(batch_size)
        current_item_count = 0
        for dataset_index, val_indices in enumerate(self.val_indices):
            for val_index in val_indices:
                sample = self._get_sample(dataset_index, val_index)
                current_raw[current_item_count] = sample['raw']
                current_gt[current_item_count] = sample['gt']
                current_mask[current_item_count] = sample['mask']
                current_item_count += 1
                total_item_count += 1
                if current_item_count == batch_size:
                    batches.append({'raw' : current_raw,
                                    'gt'  : current_gt,
                                    'mask': current_mask})
                    current_item_count = 0
                    # Check if we have less remaining indices than the batch
                    # size would need
                    batch_size = min(total_indices - total_item_count,
                                     self.batch_size)
                    current_raw, current_gt, current_mask =\
                                self._init_empty_batch(self.batch_size)
        return batches

    def training_examples(self):
        samples = []
        for training_example in self._training_examples:
            raw_image = tif.imread(training_example['raw'])
            raw_image = util.normalize(raw_image, self.mean, self.std)
            raw_image = util.img_to_tensor(raw_image)
            gt_image = tif.imread(training_example['gt'])
            if len(gt_image.shape) == 2:
                # for compatibility
                gt_image.shape = gt_image.shape + (1,)
            if self.transforms is not None and\
                isinstance(self.transforms.transforms[-1], ToTensor):
                raw_image = torch.stack([raw_image]).float()
            sample = {'raw' : raw_image,
                      'gt'  : gt_image}
            samples.append(sample)
        return samples