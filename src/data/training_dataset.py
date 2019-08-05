import os
import glob
import numpy as np
import tifffile as tif
import natsort
import torch
from torchvision.transforms import Compose, functional

import util
from data.transforms import ToTensor, Normalize, ConvertToFormat
import constants

class TrainingDataset():

    # How samples are drawn from the the specified datasets (i.e. image dirs)
    # 'even' means even and 'proportional' according to dataset size
    distribution_modes = ['even', 'proportional']

    num_datasets = 0
    # list of list of loaded raw images, if keep in memory is true
    raw_images = []
    gt_images = []
    # list of folders with raw images
    raw_images_dirs = []
    # list of lists of full paths to raw images
    raw_image_paths = []
    # analogously
    gt_images_dirs = []
    gt_image_paths = []
    # individual raw to gt factors
    factors = []
    # list of lists of indices
    train_indices = []
    # currently remaining train indices, gets filled up when all have been
    # returned once
    current_train_indices = []
    val_indices = []
    current_val_indices = []
    # we store one exapmle per dataset
    _training_examples = []

    def __init__(self, raw_images_dirs, gt_images_dirs=None, batch_size=24,
                 distribution_mode='proportional', val_ratio=0.1,
                 transforms=None, convert_to_format=None,
                 add_normalization_transform=True, keep_in_memory=True,
                 num_pixels=32.0, seed=constants.NP_RANDOM_SEED):
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
        self._mode = 'train'
        self._val_ratio = val_ratio
        self.raw_images_dirs = raw_images_dirs
        self.num_datasets = len(raw_images_dirs)
        self.gt_images_dirs = gt_images_dirs

        self._load_datasets(seed)

        raws_size = len(np.array(self.raw_image_paths).flatten())
        if self._train_mode == 'void':
            print('Using {} raw images and no gt images for training.'
                   .format(raws_size))
        else:
            print('Using {} raw and {} gt images for training.'
                  .format(raws_size,
                          len(np.array(self.gt_image_paths).flatten())))

        self.mean, self.std = self._compute_mean_and_std()
        print('Dataset has mean {} and standard deviation {}.'\
                                .format(self.mean, self.std))

        self._init_transform(transforms, add_normalization_transform,
                             convert_to_format)
        self.image_shape = self._get_sample(0, 0)['raw'].shape

    def _load_datasets(self, seed):
        for i, raw_images_dir in enumerate(self.raw_images_dirs):
            raw_image_paths, raw_images = self._load_raw_images(raw_images_dir)
            gt_images_dir, gt_image_paths, gt_images =\
                    self._load_gt_images(self.gt_images_dirs[i], raw_images_dir,
                                         raw_image_paths, raw_images)
            if gt_images is not None:
                factor = len(raw_image_paths) / len(gt_image_paths)
                assert factor.is_integer(), 'Number of raw images needs to be '+\
                                            'divisible by the number of ground-'+\
                                            'truth images.'

            self.raw_image_paths.append(raw_image_paths)
            if raw_images is not None:
                self.raw_images.append(raw_images)
            # Need to update as gt_images_dir gets set to raw_images_dir for
            # N2V training
            self.gt_images_dirs[i] = gt_images_dir
            self.gt_image_paths.append(gt_image_paths)
            if gt_images is not None:
                self.gt_images.append(gt_images)
                self.factors.append(factor)

            # One training example that is the same for all experiments
            np.random.seed(seed)
            example_index = np.random.randint(len(raw_image_paths))
            training_example = {'raw' : raw_image_paths[example_index],
                                'gt'  : gt_image_paths[example_index]}
            self._training_examples.append(training_example)

            # Create training and validation indices
            dataset_size = len(raw_image_paths)
            indices = np.array(range(dataset_size))
            split = int(self._val_ratio * dataset_size)
            # Seed numpy so that train and validation set are always the same for
            # all experiments
            indices = util.shuffle(indices, seed)
            train_indices, val_indices = indices[split:], indices[:split]
            self.train_indices.append(train_indices)
            # We call tolist() here because the current indices get altered
            # a lot during sample retrieval which is faster for plain
            # Python lists
            self.current_train_indices.append(train_indices.copy().tolist())
            self.val_indices.append(val_indices)
            self.current_val_indices.append(val_indices.copy().tolist())

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
                gt_images = None
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
            for raw_image_paths in self.raw_image_paths:
                for raw_image_path in raw_image_paths:
                    image = tif.imread(raw_image_path)
                    images.append(image)

            mean, std = np.mean(images), np.std(images)
            # Make sure the images get deleted right away
            del images
            return mean, std

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

    def _get_sample(self, dataset_index, sample_index):
        if self._keep_in_memory:
            raw_image = self.raw_images[dataset_index][sample_index]
            gt_image = self.gt_images[dataset_index]\
                            [int(sample_index / self.factors[dataset_index])]
        else:
            raw_image = tif.imread(self.raw_image_paths[dataset_index][sample_index])
            gt_image = tif.imread(self.gt_image_paths[dataset_index]\
                                    [int(sample_index / self.factors[dataset_index])])

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

    def _refill_indices_if_needed(self, dataset_index):
        if self._mode == 'train' and not self.current_train_indices[dataset_index]:
            # We used up all the indices, thus refill the array
            train_indices = self.train_indices.copy()
            train_indices = np.random.permutation(train_indices)
            self.current_train_indices = train_indices.tolist()
        elif self._mode and not self.current_val_indices[dataset_index]:
            val_indices = self.val_indices.copy()
            val_indices = np.random.permutation(val_indices)
            self.current_val_indices = val_indices

    def __iter__(self):
        return self
    
    def __next__(self):
        raw = torch.zeros((self.batch_size,) + self.image_shape)
        gt = torch.zeros((self.batch_size,) + self.image_shape)
        mask = torch.zeros((self.batch_size,) + self.image_shape)
        for i in range(self.batch_size):
            if self.distribution_mode == 'even':
                dataset_index = np.random.randint(self.num_datasets)
                self._refill_indices_if_needed(dataset_index)
                # Shuffled already, we just take the first one
                sample_index = self.current_train_indices[0]
                self.current_train_indices.remove(0)
                sample = self._get_sample(dataset_index, sample_index)
            else:
                # We globally draw a random sample and infer the corresponding
                # dataset afterwards
                zip_indices = []
                for dataset_index in range(self.num_datasets):
                    # Can't refill indices at beginning of function because
                    # the 'even' branch only needs to do this for one dataset
                    self._refill_indices_if_needed(dataset_index)
                    zip_indices.extend([(dataset_index, sample_index) for\
                                        sample_index in\
                                        self.current_train_indices[dataset_index]])
                index = zip_indices[np.random.randint(len(zip_indices))]
                self.current_train_indices[index[0]].remove(index[1])
                sample = self._get_sample(index[0], index[1])
        raw[i] = sample['raw']
        gt[i] = sample['gt']
        mask[i] = sample['mask']
        return {'raw' : raw,
                'gt'  : gt,
                'mask': mask}

    def validation_samples(self):
        samples = []
        for dataset_index, val_indices in enumerate(self.val_indices):
            for val_index in val_indices:
                if self._keep_in_memory:
                    raw_image = self.raw_images[dataset_index][val_index]
                    gt_image = self.gt_images[dataset_index][val_index]
                else:
                    raw_image = tif.imread(self.raw_image_paths[dataset_index][val_index])
                    gt_image = tif.imread(self.gt_image_paths[dataset_index][val_index])
                raw_image = torch.stack([util.img_to_tensor(raw_image)])
                gt_image = torch.stack([util.img_to_tensor(gt_image)])
                mask = torch.ones(raw_image.shape)
                samples.append({'raw' : raw_image,
                                'gt'  : gt_image,
                                'mask': mask})
        return samples

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
            sample = {'raw' : raw_image,
                      'gt'  : gt_image}
            samples.append(sample)
        return samples