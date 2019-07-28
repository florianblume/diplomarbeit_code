import os
import glob
import numpy as np
from torch.utils.data import Dataset

class TrainingDataset(Dataset):

    def __init__(self, raw_images_dir, gt_images_dir=None,
                 val_ratio=0.1, transform=None, num_pixels=32.0):
        assert os.path.exists(raw_images_dir)
        assert os.path.isdir(raw_images_dir)
        # Assert that val_ratio is a sensible value
        assert 0.0 >= val_ratio <= 1.0

        self._raw_images_dir = raw_images_dir
        self._raw_images = glob.glob(os.path.join(raw_images_dir, "*.npy"))
        self._raw_images = np.array(self._raw_images)

        # If there are no ground-truth images we learn the network
        # Noise2Void style, otherwise we train it Noise2Clean
        if gt_images_dir is not None:
            assert os.path.isdir(gt_images_dir)
            self._train_mode = 'clean'
            self._gt_images_dir = gt_images_dir
            self._gt_images = glob.glob(os.path.join(gt_images_dir, "*.npy"))
            self._gt_images = np.array(self._gt_images)
            # Same number of raw and ground-truth images
            assert len(self._raw_images) == len(self._gt_images)
        else:
            # If we want to train N2V style
            self._train_mode = 'void'
            self._gt_images_dir = raw_images_dir
            self._gt_images = self._raw_images

        self._mean, self._std = TrainingDataset\
                                    ._compute_mean_and_std(self._raw_images)

        self._transform = transform
        self._num_pixels = num_pixels

        # Create training and validation set
        self._val_ratio = val_ratio
        val_index = int((1 - val_ratio) * self._raw_images.shape[0])
        self._raw_images_train = self._raw_images[:val_index].copy()
        self._raw_images_val = self._raw_images[val_index:].copy()
        self._gt_images_train = self._gt_images[:val_index].copy()
        self._gt_images_val = self._gt_images[val_index:].copy()

    @staticmethod
    def _compute_mean_and_std(images):
        # mean can be computed sequentially
        pass

    def __len__(self):
        return len(self._raw_images_train)

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
        raw_image = np.load(os.path.join(self._raw_images_dir,
                                         self._raw_images_train[idx]))
        gt_image = np.load(os.path.join(self._gt_images_dir,
                                        self._gt_images_train[idx]))
        sample = {'raw' : raw_image, 'gt' : gt_image}
        if self._transform:
            sample = self._transform(sample)

        # Retrieve the transformed image
        transformed_raw_image = sample['raw']
        if transformed_raw_image.shape[0] == 3:
            # Image has been transformed into tensor already, thus shape is
            # now [C, H, W]
            transformed_shape = (transformed_raw_image.shape[1],
                                 transformed_raw_image.shape[2])
        elif transformed_raw_image.shape[2] == 3:
            # Image has not been transformed into tensor already, thus shape is
            # still [H, W, C]
            transformed_shape = (transformed_raw_image.shape[0],
                                 transformed_raw_image.shape[1])
        else:
            raise ValueError('Unkown shape format. Neither [C, H, W] nor [H, W, C].')

        if self._train_mode == 'void':
            # Makes no sense to want more hot pixels than pixels in the patch
            assert self._num_pixels <= transformed_shape[0] * transformed_shape[1]
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
                # Cut out the ROI from the original image
                roi = transformed_raw_image[roi_min_y:roi_max_y,
                                            roi_min_x:roi_max_x]

                # Construct set of indices in the ROI excluding the hot pixel
                x_indices, y_indices = np.mgrid[roi_min_x:roi_max_x:1,
                                                roi_min_y:roi_max_y:1]
                indices = np.vstack((x_indices.flatten(), y_indices.flatten())).T
                indices_list = indices.tolist()
                # [2, 2] are the hot pixel's coordinates, we don't want to
                # sample it thus remove them from the list
                indices_list.remove([2, 2])
                # Obtain random pixel to replace hot pixel with
                x_, y_ = indices_list[np.random.randint(len(indices_list))]

                # Obtain neighbor to replace hot pixel with
                repl = roi[y_, x_]
                # Replace content of hot pixel
                transformed_raw_image[y, x] = repl
                # Set hot pixel to active
                mask[y, x] = 1.0
        else:
            # Noise2Clean style training, we have ground-truth data available
            # and thus mask all pixels to be active
            mask = np.ones(raw_image.shape)

        sample['mask'] = mask
        return sample

    def get_validation_images(self):
        images = []
        # We have fewer validation images than training images so we can just
        # load them all and return them
        for i, raw_image in self._raw_images_val:
            raw_image = np.load(os.path.join(self._raw_images_dir,
                                                raw_image))
            gt_image = np.load(os.path.join(self._gt_images_dir,
                                            self._gt_images_val[i]))
            images.append({'raw' : raw_image,
                            'gt'  : gt_image})
        return images
        