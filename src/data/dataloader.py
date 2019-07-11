import os
import numpy as np
import sys

main_path = os.getcwd()
sys.path.append(os.path.join(main_path, 'src'))

import util

class DataLoader():
    """Helper class to load trainind and prediction data. The data loader
    automatically stores the mean and std of the last loaded raw training
    data and normalizes all datasets with these two values.
    """

    def __init__(self, data_base_path: str):
        """
        Arguments:
            data_base_path {str} -- the base path to the data, when loading
                                    data the paths are assumed to be relative
                                    to the base path
        """
        self.data_base_path = data_base_path
        self._mean = 0
        self._std = 0

    def mean(self):
        """Returns the mean of the raw data set that was loaded last
        via the load_training_data function.

        Returns:
            int -- the mean of the last loaded raw training set
        """
        return self._mean

    def std(self):
        """Returns the std of the raw data set that was loaded last
        via the load_training_data function.

        Returns:
            int -- the std of the last loaded raw training set
        """
        return self._std

    def load_training_data(self, data_raw_path: str, data_gt_path: str, convert_to=None):
        """Loads the raw and ground truth data from the given paths for
        training. The data also gets normalized by the mean and std of
        the raw data. These values are also saved in this DataLoader and
        can be access through mean() and std().

        Arguments:
            raw_data_path {str}  -- path to the saved numpy array of 
                                    raw data relative to the base path
            gt_data_path {str}   -- path to the saved numpy array of
                                    ground truth data relative to the
                                    base path
            convert_to {str}     -- if specified, the loaded data is converted
                                    to the desired numpy dtype - for more
                                    information on types see 
                                    https://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html

        Returns:
            np.array -- normalized raw data array
            np.array -- normalized gt data array
        """
        print("Loading training data...")

        data_raw_path = os.path.join(self.data_base_path, data_raw_path)
        data_raw = np.load(data_raw_path)
        if convert_to is not None:
            data_raw = data_raw.astype(np.dtype(convert_to))
        print(data_raw.dtype)

        print("Loaded " +
              str(data_raw.shape[0]) + " images from " + data_raw_path)

        if data_gt_path != "":
            data_gt_path = os.path.join(self.data_base_path, data_gt_path)
            data_gt = np.load(data_gt_path)
            if convert_to is not None:
                data_gt = data_gt.astype(np.dtype(convert_to))
            print("Loaded " +
                  str(data_gt.shape[0]) + " images from " + data_gt_path)
            img_factor = int(data_raw.shape[0]/data_gt.shape[0])
            print("Raw - GT ratio: " + str(img_factor))
        else:
            data_gt = None

        # Normalize
        self._mean = np.mean(data_raw)
        self._std = np.std(data_raw)
        print("Mean: {}, Std: {}".format(self._mean, self._std))

        print("Normalizing data...")
        data_raw = util.normalize(data_raw, self._mean, self._std)

        if data_gt is not None:
            data_gt = util.normalize(data_gt, self._mean, self._std)
            # We need to repeat the gt data because we usually have one
            # GT image for multiple raw images
            print("Repeating ground-truth data {} times...".format(img_factor))
            data_gt = np.repeat(data_gt, img_factor, axis=0)

        return data_raw, data_gt

    def load_test_data(self, data_raw_path: str, data_gt_path: str, 
                            mean: int, std: int, convert_to=None):
        """Loads the the data for prediction at the specified path
        and normalizes it using the mean and std from the raw training
        data loaded via the load_training_data function.

        Arguments:
            raw_data_path {str} -- path to the saved numpy array of raw
                                   data relative to the base path
            gt_data_path {str}  -- path to the saved numpy array of ground-truth
                                   data relative to the base path, can be "" to return None
            mean {int}          -- the mean to normalize the raw data with
            std {int}           -- the std to normalize the raw data with
            convert_to {str}    -- if specified, the loaded data is converted
                                    to the desired numpy dtype - for more
                                    information on types see 
                                    https://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html

        Returns:
            np.array         -- the normalized prediction data
            np.array or None -- the ground-truth data, if specified (None if path is "")
        """

        print("Loading prediction data...")
        data_raw = np.load(os.path.join(self.data_base_path, data_raw_path))
        if convert_to is not None:
            data_raw = data_raw.astype(np.dtype(convert_to))
        if data_gt_path != "":
            data_gt = np.load(os.path.join(self.data_base_path, data_gt_path))
            if convert_to is not None:
                data_gt = data_gt.astype(np.dtype(convert_to))
        else:
            None
        print("Normalizing RAW data with mean {} and std {}...".format(mean, std))

        data_raw = util.normalize(data_raw, mean, std)

        return data_raw, data_gt