import os
import numpy as np
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

    def load_training_data(self, data_raw_path: str, data_gt_path: str):
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

        Returns:
            np.array -- normalized raw data array
            np.array -- normalized gt data array
        """
        print("Loading training data...")

        data_raw_path = os.path.join(self.data_base_path, data_raw_path)
        data_raw = np.load(data_raw_path)
        print("Loaded " +
              str(data_raw.shape[0] + " images from " + data_raw_path))

        data_gt_path = os.path.join(self.data_base_path, data_gt_path)
        data_gt = np.load(data_gt_path)
        print("Loaded " +
              str(data_gt.shape[0] + " images from " + data_gt_path))

        img_factor = int(data_raw.shape[0]/data_gt.shape[0])
        print("Raw - GT ratio: " + img_factor)

        # Normalize
        self._mean = np.mean(data_raw)
        self._std = np.std(data_raw)
        print("Mean: {}, Std: {}".format(self._mean, self._std))

        print("Normalizing data...")
        data_raw = util.normalize(data_raw, self._mean, self._std)
        data_gt = util.normalize(data_gt, self._mean, self._std)
        # We need to repeat the gt data because we usually have one
        # GT image for multiple raw images
        print("Repeating ground-truth data {} times...".format(img_factor))
        data_gt = np.repeat(data_gt, img_factor, axis=0)

        return data_raw, data_gt

    def load_test_data(self, data_raw_path: str, data_gt_path: str):
        """Loads the the data for prediction at the specified path
        and normalizes it using the mean and std from the raw training
        data loaded via the load_training_data function.

        Arguments:
            raw_data_path {str} -- path to the saved numpy array of raw
                                   data relative to the base path
            gt_data_path {str} -- path to the saved numpy array of ground-truth
                                   data relative to the base path

        Returns:
            np.array -- the normalized prediction data
        """

        print("Loading prediction data...")
        data_raw = np.load(os.path.join(self.data_base_path, data_raw_path))
        data_gt = np.load(os.path.join(self.data_base_path, data_gt_path))
        print("Normalizing data...")
        data_raw = util.normalize(data_raw, self._mean, self._std)
        data_gt = util.normalize(data_gt, self._mean, self._std)

        # TODO Unclear what this is for
        #dataTestGT = np.load(path+"../gt/test_gt.npy")

        return data_raw, data_gt
