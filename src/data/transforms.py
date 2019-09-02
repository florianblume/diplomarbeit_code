import numpy as np
import torchvision

import util

class Crop():
    """Class RandomCrop is a transformation that randomly crops out a part of
    the input data in the sample. It is only applicable to training data as it
    expects the gt data to be present.
    """

    def __init__(self, x: int, y: int, width: int, height: int):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def __call__(self, sample):
        # We are in training so there is a gt image, even if it has been set
        # to the raw image earlier
        raw_image = sample['raw']
        gt_image = sample['gt']
        assert raw_image.shape[0] >= self.height + self.y
        assert raw_image.shape[1] >= self.width + self.x
        assert raw_image.shape == gt_image.shape

        y = self.y
        x = self.x
        height = self.height
        width = self.width

        cropped_raw_image = raw_image[y:y+height, x:x+width].copy()
        cropped_gt_image = gt_image[y:y+height, x:x+width].copy()

        return {'raw' : cropped_raw_image, 'gt' : cropped_gt_image}

class RandomCrop():
    """Class RandomCrop is a transformation that randomly crops out a part of
    the input data in the sample. It is only applicable to training data as it
    expects the gt data to be present.
    """

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height

    def __call__(self, sample):
        # We are in training so there is a gt image, even if it has been set
        # to the raw image earlier
        raw_image = sample['raw']
        gt_image = sample['gt']
        assert raw_image.shape[0] >= self.height
        assert raw_image.shape[1] >= self.width
        assert raw_image.shape == gt_image.shape
        
        x = np.random.randint(0, raw_image.shape[1] - self.width + 1)
        y = np.random.randint(0, raw_image.shape[0] - self.height + 1)

        cropped_raw_image = raw_image[y:y+self.height, x:x+self.width].copy()
        cropped_gt_image = gt_image[y:y+self.height, x:x+self.width].copy()

        return {'raw' : cropped_raw_image, 'gt' : cropped_gt_image}

class RandomFlip():
    """Transformation that randomly flips the data in the sample.
    """

    def __call__(self, sample):
        raw_image = sample['raw']
        gt_image = sample['gt']
        if np.random.choice((True, False)):
            # (-2, -1) because when we have batch and channels as first dims
            raw_image = np.flip(np.flip(raw_image, -2), -1)
            gt_image = np.flip(np.flip(gt_image, -2), -1)
        return {'raw' : raw_image, 'gt' : gt_image}

class RandomRotation():
    """Transformation that randomly rotates the data in the sample.
    """

    def __call__(self, sample):
        raw_image = sample['raw']
        rot = np.random.randint(0, 4)
        gt_image = sample['gt']
        raw_image = np.rot90(raw_image, rot)
        gt_image = np.rot90(gt_image, rot)
        return {'raw' : raw_image, 'gt' : gt_image}

class SingleActionTransformation():
    """Base class for a transformation that apply actions to raw and ground-truth
    images. It is important that only transformations subclass this class that
    do not need their action carried out symmetrically on raw and ground-truth
    data. If you e.g. want to flip images randomly it might occur that in this
    class the raw image gets flipped but the ground-truth not.
    """

    def __init__(self, action):
        self.action = action

    def __call__(self, sample):
        raw_image = sample['raw']
        raw_image = self.action(raw_image)
        if 'gt' in sample:
            gt_image = sample['gt']
            gt_image = self.action(gt_image)
            # Prediction with clean images or training
            return {'raw' : raw_image, 'gt' : gt_image}
        # Prediction without clean images
        return {'raw' : raw_image}

class ConvertToFormat(SingleActionTransformation):
    """Class ConvertToFormat converts the contents of the sample to the specified
    numpy format. Check the numpy documentation to see available formats.
    Conversion might be lossy.
    """

    def _action(self, image):
        image = image.astype(self._to_format)
        return image

    def __init__(self, to_format):
        self._to_format = to_format
        super(ConvertToFormat, self).__init__(self._action)

class Normalize(SingleActionTransformation):
    """Class ConvertToFormat converts the contents of the sample to the specified
    numpy format. Check the numpy documentation to see available formats.
    Conversion might be lossy.
    """

    def _action(self, image):
        return util.normalize(image, self._mean, self._std)

    def __init__(self, mean, std):
        self._mean = mean
        self._std = std
        super(Normalize, self).__init__(self._action)

class ToTensor(SingleActionTransformation):
    """Class ToTensor takes in a training or prediction sample, converts it to
    the format required by PyTorch (C, H, W) and returns it as a tensor. It can
    be used both for training and for prediction as ToTensor automatically
    separates the cases.
    """

    @staticmethod
    def _action(image):
        if len(image.shape) == 2:
            # (H, W) image but PyTorch expects (H, W, C) to convert to tensor
            image.shape = (image.shape[0], image.shape[1], 1)
        # if there are transforms before this one that e.g. flip the image using
        # numpy then numpy does something like image[..., ::-1] internally.
        # Pytorch can't handle negative strides that's why we get rid of this
        # by subtracting zeros. Discussion at 
        # https://discuss.pytorch.org/t/torch-from-numpy-not-support-negative-strides/3663/10
        image = image - np.zeros_like(image)
        image = torchvision.transforms.functional.to_tensor(image)
        return image

    def __init__(self):
        super(ToTensor, self).__init__(ToTensor._action)
        