import numpy as np
import torchvision

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

        x = np.random.randint(0, raw_image.shape[1] - self.width)
        y = np.random.randint(0, raw_image.shape[0] - self.height)

        cropped_raw_image = raw_image[y:y+self.height, x:x+self.width].copy()
        cropped_gt_image = gt_image[y:y+self.height, x:x+self.width].copy()

        return {'raw' : cropped_raw_image, 'gt' : cropped_gt_image}

class RandomFlip():
    """Transformation that randomly flips the data in the sample.
    """

    def __call__(self, sample):
        raw_image = sample['raw']
        if 'gt' in sample:
            gt_image = sample['gt']
            if np.random.choice((True, False)):
                raw_image = np.flip(raw_image)
                gt_image = np.flip(gt_image)
            return {'raw' : raw_image, 'gt' : gt_image}
        if np.random.choice((True, False)):
            raw_image = np.flip(raw_image)
        return {'raw' : raw_image}

class AbstractActionTransformation():
    """Base class for a transformation that is to be executed on training and
    prediction data. I.e. it is not clear whether there is a gt image, as
    prediciton can be run without one.
    """

    def __init__(self, action):
        self.action = action

    def __call__(self, sample):
        raw_image = sample['raw']
        raw_image = self.action(raw_image)
        if 'gt' in sample:
            gt_image = sample['gt']
            gt_image = self.action(gt_image)
            has_gt = True
            # Prediction with clean images or training
            return {'raw' : raw_image, 'gt' : gt_image}
        # Prediction without clean images
        return {'raw' : raw_image}

class RandomRotation(AbstractActionTransformation):
    """Transformation that randomly rotates the data in the sample.
    """

    @staticmethod
    def _numpy_action(image):
        rot = np.random.randint(0, 4)
        return np.array(np.rot90(image, rot))

    def __init__(self):
        super(RandomRotation, self).__init__(RandomRotation._numpy_action)

class ConvertToFormat(AbstractActionTransformation):
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

class ToTensor(AbstractActionTransformation):
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
        image = torchvision.transforms.functional.to_tensor(image)
        return image

    def __init__(self):
        super(ToTensor, self).__init__(ToTensor._action)
        