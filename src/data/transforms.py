import numpy as np
import torchvision

class RandomCrop():

    def __init__(self, width: int, height: int, hot_pixels=64):
        self.width = width
        self.height = height
        self.hot_pixels = hot_pixels

    def __call__(self, sample):
        pass

class AbstractNumpyAction():

    def __init__(self, numpy_action):
        self.numpy_action = numpy_action

    def __call__(self, sample):
        raw_image = sample['raw']
        raw_image = np.array(self.numpy_action(raw_image))
        mask = sample['mask']
        mask = np.array(self.numpy_action(mask))
        if 'gt' in sample:
            # N2C training
            gt_image = sample['gt']
            gt_image = np.array(self.numpy_action(gt_image))
        else:
            # N2V training
            gt_image = raw_image

        return {'raw' : raw_image, 'gt' : gt_image, 'mask' : mask}

class RandomRotation(AbstractNumpyAction):

    @staticmethod
    def _numpy_action(image):
        rot = np.random.randint(0, 4)
        return np.array(np.rot90(image, rot))

    def __init__(self):
        super(RandomRotation, self).__init__(RandomRotation._numpy_action)

class RandomFlip(AbstractNumpyAction):

    def __init__(self):
        super(RandomFlip, self).__init__(np.flip)

class TrainingAndPredictionAction():

    def __init__(self, action, mask_action):
        self.action = action
        self.mask_action = mask_action

    def __call__(self, sample):
        raw_image = sample['raw']
        raw_image = self.action(raw_image)
        has_gt, has_mask = False, False
        if 'mask' in sample:
            mask = sample['mask']
            mask = self.mask_action(mask)
            has_mask = True
        if 'gt' in sample:
            gt_image = sample['gt']
            gt_image = self.action(gt_image)
            has_gt = True
        elif has_mask:
            # We have a mask i.e. are training, if no gt_image is present
            # we train N2V style and need to set the gt image to the training
            # image
            gt_image = raw_image
        if has_mask:
            # Training
            return {'raw' : raw_image, 'gt' : gt_image, 'mask' : mask}
        if has_gt:
            # Prediction with clean target
            return {'raw' : raw_image, 'gt' : gt_image}
        # Prediction without clean target
        return {'raw' : raw_image}

class ConvertToFormat(TrainingAndPredictionAction):

    def _action(self, image):
        image.astype(np.dtype(self.to_format))

    def _mask_action(self, mask):
        return mask

    def __init__(self, to_format):
        self.to_format = to_format
        super(ConvertToFormat, self).__init__(self._action, self._mask_action)

class ToTensor(TrainingAndPredictionAction):
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

    @staticmethod
    def _mask_action(image):
        return ToTensor._action(image)

    def __init__(self, to_format):
        self.to_format = to_format
        super(ToTensor, self).__init__(ToTensor._action, ToTensor._mask_action)
        