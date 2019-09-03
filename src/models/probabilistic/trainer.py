import numpy as np

from models import AbstractTrainer
from models.probabilistic import SubUNet
from models.probabilistic import ImageProbabilisticUNet
from models.probabilistic import PixelProbabilisticUNet

class Trainer(AbstractTrainer):

    def _load_network(self):
        self.config['IS_INTEGRATED'] = True
        self.config['MEAN'] = self.dataset.mean
        self.config['STD'] = self.dataset.std
        self.weight_mode = self.config['WEIGHT_MODE']
        if self.weight_mode == 'image':
            return ImageProbabilisticUNet(self.config)
        return PixelProbabilisticUNet(self.config)

    def _write_custom_tensorboard_data_for_example(self, example_result,
                                                   example_index):
        # We only have 1 batch, thus [0]
        std = example_result['std']
        # Iterate over subnets
        for i, std_ in enumerate(std):
            name = 'std_example_{}.subnet.{}'.format(example_index, i)
            std_ -= np.min(std_)
            std_ /= np.max(std_)
            #std_ *= 255
            std_ = std_.squeeze()
            self.writer.add_image(name, std_, self.current_epoch, dataformats='HW')

        probabilities = example_result['probabilities']
        if self.weight_mode == 'image':
            self.writer.add_histogram('example.probabilities.subnets',
                                      probabilities,
                                      self.current_epoch,
                                      bins='auto')
        for i, probabilities_ in enumerate(probabilities):
            probabilities_name = 'example_{}.probabilities.subnet.{}'.format(example_index, i)
            if self.weight_mode == 'image':
                self.writer.add_scalar(probabilities_name,
                                       probabilities_,
                                       self.current_epoch)
            elif self.weight_mode == 'pixel':
                color_space = probabilities_ * 255
                color_space = color_space.astype(np.uint8)
                # We only have grayscale weights
                self.writer.add_image(probabilities_name, color_space,
                                      self.current_epoch, dataformats='HW')
            else:
                raise ValueError('Unkown weight mode.')

class SubnetworkTrainer(AbstractTrainer):

    def _load_network(self):
        self.config['IS_INTEGRATED'] = False
        self.config['MEAN'] = self.dataset.mean
        self.config['STD'] = self.dataset.std
        return SubUNet(self.config)

    def _write_custom_tensorboard_data_for_example(self, example_result,
                                                   example_index):
        # We store the standard deviance because it allows assumptions about
        # how certain the network is about pixel values
        std = example_result['std'].squeeze()
        std -= np.min(std)
        std /= np.max(std)
        #std *= 255
        name = 'std_example_{}'.format(example_index)
        self.writer.add_image(name, std, self.current_epoch, dataformats='HW')
