import numpy as np

from models import AbstractTrainer
from models.average import ImageWeightUNet
from models.average import PixelWeightUNet

class Trainer(AbstractTrainer):

    def __init__(self, config, config_path):
        self.train_loss = 0.0
        self.train_losses = []
        self.val_loss = 0.0
        super(Trainer, self).__init__(config, config_path)

    def _load_network(self):
        if self.config['WEIGHT_MODE'] == 'image':
            self.weight_mode = 'image'
            Network = ImageWeightUNet
        elif self.config['WEIGHT_MODE'] == 'pixel':
            self.weight_mode = 'pixel'
            Network = PixelWeightUNet
        else:
            raise 'Invalid config value for \"weight_mode\".'

        self.config['MEAN'] = self.dataset.mean
        self.config['STD'] = self.dataset.std
        return Network(self.config)

    def _write_custom_tensorboard_data_for_example(self,
                                                   example_result,
                                                   example_index):
        # weights are either of shape [subnets] or [subnets, H, W]
        weights = example_result['weights']
        # Normalize weights
        weights = weights / np.sum(weights, axis=0)
        for i, weights_ in enumerate(weights):
            weights_name = 'example_{}.weights.subnet.{}'.format(example_index, i)
            if self.weight_mode == 'image':
                self.writer.add_histogram(weights_name,
                                          weights_,
                                          self.current_epoch,
                                          bins='auto')
            elif self.weight_mode == 'pixel':
                color_space = weights_ * 255
                color_space = color_space.astype(np.uint8)
                # We only have grayscale weights
                self.writer.add_image(weights_name, color_space,
                                      self.current_epoch, dataformats='HW')
            else:
                raise ValueError('Unkown weight mode.')
