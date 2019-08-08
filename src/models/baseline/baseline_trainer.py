import os

from models import AbstractTrainer
from models.baseline import UNet

class Trainer(AbstractTrainer):
    """The trainer for the baseline network.
    """

    def _load_network(self):
        # Device gets automatically created in constructor
        # We persist mean and std when saving the network
        return UNet(self.config['NUM_CLASSES'],
                    self.dataset.mean, self.dataset.std,
                    in_channels=self.config['IN_CHANNELS'],
                    depth=self.config['DEPTH'],
                    augment_data=self.config['AUGMENT_DATA'])
