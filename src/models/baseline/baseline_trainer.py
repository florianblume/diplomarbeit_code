import os

from models import AbstractTrainer
from models.baseline import UNet

class Trainer(AbstractTrainer):
    """The trainer for the baseline network.
    """

    def _load_network(self):
        # Device gets automatically created in constructor
        # We persist mean and std when saving the network
        self.config['MEAN'] = self.dataset.mean
        self.config['STD'] = self.dataset.std
        return UNet(self.config)
