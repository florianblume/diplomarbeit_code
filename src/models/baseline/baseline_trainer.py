import os
import torch

from models import AbstractTrainer
from models.baseline import UNet

class Trainer(AbstractTrainer):
    """The trainer for the baseline network.
    """

    def __init__(self, config, config_path):
        self.train_loss = 0.0
        self.train_losses = []
        self.val_loss = 0.0

        super(Trainer, self).__init__(config, config_path)

    def _load_network(self):
        # Device gets automatically created in constructor
        # We persist mean and std when saving the network
        return UNet(self.config['NUM_CLASSES'],
                    self.dataset.mean, self.dataset.std,
                    depth=self.config['DEPTH'],
                    augment_data=self.config['AUGMENT_DATA'])

    def _load_network_state_from_checkpoint(self):
        train_network_path = self.config.get('TRAIN_NETWORK_PATH', None)
        if train_network_path is not None:
            train_network_path = os.path.join(self.run.experiment_base_path,
                                              self.config.get(
                                                  'TRAIN_NETWORK_PATH', None))
            checkpoint = torch.load(train_network_path)
            self.net.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizier_state_dict'])
            self.epoch = checkpoint['epoch']
            self.mean = checkpoint['mean']
            self.std = checkpoint['std']
            self.running_loss = checkpoint['running_loss']
            self.train_loss = checkpoint['train_loss']
            self.train_hist = checkpoint['train_hist']
            self.val_loss = checkpoint['val_loss']
            self.val_hist = checkpoint['val_hist']

    def _create_checkpoint(self):
        return {'model_state_dict': self.net.state_dict(),
                'optimizier_state_dict': self.optimizer.state_dict(),
                'epoch': self.run.epoch,
                'mean': self.dataset.mean,
                'std': self.dataset.std,
                'running_loss': self.run.running_loss,
                'train_loss': self.run.train_loss,
                'train_hist' : self.run.train_hist,
                'val_loss': self.run.val_loss,
                'val_hist': self.run.val_hist}
