import torch
import torchvision
import torch.distributions as tdist
import torch.optim as optim
import numpy as np
import os
import matplotlib.pyplot as plt
import importlib

import util
from models import AbstractTrainer
from models.probabilistic import StandaloneSubUNet

class SubnetworkTrainer(AbstractTrainer):

    def _load_network(self):
        # Device gets automatically created in constructor
        # We persist mean and std when saving the network
        self.net = StandaloneSubUNet(self.dataset.mean, self.dataset.std, 
                                     depth=self.config['DEPTH'])

    def _write_custom_tensorboard_data_for_example(self, resut):
        pass
