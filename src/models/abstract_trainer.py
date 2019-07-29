import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from data import TrainingDataset
from data.transforms import *

class AbstractTrainer():
    """Class AbstractTrainer is the base class of all trainers. It automatically
    loads data from the locations specified in the config and creates the net.
    It establishes a training loop and lets the subclasses perform the actual
    training iterations.
    """

    def __init__(self, config, config_path):
        self.config_path = config_path
        self.config = config
        
        # Need to set here because those values might get overwritten by
        # subclasses when loading a saved net for further training
        self.train_hist = []
        self.val_hist = []
        self.epoch = 0
        self.running_loss = 0.0
        self.avg_train_loss = 0.0
        self.avg_val_loss = 0.0
        self.val_losses = []
        self.val_counter = 0
        self.print_step = 0

        self._load_config_parameters()
        self._construct_dataset()
        self.net = self._load_network()
        # Optimizer is needed to load network weights 
        # that's why we create it first
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.0001)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 'min', patience=10, factor=0.5, verbose=True)
        self._load_network_state()
        self.net.train(True)

        # If tensorboard logs are requested create the writer
        if self.write_tensorboard_data:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(os.path.join(
                self.experiment_base_path, 'tensorboard'))

    def _load_config_parameters(self):
        # Set all parameters from the config
        self.epochs = self.config['EPOCHS']
        self.val_ratio = self.config['VALIDATION_RATIO']
        self.val_size = self.config['VALIDATION_SIZE']
        # Virtual batch size
        self.vbatch = self.config['VIRTUAL_BATCH_SIZE']
        self.steps_per_epoch = self.config['STEPS_PER_EPOCH']
        self.experiment_base_path = self.config.get('EXPERIMENT_BASE_PATH',
                                                    self.config_path)
        if self.experiment_base_path == "":
            self.experiment_base_path = self.config_path
        # don't need config path anymore
        del self.config_path
        # Needed for prediction every X training runs
        self.ps = self.config['PRED_PATCH_SIZE']
        self.overlap = self.config['OVERLAP']
        self.bs = self.config['BATCH_SIZE']
        self.size = self.config['TRAIN_PATCH_SIZE']
        self.num_pix = self.size * self.size / 32.0
        self.data_counter = None
        self.box_size = np.round(
            np.sqrt(
                self.size * self.size / self.num_pix)).astype(np.int)
        self.write_tensorboard_data = self.config['WRITE_TENSORBOARD_DATA']

    def _construct_dataset(self):
        crop_width = self.config['TRAIN_PATCH_SIZE']
        crop_height = self.config['TRAIN_PATCH_SIZE']
        if 'CONVERT_DATA_TO' in self.config:
            transforms = [RandomCrop(crop_width, crop_height),
                          RandomFlip(),
                          RandomRotation(),
                          ConvertToFormat(self.config['CONVERT_DATA_to']),
                          ToTensor()]
        else:
            transforms = [RandomCrop(crop_width, crop_height),
                          RandomFlip(),
                          RandomRotation(),
                          ToTensor()]
        # We let the dataset automatically add a normalization term with the
        # mean and std computed of the data
        self.dataset = TrainingDataset(self.config['DATA_TRAIN_RAW_PATH'],
                                       self.config.get('DATA_TRAIN_GT_PATH', None),
                                       self.config['VAL_RATIO'],
                                       transforms=transforms,
                                       add_normalization_transform=True)
        self.raw_example, self.gt_example = self.dataset.const_training_example()
        self.data_loader = DataLoader(self.dataset, self.config['BATCH_SIZE'],
                                      shuffle=True, num_workers=os.cpu_count())

    def _load_network(self):
        raise NotImplementedError

    def _load_network_state(self):
        raise NotImplementedError

    def _create_checkpoint(self):
        raise NotImplementedError

    def _write_tensorboard_data(self):
        raise NotImplementedError

    def _perform_validation(self):
        raise NotImplementedError

    def _on_epoch_end(self, step, train_losses):
        # Needed by subclasses
        self.print_step = step + 1
        running_loss = (np.mean(train_losses))
        print("Step:", self.print_step, "| Avg. epoch loss:", running_loss)
        train_losses = np.array(train_losses)
        print("Avg. loss: "+str(np.mean(train_losses))+"+-" +
            str(np.std(train_losses)/np.sqrt(train_losses.size)))
        # Average loss for the current iteration
        # Need to store on class because subclasses need the loss
        self.avg_train_loss = np.mean(train_losses)
        self.train_hist.append(self.avg_train_loss)

        self.net.train(False)
        self.val_losses = []
        self.val_counter = 0

        self._perform_validation()

        # Need to store on class because subclasses need the loss
        self.avg_val_loss = np.mean(self.val_losses)
        self.net.train(True)

        # Save the current best network
        if len(self.val_hist) == 0 or self.avg_val_loss < np.min(self.val_hist):
            torch.save(
                self._create_checkpoint(),
                os.path.join(self.experiment_base_path, 'best.net'))
        self.val_hist.append(self.avg_val_loss)

        np.save(os.path.join(self.experiment_base_path, 'history.npy'),
                (np.array([np.arange(self.epoch), self.train_hist, self.val_hist])))

        self.scheduler.step(self.avg_val_loss)

        torch.save(
            self._create_checkpoint(),
            os.path.join(self.experiment_base_path, 'last.net'))

        if self.write_tensorboard_data:
            self._write_tensorboard_data()

    def _perform_epochs(self):
        raise 'This function needs to be implemented by the subclasses.'

    def train(self):
        """This method performs training of this network using the earlier
        set configuration and parameters.
        """
        self._perform_epochs()

        if self.write_tensorboard_data:
            # The graph is nonsense and otherwise we have to
            # store the outputs on the class
            #self.writer.add_graph(self.net, outputs)
            self.writer.close()

        print('Finished Training')