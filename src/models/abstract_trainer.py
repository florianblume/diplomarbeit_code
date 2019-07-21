import os
import torch
import torchvision
import torch.optim as optim
import torch.distributions as tdist
import numpy as np

import util
from data import dataloader

class AbstractTrainer():

    def __init__(self, config):
        self.config_path = os.path.dirname(config)
        self.config = util.load_config(config)

    def _load_config_parameters(self):
        # Set all parameters from the config
        self.epochs = self.config['EPOCHS']
        self.val_ratio = self.config['VALIDATION_RATIO']
        self.val_size = self.config['VALIDATION_SIZE']
        # Virtual batch size
        self.vbatch = self.config['VIRTUAL_BATCH_SIZE']
        self.steps_per_epoch = self.config['STEPS_PER_EPOCH']
        self.experiment_base_path = self.config.get('EXPERIMENT_BASE_PATH', self.config_path)
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

    def _load_data(self):
        # The actual loading of the images is performed by the util on demand
        # here we only load the filenames
        self.loader = dataloader.DataLoader(self.config['DATA_BASE_PATH'])
        # In case the ground truth data path was not set we pass '' to
        # the loader which returns None to us
        self.data_raw, self.data_gt = self.loader.load_training_data(
            self.config['DATA_TRAIN_RAW_PATH'], 
            self.config.get('DATA_TRAIN_GT_PATH', ''),
            self.config.get('CONVERT_DATA_TO', None))

        if self.data_gt is not None:
            data_raw, data_gt = util.joint_shuffle(self.data_raw, self.data_gt)
            # If loaded, the network is trained using clean targets, otherwise it performs N2V
            val_gt_index = int((1 - self.val_ratio) * data_gt.shape[0])
            self.data_train_gt = data_gt[:val_gt_index].copy()
            self.data_val_gt = data_gt[val_gt_index:].copy()
        else:
            self.data_train_gt = None
            self.data_val_gt = None
            data_raw = np.random.shuffle(data_raw)

        val_raw_index = int((1 - self.val_ratio) * data_raw.shape[0])
        self.data_train = data_raw[:val_raw_index].copy()
        self.data_val = data_raw[val_raw_index:].copy()
        print('Using {} raw images for training and {} raw images for validation.'\
                    .format(self.data_train.shape[0], self.data_val.shape[0]))
        if self.data_train_gt.shape[0] > 0:
            print('Using {} gt images for training and {} gt images for validation.'\
                .format(self.data_train_gt.shape[0], self.data_val_gt.shape[0]))
        else:
            print('No ground-truth images available for training.')

    def _load_network(self):
        raise 'This function needs to be implemented by the subclasses.'

    def _load_network_weights(self):
        raise 'This function needs to be implemented by the subclasses.'

    def _create_checkpoint(self):
        raise 'This function needs to be implemented by the subclasses.'

    def _write_tensorboard_data(self):
        raise 'This function needs to be implemented by the subclasses.'

    def _perform_validation(self):
        raise 'This function needs to be implemented by the subclasses.'

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
        # Need to set here because those values might get overwritten by
        # subclasses when loading a saved net for further training
        self.train_hist = []
        self.val_hist = []
        self.epoch = 0
        self.running_loss = 0.0
        self.net = None

        self._load_config_parameters()
        self._load_data()
        self._load_network()
        # Optimizer is needed to load network weights that's why we create it first
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.0001)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 'min', patience=10, factor=0.5, verbose=True)
        self._load_network_weights()
        self.net.train(True)

        # If tensorboard logs are requested create the writer
        if self.write_tensorboard_data:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(os.path.join(
                self.experiment_base_path, 'tensorboard'))

        self._perform_epochs()

        if self.write_tensorboard_data:
            # The graph is nonsense and otherwise we have to
            # store the outputs on the class
            #self.writer.add_graph(self.net, outputs)
            self.writer.close()

        print('Finished Training')