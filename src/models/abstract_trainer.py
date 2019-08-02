import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import util
from data import TrainingDataset
from data.transforms import RandomCrop, RandomFlip, RandomRotation,\
                            ConvertToFormat, ToTensor
from models.training_run import TrainingRun

class AbstractTrainer():
    """Class AbstractTrainer is the base class of all trainers. It automatically
    loads data from the locations specified in the config and creates the net.
    It establishes a training loop and lets the subclasses perform the actual
    training iterations.
    """

    def __init__(self, config, config_path):
        self.config = config
        self.run = TrainingRun.from_config(config, config_path)
        self._construct_dataset()
        self.net = self._load_network()
        # Optimizer is needed to load network weights
        # that's why we create it first
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.0001)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 'min', patience=10, factor=0.5, verbose=True)
        self._load_network_state_from_checkpoint()
        self.net.train(True)

        # If tensorboard logs are requested create the writer
        if self.run.write_tensorboard_data:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(os.path.join(
                self.run.experiment_base_path, 'tensorboard'))

    def _construct_dataset(self):
        print('Constructing dataset...')
        crop_width = self.config['TRAIN_PATCH_SIZE']
        crop_height = self.config['TRAIN_PATCH_SIZE']

        transforms = []
        if self.config['AUGMENT_DATA']:
            transforms = [RandomCrop(crop_width, crop_height),
                          RandomFlip(),
                          RandomRotation()]
        if 'CONVERT_DATA_TO' in self.config:
            transforms.extend([ConvertToFormat(self.config['CONVERT_DATA_TO']),
                              ToTensor()])
        else:
            transforms.append(ToTensor())

        data_base_dir = self.config['DATA_BASE_DIR']
        data_train_raw_dir = os.path.join(data_base_dir,
                                          self.config['DATA_TRAIN_RAW_DIRS'][0])
        if 'DATA_TRAIN_GT_DIRS' in self.config:
            data_train_gt_dir = os.path.join(data_base_dir,
                                             self.config['DATA_TRAIN_GT_DIRS'][0])
        else:
            data_train_gt_dir = None

        # We let the dataset automatically add a normalization term with the
        # mean and std computed of the data
        self.dataset = TrainingDataset(data_train_raw_dir,
                                        data_train_gt_dir,
                                       self.config['VALIDATION_RATIO'],
                                       transforms=transforms,
                                       add_normalization_transform=True)
        self.raw_example = self.dataset.training_example()['raw']
        self.gt_example = self.dataset.training_example()['gt']
        train_sampler = SubsetRandomSampler(self.dataset.train_indices)
        val_sampler = SubsetRandomSampler(self.dataset.val_indices)
        self.train_loader = DataLoader(self.dataset, self.config['BATCH_SIZE'],
                                        num_workers=os.cpu_count(),
                                        sampler=train_sampler)
        self.val_loader = DataLoader(self.dataset, 1,
                                      num_workers=os.cpu_count(),
                                      sampler=val_sampler)

    def _load_network(self):
        raise NotImplementedError

    def _load_network_state_from_checkpoint(self):
        raise NotImplementedError

    def _create_checkpoint(self):
        raise NotImplementedError

    def _write_custom_tensorboard_data(self):
        pass

    def _write_tensorboard_data(self):
        print_step = self.run.current_step + 1
        self.writer.add_scalar('train_loss',
                               self.run.avg_train_loss, 
                               print_step)
        self.writer.add_scalar('val_loss', self.run.avg_val_loss, print_step)

        self.net.train(False)
        # Predict for one example image
        result = self.net.predict(self.raw_example,
                                  self.run.patch_size,
                                  self.run.overlap)
        prediction = result['result']
        self.net.train(True)
        if self.gt_example is not None:
            gt = util.denormalize(self.gt_example,
                                  self.dataset.mean,
                                  self.dataset.std)
            psnr = util.PSNR(gt, prediction, 255)
            self.writer.add_scalar('psnr', psnr, print_step)

        prediction = prediction.astype(np.uint8)
        self.writer.add_image('pred', prediction, print_step, dataformats='HW')

        for name, param in self.net.named_parameters():
            self.writer.add_histogram(
                name, param.clone().cpu().data.numpy(), print_step)

    def _validation(self):
        self.dataset.eval()
        for i, sample in enumerate(self.val_loader):
            if i == self.config['VALIDATION_SIZE']:
                break
            result = self.net.training_predict(sample)
            # Needed by subclasses that's why we store val_loss on self
            self.run.val_loss = self.net.loss_function(result)
            self.run.val_losses.append(self.run.val_loss.item())
        print("Validation loss: {}".format(self.run.val_loss.item()))
        self.dataset.train()

    def _on_epoch_end(self, step, train_losses):
        # Needed by subclasses
        print_step = self.run.current_step + 1
        running_loss = (np.mean(train_losses))
        print("Step:",print_step, "| Avg. epoch loss:", running_loss)
        train_losses = np.array(train_losses)
        print("Avg. loss: "+str(np.mean(train_losses))+"+-" +
            str(np.std(train_losses)/np.sqrt(train_losses.size)))
        # Average loss for the current iteration
        # Need to store on class because subclasses need the loss
        self.run.avg_train_loss = np.mean(train_losses)
        self.run.train_hist.append(self.run.avg_train_loss)

        self.net.train(False)
        self.run.val_losses = []
        self.run.val_counter = 0

        self._validation()

        # Need to store on class because subclasses need the loss
        self.run.avg_val_loss = np.mean(self.run.val_losses)
        self.net.train(True)

        # Save the current best network
        if len(self.run.val_hist) == 0 or\
           self.run.avg_val_loss < np.min(self.run.val_hist):
            torch.save(
                self._create_checkpoint(),
                os.path.join(self.run.experiment_base_path, 'best.net'))
        self.run.val_hist.append(self.run.avg_val_loss)

        np.save(os.path.join(self.run.experiment_base_path, 'history.npy'),
                (np.array([np.arange(self.epoch),
                           self.run.train_hist,
                           self.run.val_hist])))

        self.scheduler.step(self.run.avg_val_loss)

        torch.save(
            self._create_checkpoint(),
            os.path.join(self.run.experiment_base_path, 'last.net'))

        if self.run.write_tensorboard_data:
            self._write_tensorboard_data()
            self._write_custom_tensorboard_data()

    def train(self):
        """This method performs training of this network using the earlier
        set configuration and parameters.
        """
        print('Training...')
        # loop over the dataset multiple times
        for step in range(self.run.epochs):
            self.train_losses = []
            self.optimizer.zero_grad()

            # Iterate over virtual batch
            for _ in range(self.run.virtual_batch_size):
                sample = next(iter(self.train_loader))
                result = self.net.training_predict(sample)
                self.run.train_loss = self.net.loss_function(result)
                self.run.train_loss.backward()
                self.run.running_loss += self.run.train_loss.item()
                self.run.train_losses.append(self.run.train_loss.item())

            self.optimizer.step()
            if step % self.run.steps_per_epoch == self.run.steps_per_epoch-1:
                self.epoch = step / self.run.steps_per_epoch
                self._on_epoch_end(step, self.train_losses)

        if self.run.write_tensorboard_data:
            # The graph is nonsense and otherwise we have to
            # store the outputs on the class
            #self.writer.add_graph(self.net, outputs)
            self.writer.close()

        print('Finished Training')