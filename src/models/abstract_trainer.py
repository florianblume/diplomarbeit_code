import os
import time
import logging
import numpy as np
import torch
import torch.optim as optim

import util
from data import TrainingDataset
from data.transforms import RandomCrop, RandomFlip, RandomRotation, ToTensor

class AbstractTrainer():
    """Class AbstractTrainer is the base class of all trainers. It automatically
    loads data from the locations specified in the config and creates the net.
    It establishes a training loop and lets the subclasses perform the actual
    training iterations.
    """
    
    def _init_attributes(self):
        self.experiment_base_path = None

        self.epochs = 0
        self.virtual_batch_size = 0
        self.steps_per_epoch = 0
        self.patch_size = 0
        self.overlap = 0
        self.write_tensorboard_data = False

        self.current_epoch = 0
        self.current_step = 0
        self.running_loss = 0.0

        self.train_loss = None
        self.avg_train_loss = 0.0
        self.train_hist = []
        self.train_losses = []

        self.val_loss = None
        self.avg_val_loss = 0.0
        self.val_hist = []
        self.val_losses = []
        self.val_counter = 0

    def __init__(self, config, config_path):
        self._init_attributes()
        self.config = config
        self.config_path = config_path
        self._load_config_params()
        self._construct_dataset()
        self.net = self._load_network()
        # Optimizer is needed to load network weights
        # that's why we create it first
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.0001)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 'min', patience=10, factor=0.5, verbose=True)
        self._load_state_from_checkpoint()
        self.net.train(True)

        # If tensorboard logs are requested create the writer
        if self.write_tensorboard_data:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(os.path.join(
                self.experiment_base_path, 'tensorboard'))

    def _load_config_params(self):
        self.experiment_base_path = self.config.get('EXPERIMENT_BASE_PATH',
                                                    self.config_path)
        if self.experiment_base_path == "":
            self.experiment_base_path = self.config_path

        self.epochs = self.config['EPOCHS']
        self.virtual_batch_size = self.config['VIRTUAL_BATCH_SIZE']
        self.steps_per_epoch = self.config['STEPS_PER_EPOCH']
        self.patch_size = self.config['PRED_PATCH_SIZE']
        self.overlap = self.config['OVERLAP']
        self.write_tensorboard_data = self.config['WRITE_TENSORBOARD_DATA']

    def _construct_dataset(self):
        print('Constructing dataset...')
        crop_width = self.config['TRAIN_PATCH_SIZE']
        crop_height = self.config['TRAIN_PATCH_SIZE']

        transforms = []
        if self.config['AUGMENT_DATA']:
            transforms = [RandomCrop(crop_width, crop_height),
                          RandomFlip(),
                          RandomRotation(),
                          ToTensor()]
        else:
            transforms = [ToTensor()]

        data_base_dir = self.config['DATA_BASE_DIR']
        data_train_raw_dirs = []
        for data_train_raw_dir in self.config['DATA_TRAIN_RAW_DIRS']:
            data_train_raw_dir = os.path.join(data_base_dir, data_train_raw_dir)
            data_train_raw_dirs.append(data_train_raw_dir)
        if 'DATA_TRAIN_GT_DIRS' in self.config:
            data_train_gt_dirs = []
            for data_train_gt_dir in self.config['DATA_TRAIN_GT_DIRS']:
                data_train_gt_dir = os.path.join(data_base_dir, data_train_gt_dir)
                data_train_gt_dirs.append(data_train_gt_dir)
        else:
            data_train_gt_dirs = None

        # We let the dataset automatically add a normalization term with the
        # mean and std computed of the data
        convert_to_format = self.config.get('CONVERT_TO_FORMAT', None)
        self.dataset = TrainingDataset(data_train_raw_dirs,
                                       data_train_gt_dirs,
                                       self.config['BATCH_SIZE'],
                                       self.config['DISTRIBUTION_MODE'],
                                       self.config['VALIDATION_RATIO'],
                                       transforms=transforms,
                                       convert_to_format=convert_to_format,
                                       add_normalization_transform=True,
                                       keep_in_memory=True)
        self.training_examples = self.dataset.training_examples()

    def _load_network(self):
        raise NotImplementedError

    def _load_custom_data_from_checkpoint(self, checkpoint):
        # Can be overwritten by subclasses
        pass

    def _load_state_from_checkpoint(self):
        train_network_path = self.config.get('TRAIN_NETWORK_PATH', None)
        if train_network_path is not None:
            train_network_path = os.path.join(self.experiment_base_path,
                                              self.config.get(
                                                  'TRAIN_NETWORK_PATH', None))
            checkpoint = torch.load(train_network_path)
            self.net.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizier_state_dict'])
            self.current_epoch = checkpoint['epoch']
            self.current_step = checkpoint['current_step']
            self.mean = checkpoint['mean']
            self.std = checkpoint['std']
            self.running_loss = checkpoint['running_loss']
            self.train_loss = checkpoint['train_loss']
            self.train_hist = checkpoint['train_hist']
            self.val_loss = checkpoint['val_loss']
            self.val_hist = checkpoint['val_hist']
            self._load_custom_data_from_checkpoint(checkpoint)

    def _custom_checkpoint_data(self):
        # Can be overwritten by subclasses
        return {}

    def _create_checkpoint(self):
        default_dict = {'model_state_dict': self.net.state_dict(),
                        'optimizier_state_dict': self.optimizer.state_dict(),
                        'epoch': self.current_epoch,
                        'current_step' : self.current_step,
                        'mean': self.dataset.mean,
                        'std': self.dataset.std,
                        'running_loss': self.running_loss,
                        'train_loss': self.train_loss,
                        'train_hist' : self.train_hist,
                        'val_loss': self.val_loss,
                        'val_hist': self.val_hist}
        default_dict.update(self._custom_checkpoint_data())
        return default_dict

    def _write_custom_tensorboard_data_for_example(self, 
                                                   example_result, 
                                                   example_index):
        # Can be overwritten by subclasses
        pass

    def train(self):
        """This method performs training of this network using the earlier
        set configuration and parameters.
        """
        print('Training...')
        print('')
        # loop over the dataset multiple times
        iterator = iter(self.dataset)
        # We start at current_step because if we are further training the net
        # this gets saved to the dict
        for step in range(self.current_step, self.epochs):
            self.current_step = step
            self.train_losses = []
            self.optimizer.zero_grad()

            # Iterate over virtual batch
            start = time.clock()
            for i in range(self.virtual_batch_size):
                sample = next(iterator)
                result = self.net.training_predict(sample)
                self.train_loss = self.net.loss_function(result)
                self.train_loss.backward()
                self.running_loss += self.train_loss.item()
                self.train_losses.append(self.train_loss.item())
            logging.debug('Training {} virtual batches took {:.4f}s.'
                          .format(self.virtual_batch_size, time.clock() - start))

            start = time.clock()
            self.optimizer.step()
            logging.debug('Optimization step took {:.4f}s.'
                          .format(time.clock() - start))
            if step % self.steps_per_epoch == self.steps_per_epoch - 1:
                start = time.clock()
                self.current_epoch = (step + 1) // self.steps_per_epoch
                self._on_epoch_end()
                logging.debug('Validation took {:.4f}s'
                              .format(time.clock() - start))

        if self.write_tensorboard_data:
            # The graph is nonsense and otherwise we have to
            # store the outputs on the class
            #self.writer.add_graph(self.net, outputs)
            self.writer.close()

        print('Finished Training')

    def _on_epoch_end(self):
        # Needed by subclasses
        #self.running_loss = (np.mean(self.train_losses))
        print("Epoch:", self.current_epoch,
              "| Avg. epoch loss:", np.mean(self.train_losses))
        self.train_losses = np.array(self.train_losses)
        print("Avg. loss: "+str(np.mean(self.train_losses))+"+-" +
            str(np.std(self.train_losses)/np.sqrt(self.train_losses.size)))
        # Average loss for the current iteration
        # Need to store on class because subclasses need the loss
        self.avg_train_loss = np.mean(self.train_losses)
        self.train_hist.append(self.avg_train_loss)

        self._perform_eval()

        if self.write_tensorboard_data:
            self._write_tensorboard_data()

        print('')

    def _perform_eval(self):
        # Perform evaluation
        self.net.train(False)
        self.val_losses = []
        self.val_counter = 0
        validation_samples = self.dataset.validation_samples()
        for i, sample in enumerate(validation_samples):
            if i == self.config['MAX_VALIDATION_SIZE']:
                print('Ran validation for only {} of {} available images.'\
                      .format(i, len(validation_samples)))
                break
            result = self.net.training_predict(sample)
            # Needed by subclasses that's why we store val_loss on self
            self.val_loss = self.net.loss_function(result)
            self.val_losses.append(self.val_loss.item())
        print("Validation loss: {}".format(self.val_loss.item()))

        # Need to store on class because subclasses need the loss
        self.avg_val_loss = np.mean(self.val_losses)
        self.net.train(True)

        # Save the current best network
        if len(self.val_hist) == 0 or\
           self.avg_val_loss < np.min(self.val_hist):
            torch.save(
                self._create_checkpoint(),
                os.path.join(self.experiment_base_path, 'best.net'))
        self.val_hist.append(self.avg_val_loss)

        np.save(os.path.join(self.experiment_base_path, 'history.npy'),
                (np.array([np.arange(self.current_epoch),
                           self.train_hist,
                           self.val_hist])))
        torch.save(
            self._create_checkpoint(),
            os.path.join(self.experiment_base_path, 'last.net'))

        self.scheduler.step(self.avg_val_loss)

    def _write_tensorboard_data(self):
        # +1 because epochs start at 0
        print_step = self.current_epoch + 1
        self.writer.add_scalar('train_loss',
                               self.avg_train_loss,
                               print_step)
        self.writer.add_scalar('val_loss', self.avg_val_loss, print_step)

        self.net.train(False)
        psnrs = []
        for i, training_example in enumerate(self.training_examples):
            # Predict for one example image
            result = self.net.predict(training_example['raw'])
            prediction = result['output']
            if 'gt' in training_example:
                ground_truth = training_example['gt']
                psnr = util.PSNR(ground_truth, prediction, 255)
                self.writer.add_scalar('psnr_example_{}'.format(i),
                                       psnr,
                                       print_step)
                psnrs.append(psnr)
            prediction = prediction.astype(np.uint8)
            if prediction.shape[-1] == 1 or len(prediction.shape) == 2:
                # Grayscale image
                prediction = prediction.squeeze()
                self.writer.add_image('pred_example_{}'.format(i), prediction,
                                      print_step, dataformats='HW')
            else:
                # RGB image
                self.writer.add_image('pred_example_{}'.format(i), prediction,
                                      print_step, dataformats='HWC')
            self._write_custom_tensorboard_data_for_example(result, i)
        mean_psnr = np.mean(psnrs)
        print('Avg. PSNR on {} examples'.format(len(self.training_examples)),
              mean_psnr)
        self.writer.add_scalar('mean_psnr', mean_psnr, print_step)
        self.net.train(True)

        for name, param in self.net.named_parameters():
            self.writer.add_histogram(
                name, param.clone().cpu().data.numpy(), print_step)