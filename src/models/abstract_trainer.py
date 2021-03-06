import os
import time
import yaml
import logging
import datetime
import subprocess
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import util

from data import TrainingDataset
from data import SequentialSampler
from data.transforms import Crop, RandomCrop, RandomFlip, RandomRotation, ToTensor

class AbstractTrainer():
    """Class AbstractTrainer is the base class of all trainers. It automatically
    loads data from the locations specified in the config and creates the net.
    It establishes a training loop and lets the subclasses perform the actual
    training iterations.
    """
    
    def _init_attributes(self):
        self.experiment_base_path = None

        # To measure how long the training took
        self.start_time = 0
        self._train_mode = None

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
        torch.manual_seed(int((time.time() * 100000) % 100000))
        self._init_attributes()
        self.config = config
        self.config_path = config_path
        self._load_config_params()
        self._log_config()
        self._construct_dataset()
        self.net = self._load_network()
        # Optimizer is needed to load network weights
        # that's why we create it first
        # Get the layers corresponding to the learning rates defined in the config
        learning_rates = []
        for learning_rate_key in self.config['LEARNING_RATES']:
            learning_rates.append({"params": self.net.params_for_key(learning_rate_key),
                                   "lr"    : self.config['LEARNING_RATES'][learning_rate_key]})
        self.optimizer = optim.Adam(learning_rates)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 'min', patience=10, factor=0.5, verbose=True)
        self._load_state_from_checkpoint()
        self.net.train(True)

        """
        A bit dangerous... If accidentially the wrong training was started the
        data is gone. To get rid of wrong tensorboard data just keep only the
        latest events file.

        if 'TRAIN_NETWORK_PATH' not in self.config:
            print('Removing previous training data...')
            # Remove old training artifacts if not specified to continue training
            path = os.path.join(self.experiment_base_path, 'tensorboard')
            if os.path.exists(path):
                shutil.rmtree(path)
            path = os.path.join(self.experiment_base_path, 'best.net')
            if os.path.exists(path):
                os.remove(path)
            path = os.path.join(self.experiment_base_path, 'last.net')
            if os.path.exists(path):
                os.remove(path)
            path = os.path.join(self.experiment_base_path, 'history.npy')
            if os.path.exists(path):
                os.remove(path)
        """

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

    def _log_config(self):
        log_dir = os.path.join(self.experiment_base_path, 'logs')
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        now = datetime.datetime.now()
        log_file_name = 'config_{}_{}_{}-{}_{}_{}.log'.format(now.year,
                                                              now.month,
                                                              now.day,
                                                              now.hour,
                                                              now.minute,
                                                              now.second)
        with open(os.path.join(log_dir, log_file_name), 'w') as log_file:
            config = self.config
            commit = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'])
            config['COMMIT'] = commit.decode('ascii').strip()
            yaml.dump(config, log_file, default_flow_style=False)

    def _construct_dataset(self):
        print('Constructing dataset...')
        crop_width = self.config['TRAIN_PATCH_SIZE']
        crop_height = self.config['TRAIN_PATCH_SIZE']

        assert crop_width > 0
        assert crop_height > 0

        train_transforms = []
        if self.config['AUGMENT_DATA']:
            train_transforms = [RandomCrop(crop_width, crop_height),
                                RandomFlip(),
                                RandomRotation(),
                                ToTensor()]
        else:
            # Not entirely correct that this is without augmentation but we
            # need the random crops to reduce image size
            train_transforms = [RandomCrop(crop_width, crop_height),
                                ToTensor()]

        data_base_dir = self.config['DATA_BASE_DIR']
        data_train_raw_dirs = []
        for data_train_raw_dir in self.config['DATA_TRAIN_RAW_DIRS']:
            data_train_raw_dir = os.path.join(data_base_dir, data_train_raw_dir)
            data_train_raw_dirs.append(data_train_raw_dir)
        if 'DATA_TRAIN_GT_DIRS' in self.config:
            self._train_mode = 'clean'
            data_train_gt_dirs = []
            for data_train_gt_dir in self.config['DATA_TRAIN_GT_DIRS']:
                data_train_gt_dir = os.path.join(data_base_dir, data_train_gt_dir)
                data_train_gt_dirs.append(data_train_gt_dir)
        else:
            data_train_gt_dirs = None
            self._train_mode = 'void'

        min_shape = util.min_image_shape_of_datasets(data_train_raw_dirs)
        assert min_shape[1] > crop_width
        assert min_shape[0] > crop_height
        offset_x = (min_shape[1] - crop_width) // 2
        offset_y = (min_shape[0] - crop_height) // 2
        eval_transforms = [Crop(offset_x, offset_y, crop_width, crop_height), ToTensor()]

        # We let the dataset automatically add a normalization term with the
        # mean and std computed of the data
        convert_to_format = self.config.get('CONVERT_TO_FORMAT', None)
        # No need to normalize our data as they are already normalized
        self.dataset = TrainingDataset(data_train_raw_dirs,
                                       data_train_gt_dirs,
                                       self.config['BATCH_SIZE'],
                                       self.config['DISTRIBUTION_MODE'],
                                       self.config['VALIDATION_RATIO'],
                                       train_transforms=train_transforms,
                                       eval_transforms=eval_transforms,
                                       convert_to_format=convert_to_format,
                                       add_normalization_transform=False,
                                       keep_in_memory=False)
        self.training_examples = self.dataset.training_examples()
        # NOTE: Using dataloader does not directly provide the possibility to
        # sample from the datasets evenly. If you want to achieve this you either
        # need to replicate the datasets so that their number of images match
        # or you use the dataset directly. This omits loading data in parallel.
        train_sampler = SubsetRandomSampler(self.dataset.flattened_train_indices)
        self.train_dataloader = DataLoader(self.dataset,
                                           batch_size=self.config['BATCH_SIZE'],
                                           sampler=train_sampler,
                                           num_workers=os.cpu_count())
        # Although we use a random sampler here the validation losses should
        # be comparable nevertheless because we were cropping our random patches
        # already anyway.
        val_sampler = SequentialSampler(self.dataset.flattened_val_indices)
        self.val_dataloader = DataLoader(self.dataset,
                                         batch_size=self.config['BATCH_SIZE'],
                                         sampler=val_sampler,
                                         num_workers=os.cpu_count())

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
            # The model parameter should be there, the rest below does not
            # necessarily have to
            self.net.load_state_dict(checkpoint['model_state_dict'], strict=False)

            # If there is no epoch there also shouldn't be any of the other keys.
            # The reason is that the user either wants to load a whole checkpoint
            # i.e. continue training where they have left off (maybe crash or so)
            # or load e.g. the subnetworks of the network which have been
            # previously trained.
            if 'epoch' in checkpoint:
                # If epoch is in the dict we just assume that the rest is, too.
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
        np.random.seed(int((time.time() * 100000) % 100000))
        print('Training...')
        print('')
        self.start_time = datetime.datetime.now()
        # loop over the dataset multiple times
        iterator = iter(util.cycle(self.train_dataloader))
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
                # Important to free CUDA memory
                del sample
                self._on_epoch_end()
                logging.debug('Validation took {:.4f}s'
                              .format(time.clock() - start))

            self._on_step_end()

        if self.write_tensorboard_data:
            # The graph is nonsense and otherwise we have to
            # store the outputs on the class
            #self.writer.add_graph(self.net, outputs)
            self.writer.close()

        print('Training took {}.'.format(datetime.datetime.now() - self.start_time))
        print('Finished Training')

    def _on_step_end(self):
        """Can be used by subclasses to perform some custom operations after
        each step during training.
        """
        pass

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
        i = 0
        val_start_time = time.clock()
        # Seed so that all training runs use the same crops of the validation
        # images and produce comparalbe PSNRs
        psnrs = []
        # To make the dataset use the evaluation transforms
        self.dataset.train(False)
        for sample in self.val_dataloader:
            if 'MAX_VALIDATION_SIZE' in self.config:
                if i >= self.config['MAX_VALIDATION_SIZE']:
                    print('Ran validation for only {} of {} available images'\
                        ' due to maximum validation size set to {}.'\
                        .format(i, len(self.dataset.flattened_val_indices),
                                self.config['MAX_VALIDATION_SIZE']))
                    break
            result = self.net.training_predict(sample)
            self._store_parts_of_eval_sample(sample, result)
            # Needed by subclasses that's why we store val_loss on self
            self.val_loss = self.net.loss_function(result).item()
            self.val_losses.append(self.val_loss)
            i += self.val_dataloader.batch_size

            if self._train_mode == 'clean':
                ground_truth = sample['gt'].cpu().detach().numpy()
                output = result['output'].cpu().detach().numpy()
                psnr = util.PSNR(ground_truth, output, self.dataset.range())
                psnrs.append(psnr)
            del sample

        self._post_process_eval_samples()

        if self._train_mode == 'clean':
            mean_psnr = np.mean(psnrs)
            print('Avg. PSNR of {} validation samples'.format(i),
                mean_psnr)
            self.writer.add_scalar('mean_psnr', mean_psnr, self.current_epoch + 1)

        print("Validation loss: {}".format(self.val_loss))
        logging.debug('Validation on {} images took {:.4f}s.'
                        .format(i, time.clock() - val_start_time))

        # Need to store on class because subclasses need the loss
        self.avg_val_loss = np.mean(self.val_losses)
        self.dataset.train(True)
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

    def _store_parts_of_eval_sample(self, sample, result):
        """This method can be used by subclasses to store important parts of
        the evaluation samples and results so that it can be used in the 
        _post_process_eval_samples method. This class can't store all samples
        due to memory constraints but doesn't know what parts are important
        and need to be store so the subclasses have to take care of it.
        
        Arguments:
            sample {dict} -- the sample that was used for evlauation
            result {dict} -- the result of running the network on the sample
        """

    def _post_process_eval_samples(self):
        """This method can be implemented by subclasses to perform some post
        processing on the evaluation samples, like logging additional metrics.
        
        Arguments:
            sample {dict} -- the sample containing at least the keys 'raw' and 'gt'
            result {dict} -- the result computed by the network
        """

    def _write_tensorboard_data(self):
        # +1 because epochs start at 0
        print_step = self.current_epoch + 1
        self.writer.add_scalar('train_loss', self.avg_train_loss, print_step)
        self.writer.add_scalar('val_loss', self.avg_val_loss, print_step)

        self.net.train(False)
        for i, training_example in enumerate(self.training_examples):
            # Predict for one example image
            result = self.net.predict(training_example['raw'])
            prediction = result['output']
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
            
        self.net.train(True)

        for name, param in self.net.named_parameters():
            self.writer.add_histogram(
                name, param.clone().cpu().data.numpy(), print_step)