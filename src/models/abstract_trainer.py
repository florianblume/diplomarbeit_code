import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from data import TrainingDataset
from data.transforms import RandomCrop, RandomFlip, RandomRotation,\
                            ConvertToFormat, ToTensor

class AbstractTrainer():
    """Class AbstractTrainer is the base class of all trainers. It automatically
    loads data from the locations specified in the config and creates the net.
    It establishes a training loop and lets the subclasses perform the actual
    training iterations.
    """

    def __init__(self, config, config_path):
        self.config_path = config_path
        self._config = config
        
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
        self._net = self._load_network()
        # Optimizer is needed to load network weights 
        # that's why we create it first
        self._optimizer = optim.Adam(self._net.parameters(), lr=0.0001)
        self._scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self._optimizer, 'min', patience=10, factor=0.5, verbose=True)
        self._load_network_state()
        self._net.train(True)

        # If tensorboard logs are requested create the writer
        if self.write_tensorboard_data:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(os.path.join(
                self.experiment_base_path, 'tensorboard'))

    def _load_config_parameters(self):
        # Set all parameters from the config
        self.epochs = self._config['EPOCHS']
        # Virtual batch size
        self.vbatch = self._config['VIRTUAL_BATCH_SIZE']
        self.steps_per_epoch = self._config['STEPS_PER_EPOCH']
        self.experiment_base_path = self._config.get('EXPERIMENT_BASE_PATH',
                                                    self.config_path)
        if self.experiment_base_path == "":
            self.experiment_base_path = self.config_path
        # don't need config path anymore
        del self.config_path
        # Needed for prediction every X training runs
        self.ps = self._config['PRED_PATCH_SIZE']
        self.overlap = self._config['OVERLAP']
        self.write_tensorboard_data = self._config['WRITE_TENSORBOARD_DATA']

    def _construct_dataset(self):
        print('Constructing dataset...')
        crop_width = self._config['TRAIN_PATCH_SIZE']
        crop_height = self._config['TRAIN_PATCH_SIZE']

        transforms = []
        if self._config['AUGMENT_DATA']:
            transforms = [RandomCrop(crop_width, crop_height),
                        RandomFlip(),
                        RandomRotation()]
        if 'CONVERT_DATA_TO' in self._config:
            transforms.extend([ConvertToFormat(self._config['CONVERT_DATA_TO']),
                              ToTensor()])
        else:
            transforms.append(ToTensor())

        data_base_dir = self._config['DATA_BASE_DIR']
        data_train_raw_dirs = self._config['DATA_TRAIN_RAW_DIRS']
        for i, data_train_raw_dir in enumerate(data_train_raw_dirs):
            data_train_raw_dirs[i] = os.path.join(data_base_dir,
                                                 data_train_raw_dir)
        data_train_gt_dirs = self._config.get('DATA_TRAIN_GT_DIRS', None)
        for i, data_train_gt_dir in enumerate(data_train_gt_dirs):
            data_train_gt_dirs[i] = os.path.join(data_base_dir,
                                                 data_train_gt_dir)

        # We let the dataset automatically add a normalization term with the
        # mean and std computed of the data
        self._dataset = TrainingDataset(data_train_raw_dirs,
                                        data_train_gt_dirs,
                                       self._config['VALIDATION_RATIO'],
                                       transforms=transforms,
                                       add_normalization_transform=True)
        self._raw_example = self._dataset.const_training_example()['raw']
        self._gt_example = self._dataset.const_training_example()['gt']
        train_sampler = SubsetRandomSampler(self._dataset.train_indices())
        val_sampler = SubsetRandomSampler(self._dataset.val_indices())
        self._train_loader = DataLoader(self._dataset, self._config['BATCH_SIZE'],
                                        num_workers=os.cpu_count(),
                                        sampler=train_sampler)
        self._val_loader = DataLoader(self._dataset, self._config['BATCH_SIZE'],
                                      num_workers=os.cpu_count(),
                                      sampler=val_sampler)

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

        self._net.train(False)
        self.val_losses = []
        self.val_counter = 0

        self._perform_validation()

        # Need to store on class because subclasses need the loss
        self.avg_val_loss = np.mean(self.val_losses)
        self._net.train(True)

        # Save the current best network
        if len(self.val_hist) == 0 or self.avg_val_loss < np.min(self.val_hist):
            torch.save(
                self._create_checkpoint(),
                os.path.join(self.experiment_base_path, 'best.net'))
        self.val_hist.append(self.avg_val_loss)

        np.save(os.path.join(self.experiment_base_path, 'history.npy'),
                (np.array([np.arange(self.epoch), self.train_hist, self.val_hist])))

        self._scheduler.step(self.avg_val_loss)

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
        print('Starting training...')
        self._perform_epochs()

        if self.write_tensorboard_data:
            # The graph is nonsense and otherwise we have to
            # store the outputs on the class
            #self.writer.add_graph(self.net, outputs)
            self.writer.close()

        print('Finished Training')