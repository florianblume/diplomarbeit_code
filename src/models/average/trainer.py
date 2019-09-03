import numpy as np

from models import AbstractTrainer
from models.average import ImageWeightUNet
from models.average import PixelWeightUNet

class Trainer(AbstractTrainer):

    def __init__(self, config, config_path):
        self.samples = []
        self.results = []
        self.train_loss = 0.0
        self.train_losses = []
        self.val_loss = 0.0
        super(Trainer, self).__init__(config, config_path)

    def _load_network(self):
        if self.config['WEIGHT_MODE'] == 'image':
            self.weight_mode = 'image'
            Network = ImageWeightUNet
        elif self.config['WEIGHT_MODE'] == 'pixel':
            self.weight_mode = 'pixel'
            Network = PixelWeightUNet
        else:
            raise 'Invalid config value for \"weight_mode\".'

        self.config['MEAN'] = self.dataset.mean
        self.config['STD'] = self.dataset.std
        return Network(self.config)

    def _store_parts_of_eval_sample(self, sample, result):
        if self.weight_mode == 'image':
            self.samples.extend(sample['dataset'].cpu().detach().numpy())
            self.results.extend(result['weights'].cpu().detach().numpy())

    def _post_process_eval_samples(self):
        if self.weight_mode == 'image':
            avg_probabilities = {}

            for i, dataset in enumerate(self.samples):
                for j in range(self.config['NUM_SUBNETS']):
                    if dataset not in avg_probabilities:
                        avg_probabilities[dataset] = {}
                    if j not in avg_probabilities[dataset]:
                        avg_probabilities[dataset][j] = []
                    avg_probabilities[dataset][j].append(self.results[i][j])

            for dataset_key in avg_probabilities:
                dataset = avg_probabilities[dataset_key]
                for subnetwork_key in dataset:
                    subnetwork_probabilities = dataset[subnetwork_key]
                    self.writer.add_scalar('eval.weights.dataset_{}.subnetwork_{}'
                                        .format(dataset_key, subnetwork_key),
                                        np.mean(subnetwork_probabilities), self.current_epoch)
            # Need to reset for next eval run
            self.samples = []
            self.results = []

    def _write_custom_tensorboard_data_for_example(self,
                                                   example_result,
                                                   example_index):
        # weights are either of shape [subnets] or [subnets, H, W]
        weights = example_result['weights']
        # Normalize weights
        weights = weights / np.sum(weights, axis=0)
        if self.weight_mode == 'image':
            self.writer.add_histogram('examples.weights.subnets',
                                      weights,
                                      self.current_epoch,
                                      bins='auto')
        for i, weights_ in enumerate(weights):
            weights_name = 'example_{}.weights.subnet.{}'.format(example_index, i)
            if self.weight_mode == 'image':
                self.writer.add_scalar(weights_name,
                                       weights_,
                                       self.current_epoch)
            elif self.weight_mode == 'pixel':
                color_space = weights_ * 255
                color_space = color_space.astype(np.uint8)
                # We only have grayscale weights
                self.writer.add_image(weights_name, color_space,
                                      self.current_epoch, dataformats='HW')
            else:
                raise ValueError('Unkown weight mode.')
