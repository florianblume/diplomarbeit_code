import numpy as np

from models import AbstractTrainer
from models.q_learning import QUNet

class Trainer(AbstractTrainer):

    def __init__(self, config, config_path):
        self.train_loss = 0.0
        self.train_losses = []
        self.val_loss = 0.0
        self.samples = []
        self.results = []
        # Network expects key EPSILON
        config['EPSILON'] = config['EPSILON_START']
        super(Trainer, self).__init__(config, config_path)

    def _load_network(self):
        # Not sure if we are going to make this differentiation in the future
        """
        if self.config['WEIGHT_MODE'] == 'image':
            self.weight_mode = 'image'
            Network = ImageWeightUNet
        elif self.config['WEIGHT_MODE'] == 'pixel':
            self.weight_mode = 'pixel'
            Network = PixelWeightUNet
        else:
            raise 'Invalid config value for \"weight_mode\".'
        """
        self.config['MEAN'] = self.dataset.mean
        self.config['STD'] = self.dataset.std
        return QUNet(self.config)

    def _store_parts_of_eval_sample(self, sample, result):
        self.samples.extend(sample['dataset'].cpu().detach().numpy())
        self.results.extend(result['indices'].cpu().detach().numpy())

    def _post_process_eval_samples(self):
        bins = np.zeros((self.dataset.num_datasets, self.config['NUM_SUBNETS']))

        for i, dataset_index in enumerate(self.samples):
            # Indices of which subnetwork was used
            subnetwork_index = self.results[i]
            bins[dataset_index][subnetwork_index] += 1

        summed_bins = np.expand_dims(np.sum(bins, axis=1), -1)
        bins = np.divide(bins, summed_bins, out=bins, where=summed_bins != 0)

        # Only log for present datasets
        for i, bin_ in enumerate(bins):
            for j, subnet_percentage in enumerate(bin_):
                self.writer.add_scalar('eval.q_learning.dataset_{}.subnetwork_{}'
                                       .format(i, j),
                                       subnet_percentage, self.current_epoch)
        # Need to reset for next eval run
        self.samples = []
        self.results = []

    def _write_custom_tensorboard_data_for_example(self,
                                                   example_result,
                                                   example_index):
        # weights are either of shape [subnets] or [subnets, H, W]
        q_values = example_result['q_values']
        self.writer.add_histogram('examples.q_values.subnets',
                                  q_values,
                                  self.current_epoch,
                                  bins='auto')
        for i, q_value in enumerate(q_values):
            q_values_name = 'example_{}.q_vlaues.subnet.{}'.format(example_index, i)
            self.writer.add_scalar(q_values_name,
                                   q_value,
                                   self.current_epoch)

    def _on_step_end(self):
        next_epsilon = self.net.epsilon * self.config['EPSILON_DECAY']
        self.net.epsilon = max(next_epsilon, self.config['EPSILON_MIN'])
