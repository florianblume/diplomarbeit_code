import numpy as np

from models import AbstractTrainer
from models.reinforce import ReinforceUNet

class Trainer(AbstractTrainer):

    def __init__(self, config, config_path):
        self.train_loss = 0.0
        self.train_losses = []
        self.val_loss = 0.0
        self.samples = []
        self.results = []
        super(Trainer, self).__init__(config, config_path)

    def _load_network(self):
        self.config['MEAN'] = self.dataset.mean
        self.config['STD'] = self.dataset.std
        return ReinforceUNet(self.config)

    def _store_parts_of_eval_sample(self, sample, result):
        self.samples.extend(sample['dataset'].cpu().detach().numpy())
        self.results.extend(result['actions'].cpu().detach().numpy())

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
                self.writer.add_scalar('eval.reinforce.dataset_{}.subnetwork_{}'
                                       .format(i, j),
                                       subnet_percentage, self.current_epoch)
        # Need to reset for next eval run
        self.samples = []
        self.results = []

    def _write_custom_tensorboard_data_for_example(self,
                                                   example_result,
                                                   example_index):
        # weights are either of shape [subnets] or [subnets, H, W]
        action_probs = example_result['action_probs']
        self.writer.add_histogram('example.reinforces.subnets',
                                  action_probs,
                                  self.current_epoch,
                                  bins='auto')
        for i, action_prob in enumerate(action_probs):
            action_prob_name = 'example_{}.reinforce.subnet.{}'.format(example_index, i)
            self.writer.add_scalar(action_prob_name,
                                   action_prob,
                                   self.current_epoch)
