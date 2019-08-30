
from models import AbstractTrainer
from models.q_learning import QUNet

class Trainer(AbstractTrainer):

    def __init__(self, config, config_path):
        self.train_loss = 0.0
        self.train_losses = []
        self.val_loss = 0.0
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

    def _write_custom_tensorboard_data_for_example(self,
                                                   example_result,
                                                   example_index):
        # weights are either of shape [subnets] or [subnets, H, W]
        q_values = example_result['q_values']
        for i, q_value in enumerate(q_values):
            q_values_name = 'example_{}.q_vlaues.subnet.{}'.format(example_index, i)
            self.writer.add_histogram(q_values_name,
                                      q_value,
                                      self.current_epoch,
                                      bins='auto')

    def _on_step_end(self):
        next_epsilon = self.net.epsilon * self.config['EPSILON_DECAY']
        self.net.epsilon = max(next_epsilon, self.config['EPSILON_MIN'])
