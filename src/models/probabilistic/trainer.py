from models import AbstractTrainer
from models.probabilistic import SubUNet
from models.probabilistic import ImageProbabilisticUNet
from models.probabilistic import PixelProbabilisticUNet

class ProbabilisticTrainer(AbstractTrainer):

    def _load_network(self):
        pass

class SubnetworkTrainer(AbstractTrainer):

    def _load_network(self):
        self.config['IS_INTEGRATED'] = True
        return SubUNet(self.config)

    def _write_custom_tensorboard_data_for_example(self, example_result,
                                                   example_index):
        # We store the standard deviance because it allows assumptions about
        # how certain the network is about pixel values
        std = example_result['std'].squeeze()
        name = 'std_example_{}'.format(example_index)
        self.writer.add_image(name, std, self.current_epoch, dataformats='HW')