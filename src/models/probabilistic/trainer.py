from models import AbstractTrainer
from models.probabilistic import SubUNet
from models.probabilistic import ImageProbabilisticUNet
from models.probabilistic import PixelProbabilisticUNet

class ProbabilisticTrainer(AbstractTrainer):

    def _load_network(self):
        self.config['IS_INTEGRATED'] = True
        self.config['MEAN'] = self.dataset.mean
        self.config['STD'] = self.dataset.std
        if self.config['WEIGHT_MODE'] == 'image':
            return ImageProbabilisticUNet(self.config)
        return PixelProbabilisticUNet(self.config)

    def _write_custom_tensorboard_data_for_example(self, example_result,
                                                   example_index):
        # We only have 1 batch, thus [0]
        std = example_result['std'].squeeze()[0]
        # Iterate over subnets
        for i, std_ in enumerate(std):
            name = 'std_example_{}.subnet.{}'.format(example_index, i)
            self.writer.add_image(name, std_, self.current_epoch, dataformats='HW')

class SubnetworkTrainer(AbstractTrainer):

    def _load_network(self):
        self.config['IS_INTEGRATED'] = False
        self.config['MEAN'] = self.dataset.mean
        self.config['STD'] = self.dataset.std
        return SubUNet(self.config)

    def _write_custom_tensorboard_data_for_example(self, example_result,
                                                   example_index):
        # We store the standard deviance because it allows assumptions about
        # how certain the network is about pixel values
        std = example_result['std'].squeeze()
        name = 'std_example_{}'.format(example_index)
        self.writer.add_image(name, std, self.current_epoch, dataformats='HW')