
from models import AbstractTrainer
from models.q_learning import QUNet

class Trainer(AbstractTrainer):

    def _load_network(self):
        return QUNet(config)
