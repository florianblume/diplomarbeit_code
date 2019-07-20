import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchtest import assert_vars_change

import os
import sys

sys.path.append(os.path.join(os.getcwd(), 'src'))

from src.models.average.weight_network import ImageWeightUnet
from src.models.average.weight_network import PixelWeightUnet

def execute():
    inputs = Variable(torch.randn(20, 20))
    targets = Variable(torch.randn(20, 20))
    batch = [inputs, targets]
    model = ImageWeightUnet(
                1, 0, 0, in_channels=1,
                main_net_depth=1, sub_net_depth=3, num_subnets=2,
                 start_filts=64, up_mode='transpose', merge_mode='add',
                 augment_data=True)

    # what are the variables?
    print('Our list of parameters', [np[0] for np in model.named_parameters()])

    # do they change after a training step?
    #  let's run a train step and see
    assert_vars_change(
        model=model,
        loss_fn=F.cross_entropy,
        optim=torch.optim.Adam(model.parameters()),
        batch=batch,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))