import pytest
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

# We want the local torchtest that we modified
from tests.torchtest import assert_vars_change
from tests import base_test
from models.average import ImageWeightUNet

def test_average_model():
    inputs = torch.rand(1, 1, 20, 20)
    mask = torch.ones_like(inputs)
    targets = Variable(torch.randn((20, 20)))
    batch = {'raw' : inputs,
             'gt'  : targets,
             'mask': mask}
    # Cuda backend not yet working
    device = torch.device("cpu")
    model = ImageWeightUNet(
                1, 0, 0, in_channels=1,
                main_net_depth=1, sub_net_depth=3, num_subnets=2,
                 start_filts=64, up_mode='transpose', merge_mode='add',
                 augment_data=True, device=device)

    # what are the variables?
    print('List of parameters', [np[0] for np in model.named_parameters()])

    # do they change after a training step?
    #  let's run a train step and see
    assert_vars_change(
        model=model,
        loss_fn=F.mse_loss,
        optim=torch.optim.Adam(model.parameters()),
        batch=batch,
        #patch_size=128,
        #overlap=48,
        device=device)
