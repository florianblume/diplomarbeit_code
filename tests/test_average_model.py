import pytest
import torch
import torch.nn.functional as F
from torch.autograd import Variable

# We want the local torchtest that we modified
from tests.torchtest import assert_vars_change
from tests import base_test
from models.average import ImageWeightUNet
from models.average import PixelWeightUNet

@pytest.fixture
def config():
    return {'MEAN': 1, 'STD': 0, 'IN_CHANNELS': 1, 'MAIN_NET_DEPTH': 1,
            'SUB_NET_DEPTH': 1, 'NUM_SUBNETS': 2, 'PRED_PATCH_SIZE': 10,
            'WEIGHT_CONSTRAINT': None, 'WEIGHT_CONSTRAINT_LAMBDA': 0.1,
            'OVERLAP': 5, 'START_FILTS': 64, 'UP_MODE': 'transpose',
            'MERGE_MODE': 'add', 'AUGMENT_DATA': True, 'DEVICE': 'cpu'}

def test_average_model(config):
    inputs = torch.rand(1, 1, 20, 20)
    mask = torch.ones_like(inputs)
    targets = Variable(torch.randn((20, 20)))
    batch = {'raw' : inputs,
             'gt'  : targets,
             'mask': mask}
    # Cuda backend not yet working
    device = torch.device("cpu")
    config['WEIGHT_MODE'] = 'image'
    model = ImageWeightUNet(config)

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

def test_average_model_parameter_setting(config):
    device = torch.device("cpu")
    config['WEIGHT_MODE'] = 'image'
    model = ImageWeightUNet(config)
    assert model.mean == 1
    assert model.std == 0
    assert model.in_channels == 1
    assert model.depth == 1
    assert model.num_subnets == 2
    assert model.weight_constraint == None
    assert model.weight_constraint_lambda == 0.1

    config['WEIGHT_MODE'] = 'pixel'
    model = PixelWeightUNet(config)
    assert model.mean == 1
    assert model.std == 0
    assert model.in_channels == 1
    assert model.depth == 1
    assert model.num_subnets == 2
    assert model.weight_constraint == None
    assert model.weight_constraint_lambda == 0.1

    config['IN_CHANNELS'] = 3
    config['MEAN'] = 0
    config['STD'] = 1
    config['WEIGHT_CONSTRAINT'] = 'entropy'
    config['WEIGHT_CONSTRAINT_LAMBDA'] = 0.01

    config['WEIGHT_MODE'] = 'image'
    model = ImageWeightUNet(config)
    assert model.mean == 0
    assert model.std == 1
    assert model.in_channels == 3
    assert model.depth == 1
    assert model.num_subnets == 2
    assert model.weight_constraint == 'entropy'
    assert model.weight_constraint_lambda == 0.01

    config['WEIGHT_MODE'] = 'pixel'
    model = PixelWeightUNet(config)
    assert model.mean == 0
    assert model.std == 1
    assert model.in_channels == 3
    assert model.depth == 1
    assert model.num_subnets == 2
    assert model.weight_constraint == 'entropy'
    assert model.weight_constraint_lambda == 0.01
