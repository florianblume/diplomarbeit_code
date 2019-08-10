import torch
import torch.nn.functional as F
from torch.autograd import Variable

# We want the local torchtest that we modified
from tests.torchtest import assert_vars_change
from tests import base_test
from models.average import ImageWeightUNet
from models.average import PixelWeightUNet

def test_average_model():
    inputs = torch.rand(1, 1, 20, 20)
    mask = torch.ones_like(inputs)
    targets = Variable(torch.randn((20, 20)))
    batch = {'raw' : inputs,
             'gt'  : targets,
             'mask': mask}
    # Cuda backend not yet working
    device = torch.device("cpu")
    model = ImageWeightUNet(mean=1, std=0, in_channels=1,
                            main_net_depth=1, sub_net_depth=1, num_subnets=2,
                            prediction_patch_size=10, prediction_patch_overlap=5,
                            weight_constraint=None, weights_lambda=0.1,
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

def test_average_model_parameter_setting():
    device = torch.device("cpu")
    model = ImageWeightUNet(mean=1, std=0, in_channels=1,
                            main_net_depth=1, sub_net_depth=1, num_subnets=2,
                            prediction_patch_size=10, prediction_patch_overlap=5,
                            weight_constraint=None, weights_lambda=0.1,
                            start_filts=64, up_mode='transpose', merge_mode='add',
                            augment_data=True, device=device)
    assert model.mean == 1
    assert model.std == 0
    assert model.in_channels == 1
    assert model.depth == 1
    assert model.num_subnets == 2
    assert model.weight_constraint == None
    assert model.weights_lambda == 0.1

    model = ImageWeightUNet(mean=0, std=1, in_channels=3,
                            main_net_depth=1, sub_net_depth=1, num_subnets=2,
                            prediction_patch_size=10, prediction_patch_overlap=5,
                            weight_constraint='entropy', weights_lambda=0.01,
                            start_filts=64, up_mode='transpose', merge_mode='add',
                            augment_data=True, device=device)
    assert model.mean == 0
    assert model.std == 1
    assert model.in_channels == 3
    assert model.depth == 1
    assert model.num_subnets == 2
    assert model.weight_constraint == 'entropy'
    assert model.weights_lambda == 0.01

    model = PixelWeightUNet(mean=1, std=0, in_channels=1,
                            main_net_depth=1, sub_net_depth=1, num_subnets=2,
                            prediction_patch_size=10, prediction_patch_overlap=5,
                            weight_constraint=None, weights_lambda=0.1,
                            start_filts=64, up_mode='transpose', merge_mode='add',
                            augment_data=True, device=device)
    assert model.mean == 1
    assert model.std == 0
    assert model.in_channels == 1
    assert model.depth == 1
    assert model.num_subnets == 2
    assert model.weight_constraint == None
    assert model.weights_lambda == 0.1

    model = PixelWeightUNet(mean=0, std=1, in_channels=3,
                            main_net_depth=1, sub_net_depth=1, num_subnets=2,
                            prediction_patch_size=10, prediction_patch_overlap=5,
                            weight_constraint='entropy', weights_lambda=0.01,
                            start_filts=64, up_mode='transpose', merge_mode='add',
                            augment_data=True, device=device)
    assert model.mean == 0
    assert model.std == 1
    assert model.in_channels == 3
    assert model.depth == 1
    assert model.num_subnets == 2
    assert model.weight_constraint == 'entropy'
    assert model.weights_lambda == 0.01
