import torch
import torchvision
import torch.distributions as tdist
import torch.optim as optim
import numpy as np
import os
import matplotlib.pyplot as plt
import importlib

# Import it dynamically later
#import network
import dataloader
import util

def create_checkpoint(model, optimizer, epoch, mean, std, train_loss, val_loss):
    return {'model_state_dict': model.state_dict(),
            'optimizier_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'mean': mean,
            'std': std,
            'train_loss': train_loss,
            'val_loss': val_loss}

def on_end_epoch(net, stepsPerEpoch, stepCounter, losses, trainHist, valHist,
                 valSize, data_val, data_val_gt, size, box_size, bs, data_raw, data_gt,
                 optimizer, scheduler, experiment_base_path, train_loss,
                 write_tensorboard_data, writer, loader, ps, overlap):
    print_step = stepCounter + 1
    epoch = (stepCounter / stepsPerEpoch)

    running_loss = (np.mean(losses))
    print("Step:", print_step, "| Avg. epoch loss:", running_loss)
    losses = np.array(losses)
    print("avg. loss: "+str(np.mean(losses))+"+-" +
          str(np.std(losses)/np.sqrt(losses.size)))

    # Average loss for the current iteration
    avg_train_loss = np.mean(losses)
    trainHist.append(avg_train_loss)
    losses = []

    valCounter = 0
    net.train(False)
    losses = []
    for _ in range(valSize):
        outputs, labels, masks, valCounter = net.training_predict(
            data_val, data_val_gt, valCounter, size, box_size, bs)
        val_loss = net.loss_function(outputs, labels, masks)
        losses.append(val_loss.item())
    net.train(True)
    avg_val_loss = np.mean(losses)

    # Save the current best network
    if len(valHist) == 0 or avg_val_loss < np.min(valHist):
        torch.save(create_checkpoint(net, optimizer, epoch, net.mean, net.std, train_loss, val_loss),
                   os.path.join(experiment_base_path, 'best.net'))
    valHist.append(avg_val_loss)

    np.save(os.path.join(experiment_base_path, 'history.npy'),
            (np.array([np.arange(epoch), trainHist, valHist])))

    scheduler.step(avg_val_loss)

    torch.save(create_checkpoint(net, optimizer, epoch, net.mean, net.std, train_loss, val_loss),
               os.path.join(experiment_base_path, 'last.net'))

    if write_tensorboard_data:
        tensorboard_data(writer, avg_train_loss, avg_val_loss,
                         print_step, net, data_raw, data_gt, loader, ps, overlap)


def tensorboard_data(writer, avg_train_loss, avg_val_loss,
                     print_step, net, data_raw, data_gt, loader, ps, overlap):
    writer.add_scalar('train_loss', avg_train_loss, print_step)
    writer.add_scalar('val_loss', avg_val_loss, print_step)

    net.train(False)
    # Predict for one example image
    raw = data_raw[0]
    prediction = net.predict(raw, ps, overlap)
    net.train(True)
    gt = data_gt[0]
    gt = util.denormalize(gt, loader.mean(), loader.std())
    psnr = util.PSNR(gt, prediction, 255)
    writer.add_scalar('psnr', psnr, print_step)

    # Ugly but it works
    plt.imsave('pred.png', prediction)
    pred = plt.imread('pred.png')
    writer.add_image('pred', pred, print_step, dataformats='HWC')
    os.remove('pred.png')

    for name, param in net.named_parameters():
        writer.add_histogram(
            name, param.clone().cpu().data.numpy(), print_step)


def train(config):
    loader, data_raw, data_gt, data_train, \
        data_train_gt, data_val, data_val_gt = load_data(config)

    # Device gets automatically created in constructor
    # We persist mean and std when saving the network
    network = importlib.import_module(config['NETWORK'] + ".network")
    net = network.UNet(config['NUM_CLASSES'], loader.mean(),
                       loader.std(), depth=config['DEPTH'])

    experiment_base_path = config['EXPERIMENT_BASE_PATH']

    # Needed for prediction every X training runs
    ps = config['PRED_PATCH_SIZE']
    overlap = config['OVERLAP']

    net.train(True)
    bs = config['BATCH_SIZE']
    size = config['TRAIN_PATCH_SIZE']
    num_pix = size * size / 32.0
    dataCounter = None
    box_size = np.round(np.sqrt(size * size / num_pix)).astype(np.int)

    vbatch = config['VIRTUAL_BATCH_SIZE']  # Virtual batch size
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=10, factor=0.5, verbose=True)

    running_loss = 0.0
    stepCounter = 0

    valSize = config['VALIDATION_SIZE']
    stepsPerEpoch = config['STEPS_PER_EPOCH']
    write_tensorboard_data = config['WRITE_TENSORBOARD_DATA']

    # If tensorboard logs are requested create the writer
    if write_tensorboard_data:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(os.path.join(
            experiment_base_path, 'tensorboard'))

    trainHist = []
    valHist = []

    # loop over the dataset multiple times
    for step in range(config['EPOCHS']):
        losses = []
        optimizer.zero_grad()
        stepCounter += 1

        # Iterate over virtual batch
        for _ in range(vbatch):

            """
            training_predict performs cutting out appropriate regions and replaces
            the center pixel with a random neighbor if NO clean targets are specified
            (i.e. we perform N2V), if clean targets are specified, no such replacement
            takes place
            """
            outputs, labels, masks, dataCounter = net.training_predict(
                data_train, data_train_gt, dataCounter, size, box_size, bs)

            train_loss = net.loss_function(outputs, labels, masks)
            train_loss.backward()
            running_loss += train_loss.item()
            losses.append(train_loss.item())

        optimizer.step()

        if stepCounter % stepsPerEpoch == stepsPerEpoch-1:
            on_end_epoch(net, stepsPerEpoch, stepCounter, losses, trainHist, valHist,
                    valSize, data_val, data_val_gt, size, box_size, bs, data_raw, data_gt,
                    optimizer, scheduler, experiment_base_path, train_loss,
                    write_tensorboard_data, writer, loader, ps, overlap)

            if stepCounter / stepsPerEpoch > 200:
                break

    if write_tensorboard_data:
        # Remove temp images
        writer.add_graph(net, outputs)
        writer.close()

    print('Finished Training')
