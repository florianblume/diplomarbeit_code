import torch
import torch.distributions as tdist
import torch.optim as optim
import numpy as np
import os
import argparse

import network
import dataloader
import util

def main(config):
    # data_c=np.concatenate((data.copy(),dataTest.copy()))
    loader = dataloader.DataLoader(config['DATA_BASE_PATH'])
    data_raw, data_gt = loader.load_training_data(config['DATA_TRAIN_RAW_PATH'], config['DATA_TRAIN_GT_PATH'])
    data_raw, data_gt = util.joint_shuffle(data_raw, data_gt)

    data_train = data_raw.copy()
    data_val = data_raw.copy()

    # If loaded, the network is trained using clean targets, otherwise it performs N2V
    data_train_gt = data_gt.copy()
    data_val_gt = data_gt.copy()

    #device = torch.device("cpu")
    # Device gets automatically created in constructor
    # Mean and std will be persisted by the network when it is saved
    net = network.UNet(1, loader.mean, loader.std, depth=config['DEPTH'])

    net.train(True)
    bs = config['BATCH_SIZE']
    size = config['TRAIN_PATCH_SIZE']
    num_pix = size * size / 32.0
    dataCounter = None
    #box_size = np.round(np.sqrt(size * size / num_pix)).astype(np.int)

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
        writer = SummaryWriter(config['HISTORY_PATH'])

    trainHist = []
    valHist = []

    for step in range(config['EPOCHS']):  # loop over the dataset multiple times
        losses = []
        optimizer.zero_grad()
        stepCounter += 1

        # Iterate over virtual batch
        for a in range(vbatch):

            """
            training_predict performs cutting out appropriate regions and replaces
            the center pixel with a random neighbor if NO clean targets are specified
            (i.e. we perform N2V), if clean targets are specified, no such replacement
            takes place
            """
            outputs, labels, masks, dataCounter = net.training_predict(
                data_train, data_train_gt, dataCounter, size, bs)

            train_loss = net.loss_function(outputs, labels, masks)
            train_loss.backward()
            running_loss += train_loss.item()
            losses.append(train_loss.item())

        optimizer.step()

        if stepCounter % stepsPerEpoch == stepsPerEpoch-1:
            running_loss = (np.mean(losses))
            print("Step:", stepCounter, "| Avg. epoch loss:", running_loss)
            losses = np.array(losses)
            print("avg. loss: "+str(np.mean(losses))+"+-" +
                str(np.std(losses)/np.sqrt(losses.size)))
            
            # Average loss for the current iteration
            avg_train_loss = np.mean(losses)
            trainHist.append(avg_train_loss)
            losses = []

            torch.save(net, os.path.join(config['LAST_NET_PATH'], 'last.net'))

            valCounter = 0
            net.train(False)
            losses = []
            for _ in range(valSize):
                outputs, labels, masks, valCounter = net.training_predict(
                    data_val, data_val_gt, valCounter, size, bs)
                val_loss = net.loss_function(outputs, labels, masks)
                losses.append(val_loss.item())
            net.train(True)
            avg_val_loss = np.mean(losses)
            if len(valHist) == 0 or avg_val_loss < np.min(valHist):
                torch.save(net, os.path.join(config['BEST_NET_PATH'], 'best.net'))
            valHist.append(avg_val_loss)

            epoch = (stepCounter / stepsPerEpoch)

            np.save(os.path.join(config['HISTORY_PATH'], 'history.npy'),
                    (np.array([np.arange(epoch), trainHist, valHist])))

            scheduler.step(avg_val_loss)

            if write_tensorboard_data:
                writer.add_scalar('train_loss', avg_train_loss, step)
                writer.add_scalar('val_loss', avg_val_loss, step)
                # Produce example image
                samples = (outputs).permute(1, 0, 2, 3)
                means = samples[0, ...]
                # Get data from GPU
                means = means.cpu().detach().numpy()
                writer.add_image(means, step)
                for name, param in net.named_parameters():
                    writer.add_histogram(name, param.clone().cpu().data.numpy(), step)

            if stepCounter / stepsPerEpoch > 200:
                break

    if write_tensorboard_data:
        writer.add_graph(net)
        writer.close()
    print('Finished Training')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", help="Path to the config.")
    args = parser.parse_args()
    config = util.load_config(args.config)
    main(config)