import torch
import torch.distributions as tdist
import torch.optim as optim
import numpy as np
import os
import argparse

import network
import dataloader
import util

def main(network_config, data_config):
    # data_c=np.concatenate((data.copy(),dataTest.copy()))
    loader = dataloader.DataLoader(data_config['DATA_BASE_PATH'])
    data_raw, data_gt = loader.load_training_data(data_config['DATA_RAW_PATH'], data_config['DATA_GT_PATH'])
    data_raw, data_gt = util.jointShuffle(data_raw, data_gt)

    # my_train_data=data_c.copy()
    # my_val_data=data_c.copy()

    # my_train_dataGT=dataGT_c.copy()
    # my_val_dataGT=dataGT_c.copy()

    # If loaded, the network is trained using clean targets, otherwise it performs N2V
    data_train_gt = None
    data_val_gt = None

    data_train = data_raw.copy()
    data_val = data_raw.copy()

    #device = torch.device("cpu")
    # Device gets automatically created in constructor
    net = network.UNet(1, 0, 0, depth=network_config['DEPTH'])

    net.train(True)
    bs = network_config['BATCH_SIZE']
    size = network_config['PATCH_SIZE']
    num_pix = size * size / 32.0
    dataCounter = None
    box_size = np.round(np.sqrt(size * size / num_pix)).astype(np.int)

    vbatch = network_config['VIRTUAL_BATCH_SIZE']  # Virtual batch size
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=10, factor=0.5, verbose=True)

    running_loss = 0.0
    stepCounter = 0

    valSize = network_config['VALIDATION_SIZE']
    stepsPerEpoch = network_config['STEPS_PER_EPOCH']
    trainHist = []
    valHist = []

    for step in range(network_config['EPOCHS']):  # loop over the dataset multiple times
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

            loss = net.loss_function(outputs, labels, masks)
            loss.backward()
            running_loss += loss.item()
            losses.append(loss.item())

        optimizer.step()

        if stepCounter % stepsPerEpoch == stepsPerEpoch-1:
            running_loss = (np.mean(losses))
            print("Step:", stepCounter, "| Avg. epoch loss:", running_loss)
            losses = np.array(losses)
            print("avg. loss: "+str(np.mean(losses))+"+-" +
                str(np.std(losses)/np.sqrt(losses.size)))
            trainHist.append(np.mean(losses))
            losses = []

            torch.save(net, os.path.join(data_config['LAST_NET_PATH'], 'last.net'))

            valCounter = 0
            net.train(False)
            losses = []
            for i in range(valSize):
                outputs, labels, masks, valCounter = net.training_predict(
                    data_val, data_val_gt, valCounter, size, bs)
                loss = net.loss_function(outputs, labels, masks)
                losses.append(loss.item())
            net.train(True)
            avgValLoss = np.mean(losses)
            if len(valHist) == 0 or avgValLoss < np.min(valHist):
                torch.save(net, os.path.join(data_config['BEST_NET_PATH'], 'best.net'))
            valHist.append(avgValLoss)

            epoch = (stepCounter / stepsPerEpoch)

            np.save(os.path.join(data_config['HISTORY_PATH'], 'history.npy'),
                    (np.array([np.arange(epoch), trainHist, valHist])))

            scheduler.step(avgValLoss)

            if stepCounter / stepsPerEpoch > 200:
                break

    print('Finished Training')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--network_config", "-nc", help="Path to the network config.")
    parser.add_argument("--data_config", "-dc", help="Path to the data config.")
    args = parser.parse_args()
    main(args.network_config, args.data_config)