import torch
import torchvision
import torch.distributions as tdist
import torch.optim as optim
import numpy as np
import os
import matplotlib.pyplot as plt
import importlib

import util
from models import abstract_trainer
from models.average import weight_network

class Trainer(abstract_trainer.AbstractTrainer):

    def _load_network(self):
        if self.config['WEIGHT_MODE'] == 'image':
            Network = weight_network.ImageWeightUNet
        elif self.config['WEIGHT_MODE'] == 'pixel':
            Network = weight_network.PixelWeightUNet
        else:
            raise 'Invalid config value for \"weight_mode\".'

        self.net = Network(self.config['NUM_CLASSES'], self.loader.mean(),
                    self.loader.std(), depth=self.config['DEPTH'],
                    main_net_depth=self.config['MAIN_NET_DEPTH'],
                    sub_net_depth=self.config['SUB_NET_DEPTH'],
                    num_subnets=self.config['NUM_SUBNETS'],
                    augment_data=self.config['AUGMENT_DATA'])
                        
        #TODO load pre-trained weights of network, if available

    def _load_network_weights(self):
        #TODO only for now
        pass

    def _create_checkpoint(self):
        return {'model_state_dict': self.net.state_dict(),
                'optimizier_state_dict': self.optimizer.state_dict(),
                'epoch': self.epoch,
                'mean': self.loader.mean(),
                'std': self.loader.std(),
                'train_loss': self.train_loss,
                'val_loss': self.val_loss}

    def _write_tensorboard_data(self):
        self.writer.add_scalar('train_loss',self. avg_train_loss, self.print_step)
        self.writer.add_scalar('val_loss', self.avg_val_loss, self.print_step)

        self.net.train(False)
        # Predict for one example image
        raw = self.data_raw[0]
        prediction, _ = self.net.predict(raw, self.ps, self.overlap)
        self.net.train(True)
        gt = self.data_gt[0]
        gt = util.denormalize(gt, self.loader.mean(), self.loader.std())
        psnr = util.PSNR(gt, prediction, 255)
        self.writer.add_scalar('psnr', psnr, self.print_step)

        prediction = prediction.astype(np.uint8)
        self.writer.add_image('pred', prediction, self.print_step, dataformats='HW')

        for name, param in self.net.named_parameters():
            self.writer.add_histogram(
                name, param.clone().cpu().data.numpy(), self.print_step)

    def _perform_validation(self):
        for _ in range(self.val_size):
            #TODO just a temporary fix to handle the weights returned by the average network
            # need to fix this properly
            outputs, _, labels, masks, self.val_counter = self.net.training_predict(
                    self.data_val, self.data_val_gt, self.val_counter, 
                    self.size, self.box_size, self.bs)
            # Needed by subclasses that's why we store val_loss on self
            self.val_loss = self.net.loss_function(outputs, labels, masks)
            self.val_losses.append(self.val_loss.item())

    def _perform_epochs(self):
        # loop over the dataset multiple times
        for step in range(self.epochs):
            self.train_losses = []
            self.optimizer.zero_grad()

            # Iterate over virtual batch
            for _ in range(self.vbatch):
                outputs, _, labels, masks, self.data_counter = self.net.training_predict(
                    self.data_train, self.data_train_gt, self.data_counter, 
                    self.size, self.box_size, self.bs)

                self.train_loss = self.net.loss_function(outputs, labels, masks)
                self.train_loss.backward()
                self.running_loss += self.train_loss.item()
                self.train_losses.append(self.train_loss.item())

            self.optimizer.step()
            if step % self.steps_per_epoch == self.steps_per_epoch-1:
                self.epoch = step / self.steps_per_epoch
                self._on_epoch_end(step, self.train_losses)

                #if step / self.steps_per_epoch > 200:
                    #break
