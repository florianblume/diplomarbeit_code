import torch
import torchvision
import torch.distributions as tdist
import torch.optim as optim
import numpy as np
import os
import matplotlib.pyplot as plt
import importlib

import util
import abstract_trainer
from . import weight_network

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
        prediction, weights = self.net.predict(raw, self.ps, self.overlap)
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

    def _train(self):
        sub_outputs, weights, labels, masks, self.dataCounter =\
            self.net.training_predict(
                self.data_train, self.data_train_gt, self.dataCounter, 
                self.size, self.box_size, self.bs)

        self.train_loss = self.net.loss_function(sub_outputs, labels, masks)
        self.train_loss.backward()
        self.running_loss += self.train_loss.item()
        self.train_losses.append(self.train_loss.item())
