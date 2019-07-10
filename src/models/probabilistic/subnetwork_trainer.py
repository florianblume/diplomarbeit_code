import torch
import torchvision
import torch.distributions as tdist
import torch.optim as optim
import numpy as np
import os
import matplotlib.pyplot as plt
import importlib

from . import subnetwork_standalone as subnetwork
import trainer
import util

class Trainer(trainer.Trainer):

    def _load_network(self):
        # Device gets automatically created in constructor
        # We persist mean and std when saving the network
        self.net = subnetwork.SubUNet(self.config['NUM_CLASSES'], self.loader.mean(),
                        self.loader.std(), depth=self.config['DEPTH'])

    def _create_checkpoint(self):
        return {'model_state_dict': self.net.state_dict(),
                'optimizier_state_dict': self.optimizer.state_dict(),
                'epoch': self.epoch,
                'mean': self.loader.mean(),
                'std': self.loader.std(),
                'train_loss': self.train_loss,
                'val_loss': self.val_loss}

    def _write_tensorboard_data(self):
        self.writer.add_scalar('train_loss', self.avg_train_loss, self.print_step)
        self.writer.add_scalar('val_loss', self.avg_val_loss, self.print_step)

        self.net.train(False)
        # Predict for one example image
        raw = self.data_raw[0]
        prediction = self.net.predict(raw, self.ps, self.overlap)
        self.net.train(True)
        gt = self.data_gt[0]
        gt = util.denormalize(gt, self.loader.mean(), self.loader.std())
        psnr = util.PSNR(gt, prediction, 255)
        self.writer.add_scalar('psnr', psnr, self.print_step)

        prediction = prediction.astype(np.uint8)
        self.writer.add_image('pred', prediction, self.print_step, dataformats='HW')

        self.writer.add_image('std', self.std.astype(np.uint8), self.print_step, dataformats='HW')

        for name, param in self.net.named_parameters():
            self.writer.add_histogram(
                name, param.clone().cpu().data.numpy(), self.print_step)

    def _train(self):
        mean, self.std, labels, masks, self.dataCounter = self.net.training_predict(
            self.data_train, self.data_train_gt, self.dataCounter, 
            self.size, self.box_size, self.bs)

        self.train_loss = self.net.loss_function(mean, labels, masks)
        self.train_loss.backward()
        self.running_loss += self.train_loss.item()
        self.losses.append(self.train_loss.item())
