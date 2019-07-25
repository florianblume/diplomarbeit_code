import torch
import numpy as np
import os

import util
from models import AbstractTrainer
from models.average import ImageWeightUNet
from models.average import PixelWeightUNet

class Trainer(AbstractTrainer):

    def __init__(self, config, config_path):
        super(Trainer, self).__init__(config, config_path)
        self.train_loss = 0.0
        self.train_losses = []
        self.val_loss = 0.0
        self.weight_constraint =\
                self.config.get('WEIGHT_CONSTRAINT', None)
        self.weight_constraint_lambda =\
                self.config.get('WEIGHT_CONSTRAINT_LAMBDA', 0)

    def _load_network(self):
        if self.config['WEIGHT_MODE'] == 'image':
            self.weight_mode = 'image'
            Network = ImageWeightUNet
        elif self.config['WEIGHT_MODE'] == 'pixel':
            self.weight_mode = 'pixel'
            Network = PixelWeightUNet
        else:
            raise 'Invalid config value for \"weight_mode\".'

        return Network(self.config['NUM_CLASSES'], self.loader.mean(),
                       self.loader.std(),
                       main_net_depth=self.config['MAIN_NET_DEPTH'],
                       sub_net_depth=self.config['SUB_NET_DEPTH'],
                       num_subnets=self.config['NUM_SUBNETS'],
                       augment_data=self.config['AUGMENT_DATA'])

    def _load_network_state(self):
        train_network_path = self.config.get('TRAIN_NETWORK_PATH', None)
        if train_network_path is not None:
            train_network_path = os.path.join(self.experiment_base_path,
                                              self.config.get(
                                                  'TRAIN_NETWORK_PATH', None))
            checkpoint = torch.load(train_network_path)
            self.net.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizier_state_dict'])
            self.epoch = checkpoint['epoch']
            self.mean = checkpoint['mean']
            self.std = checkpoint['std']
            self.running_loss = checkpoint['running_loss']
            self.train_loss = checkpoint['train_loss']
            self.train_hist = checkpoint['train_hist']
            self.val_loss = checkpoint['val_loss']
            self.val_hist = checkpoint['val_hist']

    def _create_checkpoint(self):
        return {'model_state_dict': self.net.state_dict(),
                'optimizier_state_dict': self.optimizer.state_dict(),
                'epoch': self.epoch,
                'mean': self.loader.mean(),
                'std': self.loader.std(),
                'running_loss': self.running_loss,
                'train_loss': self.train_loss,
                'train_hist' : self.train_hist,
                'val_loss': self.val_loss,
                'val_hist': self.val_hist}

    def _write_weights_to_tensorboard(self, weights):
        # In case that we predict the weights for the whole image we only
        # want to plot their histograms. In case that we predict the weights
        # on a per-pixel basis we store it as an image.
        for i in weights.shape[0]:
            weights_name = 'weights.subnet.{}'.format(i)
            if self.weight_mode == 'image':
                self.writer.add_histogram(weights_name,
                                          weights,
                                          self.print_step,
                                          bins='auto')
            elif self.weight_mode == 'pixel':
                # Normalize weights
                weights = weights / np.sum(weights, axis=0)
                weights = weights * 255
                weights = weights.astype(np.uint8)
                self.writer.add_image(weights_name, weights, self.print_step)
            else:
                raise ValueError('Unkown weight mode.')

    def _write_tensorboard_data(self):
        self.writer.add_scalar('train_loss', self. avg_train_loss, self.print_step)
        self.writer.add_scalar('val_loss', self.avg_val_loss, self.print_step)

        self.net.train(False)
        # Predict for one example image
        raw = self.raw_example
        # Second is the sub images
        prediction, _, weights = self.net.predict(raw, self.ps, self.overlap)
        self._write_weights_to_tensorboard(weights)
        self.net.train(True)
        gt = self.gt_example
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
            #TODO just a temporary fix to handle the weights returned by 
            # the average network need to fix this properly
            outputs, weights, labels, masks, self.val_counter = self.net.training_predict(
                    self.data_val, self.data_val_gt, self.val_counter,
                    self.size, self.box_size, self.bs)
            # Needed by subclasses that's why we store val_loss on self
            if self.weight_constraint is None or self.weight_constraint == '':
                self.val_loss =\
                    self.net.loss_function(outputs, labels, masks)
            elif self.weight_constraint == 'entropy':
                self.val_loss =\
                    self.net.loss_function_with_entropy(outputs,
                                                        labels,
                                                        masks,
                                                        weights,
                                                        self.weight_constraint_lambda)
            self.val_losses.append(self.val_loss.item())

    def _perform_epochs(self):
        # loop over the dataset multiple times
        for step in range(self.epochs):
            self.train_losses = []
            self.optimizer.zero_grad()

            # Iterate over virtual batch
            for _ in range(self.vbatch):
                outputs, weights, labels, masks, self.data_counter =\
                    self.net.training_predict(self.data_train,
                                              self.data_train_gt,
                                              self.data_counter,
                                              self.size,
                                              self.box_size,
                                              self.bs)
                if self.weight_constraint is None or self.weight_constraint == '':
                    self.train_loss =\
                        self.net.loss_function(outputs, labels, masks)
                elif self.weight_constraint == 'entropy':
                    self.train_loss =\
                        self.net.loss_function_with_entropy(outputs,
                                                            labels,
                                                            masks,
                                                            weights,
                                                            self.weight_constraint_lambda)
                self.train_loss.backward()
                self.running_loss += self.train_loss.item()
                self.train_losses.append(self.train_loss.item())

            self.optimizer.step()
            if step % self.steps_per_epoch == self.steps_per_epoch-1:
                self.epoch = step / self.steps_per_epoch
                self._on_epoch_end(step, self.train_losses)

                #if step / self.steps_per_epoch > 200:
                    #break
