#!/usr/bin/python
# -*- coding:utf-8 -*-

import logging
import os
import time
import warnings
import torch
from torch import nn
from torch import optim
import models
import AE_Datasets
import torch.nn.functional as F


def SAEloss(recon_x, x, z):
    """
    recon_x: generating images
    x: origin images
    mu: latent mean
    logvar: latent log variance
    """
    reconstruction_function = nn.MSELoss()  # mse loss
    BCE = reconstruction_function(recon_x, x)
    pmean = 0.5
    p = F.sigmoid(z)
    p = torch.mean(p, 1)
    KLD = pmean * torch.log(pmean / p) + (1 - pmean) * torch.log((1 - pmean) / (1 - p))
    KLD = torch.sum(KLD, 0)
    return BCE + KLD


class train_utils(object):
    def __init__(self, args, save_dir):
        self.args = args
        self.save_dir = save_dir

    def setup(self):
        """
        Initialize the datasets, model, loss and optimizer
        :param args:
        :return:
        """
        args = self.args

        # Consider the gpu or cpu condition
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            logging.info('using {} gpus'.format(self.device_count))
            assert args.batch_size % self.device_count == 0, "batch size should be divided by device count"
        else:
            warnings.warn("gpu is not available")
            self.device = torch.device("cpu")
            self.device_count = 1
            logging.info('using {} cpu'.format(self.device_count))


        # Load the datasets
        if args.processing_type == 'O_A':
            from AE_Datasets.O_A import datasets
            Dataset = getattr(datasets, args.data_name)
        elif args.processing_type == 'R_A':
            from AE_Datasets.R_A import datasets
            Dataset = getattr(datasets, args.data_name)
        elif args.processing_type == 'R_NA':
            from AE_Datasets.R_NA import datasets
            Dataset = getattr(datasets, args.data_name)
        else:
            raise Exception("processing type not implement")


        self.datasets = {}

        self.datasets['train'], self.datasets['val'] = Dataset(args.data_dir, args.normlizetype).data_preprare()

        self.dataloaders = {x: torch.utils.data.DataLoader(self.datasets[x], batch_size=args.batch_size,
                                                           shuffle=(True if x == 'train' else False),
                                                           num_workers=args.num_workers,
                                                           pin_memory=(True if self.device == 'cuda' else False))
                            for x in ['train', 'val']}
        # Define the model
        fmodel=getattr(models, args.model_name)
        self.encoder = getattr(fmodel, 'encoder')(in_channel=Dataset.inputchannel, out_channel=Dataset.num_classes)

        self.decoder = getattr(fmodel, 'decoder')(in_channel=Dataset.inputchannel,
                                                                   out_channel=Dataset.num_classes)
        self.classifier = getattr(fmodel, 'classifier')(in_channel=Dataset.inputchannel,
                                                                   out_channel=Dataset.num_classes)

        # Define the optimizer
        if args.opt == 'sgd':
            self.optimizer = optim.SGD([{'params': self.encoder.parameters()}, {'params': self.decoder.parameters()}],
                                       lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        elif args.opt == 'adam':
            self.optimizer = optim.Adam([{'params': self.encoder.parameters()}, {'params': self.decoder.parameters()}],
                                        lr=args.lr, weight_decay=args.weight_decay)
        else:
            raise Exception("optimizer not implement")

        # Define the learning rate decay
        if args.lr_scheduler == 'step':
            steps = [int(step) for step in args.steps.split(',')]
            self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, steps, gamma=args.gamma)
        elif args.lr_scheduler == 'exp':
            self.lr_scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, args.gamma)
        elif args.lr_scheduler == 'stepLR':
            steps = int(args.steps)
            self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, steps, args.gamma)
        elif args.lr_scheduler == 'fix':
            self.lr_scheduler = None
        else:
            raise Exception("lr schedule not implement")

        # Define the optimizer
        if args.opt == 'sgd':
            self.optimizer1 = optim.SGD([{'params': self.encoder.parameters()}, {'params': self.classifier.parameters()}],
                                        lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        elif args.opt == 'adam':
            self.optimizer1 = optim.Adam([{'params': self.encoder.parameters()}, {'params': self.classifier.parameters()}],
                                         lr=args.lr, weight_decay=args.weight_decay)
        else:
            raise Exception("optimizer not implement")

        # Define the learning rate decay
        if args.lr_scheduler == 'step':
            steps1 = [int(step) for step in args.steps1.split(',')]
            self.lr_scheduler1 = optim.lr_scheduler.MultiStepLR(self.optimizer1, steps1, gamma=args.gamma)
        elif args.lr_scheduler == 'exp':
            self.lr_scheduler1 = optim.lr_scheduler.ExponentialLR(self.optimizer1, args.gamma)
        elif args.lr_scheduler == 'stepLR':
            steps1 = int(args.steps1)
            self.lr_scheduler1 = optim.lr_scheduler.StepLR(self.optimizer1, steps1, args.gamma)
        elif args.lr_scheduler == 'fix':
            self.lr_scheduler1 = None
        else:
            raise Exception("lr schedule not implement")

        self.start_epoch = 0
        # Invert the model and define the loss
        self.encoder.to(self.device)
        self.encoder.to(self.device)
        self.decoder.to(self.device)
        self.classifier.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.criterion1 = nn.MSELoss()


    def train(self):
        """
        Training process
        :return:
        """
        args = self.args

        step = 0
        best_acc = 0.0
        batch_count = 0
        batch_loss = 0.0
        batch_acc = 0
        step_start = time.time()

        traing_acc = []
        testing_acc = []

        traing_loss = []
        testing_loss = []

        print("Training Autoencoder with minimum loss")
        for epoch in range(args.middle_epoch):

            logging.info('-'*5 + 'Epoch {}/{}'.format(epoch, args.middle_epoch - 1) + '-'*5)
            # Update the learning rate
            if self.lr_scheduler is not None:
                self.lr_scheduler.step(epoch)
                logging.info('current lr: {}'.format(self.lr_scheduler.get_lr()))
            else:
                logging.info('current lr: {}'.format(args.lr))

            # Each epoch has a training and val phase
            for phase in ['train', 'val']:
                # Define the temp variable
                epoch_start = time.time()
                epoch_loss = 0.0

                # Set model to train mode or test mode
                if phase == 'train':
                    self.encoder.train()
                    self.decoder.train()
                else:
                    self.encoder.eval()
                    self.decoder.eval()

                for batch_idx, (inputs, labels) in enumerate(self.dataloaders[phase]):
                    inputs = inputs.to(self.device)

                    # Do the learning process, in val, we do not care about the gradient for relaxing
                    with torch.set_grad_enabled(phase == 'train'):
                    #forward
                        if args.model_name in ["Vae1d", "Vae2d"]:
                            mu, logvar = self.encoder(inputs)
                            recx = self.decoder(mu, logvar)
                            loss = self.criterion1(recx, inputs)
                        elif args.model_name in ["Sae1d", "Sae2d"]:
                            z = self.encoder(inputs)
                            recx = self.decoder(z)
                            loss = SAEloss(recx, inputs, z)
                        elif args.model_name in ["Ae1d", "Ae2d", "Dae1d", "Dae2d"]:
                            z = self.encoder(inputs)
                            recx = self.decoder(z)
                            loss = self.criterion1(recx, inputs)

                        loss_temp = loss.item() * inputs.size(0)
                        epoch_loss += loss_temp

                        # Calculate the training information
                        if phase == 'train':
                            # backward
                            self.optimizer.zero_grad()
                            loss.backward()
                            self.optimizer.step()

                            batch_loss += loss_temp
                            batch_count += inputs.size(0)

                            # Print the training information
                            if step % args.print_step == 0:
                                batch_loss = batch_loss / batch_count
                                temp_time = time.time()
                                train_time = temp_time - step_start
                                step_start = temp_time
                                batch_time = train_time / args.print_step if step != 0 else train_time
                                sample_per_sec = 1.0*batch_count/train_time
                                logging.info('Epoch: {} [{}/{}], Train Loss: {:.4f}'
                                             '{:.1f} examples/sec {:.2f} sec/batch'.format(
                                    epoch, batch_idx*len(inputs), len(self.dataloaders[phase].dataset),
                                    batch_loss, sample_per_sec, batch_time
                                ))
                                batch_loss = 0.0
                                batch_count = 0
                            step += 1


                # Print the train and val information via each epoch
                epoch_loss = epoch_loss / len(self.dataloaders[phase].dataset)
                logging.info('Epoch: {} {}-Loss: {:.4f}, Cost {:.4f} sec'.format(
                    epoch, phase, epoch_loss, time.time()-epoch_start
                ))

        for epoch1 in range(self.start_epoch, args.max_epoch):
            logging.info('-' * 5 + 'Epoch {}/{}'.format(epoch1, args.max_epoch - 1) + '-' * 5)
            # Update the learning rate
            if self.lr_scheduler1 is not None:
                self.lr_scheduler1.step(epoch1)
                logging.info('current lr: {}'.format(self.lr_scheduler1.get_lr()))
            else:
                logging.info('current lr: {}'.format(args.lr))

            # Each epoch has a training and val phase
            for phase in ['train', 'val']:
                # Define the temp variable
                epoch_start = time.time()
                epoch_acc = 0
                epoch_loss = 0.0

                # Set model to train mode or test mode
                if phase == 'train':
                    self.encoder.train()
                    self.classifier.train()
                else:
                    self.encoder.eval()
                    self.classifier.eval()
                for batch_idx, (inputs, labels) in enumerate(self.dataloaders[phase]):
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # Do the learning process, in val, we do not care about the gradient for relaxing
                    with torch.set_grad_enabled(phase == 'train'):
                        # forward
                        if args.model_name in ["Vae1d", "Vae2d"]:
                            mu, logvar = self.encoder(inputs)
                            logits = self.classifier(mu, logvar)
                            loss = self.criterion(logits, labels)
                        elif args.model_name in ["Sae1d", "Sae2d"]:
                            z = self.encoder(inputs)
                            logits = self.classifier(z)
                            loss = self.criterion(logits, labels)
                        elif args.model_name in ["Ae1d", "Ae2d", "Dae1d", "Dae2d"]:
                            z = self.encoder(inputs)
                            logits = self.classifier(z)
                            loss = self.criterion(logits, labels)

                        pred = logits.argmax(dim=1)
                        correct = torch.eq(pred, labels).float().sum().item()
                        loss_temp = loss.item() * inputs.size(0)
                        epoch_loss += loss_temp
                        epoch_acc += correct

                        # Calculate the training information
                        if phase == 'train':
                            # backward
                            self.optimizer1.zero_grad()
                            loss.backward()
                            self.optimizer1.step()

                            batch_loss += loss_temp
                            batch_acc += correct
                            batch_count += inputs.size(0)

                            # Print the training information
                            if step % args.print_step == 0:
                                batch_loss = batch_loss / batch_count
                                batch_acc = batch_acc / batch_count
                                temp_time = time.time()
                                train_time = temp_time - step_start
                                step_start = temp_time
                                batch_time = train_time / args.print_step if step != 0 else train_time
                                sample_per_sec = 1.0 * batch_count / train_time
                                logging.info('Epoch: {} [{}/{}], Train Loss: {:.4f} Train Acc: {:.4f},'
                                             '{:.1f} examples/sec {:.2f} sec/batch'.format(
                                    epoch1, batch_idx * len(inputs), len(self.dataloaders[phase].dataset),
                                    batch_loss, batch_acc, sample_per_sec, batch_time
                                ))
                                batch_acc = 0
                                batch_loss = 0.0
                                batch_count = 0
                            step += 1

                # Print the train and val information via each epoch
                epoch_loss = epoch_loss / len(self.dataloaders[phase].dataset)
                epoch_acc = epoch_acc / len(self.dataloaders[phase].dataset)

                if phase == "train":
                    traing_acc.append(epoch_acc)
                    traing_loss.append(epoch_loss)
                else:
                    testing_acc.append(epoch_acc)
                    testing_loss.append(epoch_loss)
                logging.info('Epoch: {} {}-Loss: {:.4f} {}-Acc: {:.4f}, Cost {:.4f} sec'.format(
                    epoch1, phase, epoch_loss, phase, epoch_acc, time.time() - epoch_start
                ))

                # save the model
                if phase == 'val':
                    # save the checkpoint for other learning
                    model_state_dic = self.classifier.module.state_dict() if self.device_count > 1 else self.classifier.state_dict()
                    # save the best model according to the val accuracy
                    if epoch_acc > best_acc or epoch1 > args.max_epoch-2:
                        best_acc = epoch_acc
                        logging.info("save best model epoch {}, acc {:.4f}".format(epoch1, epoch_acc))
                        torch.save(model_state_dic,
                                   os.path.join(self.save_dir, '{}-{:.4f}-best_model.pth'.format(epoch1, best_acc)))















