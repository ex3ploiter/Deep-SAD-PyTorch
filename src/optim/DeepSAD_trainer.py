from base.base_trainer import BaseTrainer
from base.base_dataset import BaseADDataset
from base.base_net import BaseNet
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import roc_auc_score

import logging
import time
import torch
import torch.optim as optim
import numpy as np

from .Attack2 import *

from .fgsm import FGSM

class DeepSADTrainer(BaseTrainer):

    def __init__(self, c, eta: float, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 150,
                 lr_milestones: tuple = (), batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda',
                 n_jobs_dataloader: int = 0):
        super().__init__(optimizer_name, lr, n_epochs, lr_milestones, batch_size, weight_decay, device,
                         n_jobs_dataloader)

        # Deep SAD parameters
        self.c = torch.tensor(c, device=self.device) if c is not None else None
        self.eta = eta

        # Optimization parameters
        self.eps = 1e-6

        # Results
        self.train_time = None
        self.test_auc = None
        self.test_time = None
        self.test_scores = None
        
        self.test_auc_clear = None
        self.test_auc_normal = None
        self.test_auc_anomal = None
        self.test_auc_both = None        

    def train(self, dataset: BaseADDataset, net: BaseNet):
        logger = logging.getLogger()

        # Get train data loader
        train_loader, _ = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Set device for network
        net = net.to(self.device)

        # Set optimizer (Adam optimizer for now)
        optimizer = optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # Set learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)

        # Initialize hypersphere center c (if c not loaded)
        if self.c is None:
            logger.info('Initializing center c...')
            self.c = self.init_center_c(train_loader, net)
            logger.info('Center c initialized.')

        # Training
        logger.info('Starting training...')
        start_time = time.time()
        net.train()
        for epoch in range(self.n_epochs):

            scheduler.step()
            if epoch in self.lr_milestones:
                logger.info('  LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))

            epoch_loss = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            for data in train_loader:
                inputs, _, semi_targets, _ = data
                inputs, semi_targets = inputs.to(self.device), semi_targets.to(self.device)

                # Zero the network parameter gradients
                optimizer.zero_grad()

                # Update network parameters via backpropagation: forward + backward + optimize
                outputs = net(inputs)
                dist = torch.sum((outputs - self.c) ** 2, dim=1)
                losses = torch.where(semi_targets == 0, dist, self.eta * ((dist + self.eps) ** semi_targets.float()))
                loss = torch.mean(losses)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            logger.info(f'| Epoch: {epoch + 1:03}/{self.n_epochs:03} | Train Time: {epoch_train_time:.3f}s '
                        f'| Train Loss: {epoch_loss / n_batches:.6f} |')

        self.train_time = time.time() - start_time
        logger.info('Training Time: {:.3f}s'.format(self.train_time))
        logger.info('Finished training.')

        return net

    def test(self, dataset: BaseADDataset, net: BaseNet,attack_type='fgsm',epsilon=8/255,alpha=1e-2):
        logger = logging.getLogger()

        # Get test data loader
        
        
        _, test_loader = dataset.loaders(batch_size=1, num_workers=self.n_jobs_dataloader)

        # Set device for network
        net = net.to(self.device)

        # Testing
        logger.info('Starting testing...')
        epoch_loss = 0.0
        n_batches = 0
        start_time = time.time()
        idx_label_score = []
        net.eval()
        # with torch.no_grad():
        for data in test_loader:
            inputs, labels, semi_targets, idx = data

            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            semi_targets = semi_targets.to(self.device)
            idx = idx.to(self.device)
            
            loss,no_adv_scores=self.getScore(net,inputs,semi_targets)
            
            if attack_type=='fgsm':
                # adv_delta=attack_pgd(net,inputs,epsilon=1.25*epsilon,attack_iters=1,restarts=1, norm="l_inf",c=self.c)
                
                attack = FGSM(net, eps=8/255)
                adv_images = attack(inputs,semi_targets,self.c,self.eta,self.eps)
            
            if attack_type=='pgd':
                adv_delta=attack_pgd(net, inputs, epsilon=epsilon,alpha=alpha,attack_iters= 10,restarts=1, norm="l_inf",c=self.c)
            
            # inputs = inputs+adv_delta if labels==0 else inputs-adv_delta

            _,adv_scores=self.getScore(net,adv_images,semi_targets)

            # Save triples of (idx, label, score) in a list
            idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                        labels.cpu().data.numpy().tolist(),
                                        no_adv_scores.cpu().data.numpy().tolist(),
                                        adv_scores.cpu().data.numpy().tolist()))

            epoch_loss += loss.item()
            n_batches += 1

        self.test_time = time.time() - start_time
        self.test_scores = idx_label_score

        # Compute AUC
        _, labels, no_adv_scores,adv_scores = zip(*idx_label_score)
        no_adv_scores = np.array(no_adv_scores)
        adv_scores = np.array(adv_scores)
        labels = np.array(labels)
        
        
        normal_imgs_idx=np.argwhere(labels==0).flatten().tolist()
        anomal_imgs_idx=np.argwhere(labels==1).flatten().tolist() 


        self.test_auc_clear=roc_auc_score(labels, no_adv_scores)
        self.test_auc_normal=roc_auc_score(labels[normal_imgs_idx].tolist()+labels[anomal_imgs_idx].tolist(),adv_scores[normal_imgs_idx].tolist()+no_adv_scores[anomal_imgs_idx].tolist())
        self.test_auc_anomal=roc_auc_score(labels[normal_imgs_idx].tolist()+labels[anomal_imgs_idx].tolist(),no_adv_scores[normal_imgs_idx].tolist()+adv_scores[anomal_imgs_idx].tolist())
        self.test_auc_both=roc_auc_score(labels, adv_scores)          
        
        
        
        # self.test_auc = roc_auc_score(labels, scores)

        # Log results
        # logger.info('Test Loss: {:.6f}'.format(epoch_loss / n_batches))
        # logger.info('Test AUC: {:.2f}%'.format(100. * self.test_auc))
        # logger.info('Test Time: {:.3f}s'.format(self.test_time))
        logger.info('Finished testing.')

   
    def getScore(self,net,inputs,semi_targets):
        outputs = net(inputs)
        dist = torch.sum((outputs - self.c) ** 2, dim=1)
        losses = torch.where(semi_targets == 0, dist, self.eta * ((dist + self.eps) ** semi_targets.float()))
        loss = torch.mean(losses)
        scores = dist
        
        return loss,scores

   
    def init_center_c(self, train_loader: DataLoader, net: BaseNet, eps=0.1):
        """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
        n_samples = 0
        c = torch.zeros(net.rep_dim, device=self.device)

        net.eval()
        with torch.no_grad():
            for data in train_loader:
                # get the inputs of the batch
                inputs, _, _, _ = data
                inputs = inputs.to(self.device)
                outputs = net(inputs)
                n_samples += outputs.shape[0]
                c += torch.sum(outputs, dim=0)

        c /= n_samples

        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        return c