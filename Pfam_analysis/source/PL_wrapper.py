"""

author: Niksa Praljak
"""
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.autograd import Variable
from torch.nn import functional as F


# super Pytorch Lightning
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping

import source.preprocess as prep
import source.wavenet_decoder as wavenet
import source.model_components as model_comps

#import utils.GFP_SS_utils as GFP_utils

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import sys
import argparse
from tqdm import tqdm

from sklearn.metrics import mean_squared_error, precision_score, recall_score, f1_score
from scipy.stats import pearsonr, spearmanr



# Unsupervised InfoVAE for the wavenet generator

class Lit_InfoVAE(pl.LightningModule):
    
    def __init__(
              self,
              DEVICE: str,
              model: nn.Module,
              xi_weight: float=1.0,
              alpha_weight: float=0.95,
              lambda_weight: float=2.0,
              lr: float=1e-4,
              z_dim: int=10
        ):
        super().__init__()
        
        # device
        self.DEVICE=DEVICE

        # model    
        self.model=model
 
         # prefactor weights
        self.xi_weight=xi_weight
        self.alpha_weight=alpha_weight
        self.lambda_weight=lambda_weight
        
        # learning rate
        self.lr=lr 
        # latent space
        self.z_dim=z_dim
         
        # log containers
        self.L_train, self.L_valid = [], [] # total loss values
        self.L_nll_train, self.L_nll_valid = [], [] # nll loss values
        self.L_kld_train, self.L_kld_valid = [], [] # kld loss values
        self.L_mmd_train, self.L_mmd_valid = [], [] # mmd loss values
        
        # log container of the results at each epoch
        self.L_train_list, self.L_val_list = [], []
        self.L_train_nll_list, self.L_val_nll_list = [], []
        self.L_train_kld_list, self.L_val_kld_list = [], []
        self.L_train_mmd_list, self.L_val_mmd_list = [], []
              
        self.save_hyperparameters("lr", "z_dim", "xi_weight", "alpha_weight", "lambda_weight")
        
    def forward(self,x: torch.FloatTensor) -> (
            torch.FloatTensor,
            torch.FloatTensor,
            torch.FloatTensor,
            torch.FloatTensor,
        ):
     
        # data properties 
        batch_size, protein_len, aa_vars = x.size() 
        
        # forward pass
        logits_xrc, z_pred, z_mu, z_var = self.model(x) 
       
        return (
                logits_xrc, 
                z_pred, 
                z_mu, 
                z_var
        )
   
    def configure_optimizers(self,):
        return torch.optim.Adam(self.parameters(), lr = self.lr)

    def training_step(
            self,
            batch: torch.FloatTensor,
            batch_idx: any
        ) -> dict:
        
        x_num, x_onehot = batch

        # forward pass
        logits_xr, z_pred, z_mu, z_var = self(x_onehot)


 	# sample from some noise:
        # normal distribution:
        if self.DEVICE == 'cuda':
           z_true_samples = Variable(torch.randn((len(x_onehot), self.z_dim)).to(self.DEVICE))
        else:
           z_true_samples = torch.randn((len(x_onehot), self.z_dim))


        # compute loss
        loss_nll, loss_kld, loss_mmd = self.model.compute_loss(
                                                                    logits_xr,
                                                                    x_onehot,
                                                                    z_pred,
                                                                    z_true_samples,
                                                                    z_mu, 
                                                                    z_var
        )
        
        
        # stop track these variables 
        z_true_samples = z_true_samples.detach().cpu()

        # compute total loss
        loss = self.xi_weight * loss_nll + ( 1 - self.alpha_weight) * loss_kld + (self.alpha_weight + self.lambda_weight - 1) * loss_mmd

        # track all of loss values at each batch iteration in single epochs
        self.L_train.append(loss.item())
        self.L_nll_train.append(loss_nll.item())
        self.L_kld_train.append(loss_kld.item())
        self.L_mmd_train.append(loss_mmd.item())
        
        # for now.. only track classification predictions
        return {'loss': loss}

    def training_epoch_end(self, outputs: any):
        """
        Function recieves outputs from training_step().

        (*) outputs is a list/dict which contains the following simple example:
               [{'pred':x_pred,'target':x_target, 'loss':L}, ...]
              
        """
        
        # option 1
        # unfold the outputs and calculate the accuracy for each batch; then, we can 
        # average the mean each batch
        
       
        # compute loss value
        L = np.mean(self.L_train)
        L_nll = np.mean(self.L_nll_train)
        L_kld = np.mean(self.L_kld_train)
        L_mmd = np.mean(self.L_mmd_train)
        
        # reset lists for next epoch
        self.L_train, self.L_nll_train, self.L_kld_train, self.L_mmd_train = [], [], [], []
      
        # track the whole training history
        self.L_train_list.append(L.item())
        self.L_train_nll_list.append(L_nll.item())
        self.L_train_kld_list.append(L_kld.item())
        self.L_train_mmd_list.append(L_mmd.item())
        
        # save the metrics and losses at each epoch
        self.log('L_train_epoch', L.item(), on_epoch = True, prog_bar = True, logger = True)
        self.log('L_nll_train_epoch', L_nll.item(), on_epoch = True, prog_bar = True, logger = True)
        self.log('L_kld_train_epoch', L_kld.item(), on_epoch = True, prog_bar = True, logger = True)
        self.log('L_mmd_train_epoch', L_mmd.item(), on_epoch = True, prog_bar = True, logger = True)
        

    def validation_step(
            self,
            batch: torch.FloatTensor,
            batch_idx: any
        ) -> dict:
        
        # get data tensors from dataloader for current batch
        x_num, x_onehot = batch
           
        # forward pass
        logits_xr, z_pred, z_mu, z_var = self.forward(x_onehot)
  
       
        # sample from some noise:
        # normal distribution:

        if self.DEVICE == 'cuda':
           z_true_samples = torch.randn((len(x_onehot), self.z_dim)).to(self.DEVICE)
        else:
           z_true_samples = torch.randn((len(x_onehot), self.z_dim))

        # prepare predictions
        # note: drop any samples with ground truth r.e. scores
        # ----------------------------------------------------
      
        # loss
        loss_nll, loss_kld, loss_mmd  = self.model.compute_loss(
                                                                    xr=logits_xr,
                                                                    x=x_onehot,
                                                                    z_pred=z_pred, 
                                                                    true_samples=z_true_samples,
                                                                    z_mu=z_mu, 
                                                                    z_var=z_var
        )
        
        # stop tracking these variables
        z_true_samples = z_true_samples.detach().cpu()
 
        # compute total loss
        loss = self.xi_weight * loss_nll + ( 1 - self.alpha_weight) * loss_kld + (self.alpha_weight + self.lambda_weight - 1)*loss_mmd 
        # track all of loss values at each batch iteration in single epochs
        self.L_valid.append(loss.item())
        self.L_nll_valid.append(loss_nll.item())
        self.L_kld_valid.append(loss_kld.item())
        self.L_mmd_valid.append(loss_mmd.item())
               
        return {'val_loss': loss}

   
    def validation_epoch_end(self, outputs: any):
        """
        Function recvieves outputs from validation_step().

        (*) outputs is a list/dict which contains the following simple example:
               [{'pred':x_pred,'target':x_target, 'loss':L}, ...]
              
        """
        
        
        # option 1
        # unfold the outputs and calculate the accuracy for each batch; then, we can 
        # average the mean each batch
             
        # compute loss value
        L = np.mean(self.L_valid)
        L_nll = np.mean(self.L_nll_valid)
        L_kld = np.mean(self.L_kld_valid)
        L_mmd = np.mean(self.L_mmd_valid)
        
        # reset lists for next epoch
        self.L_valid, self.L_nll_valid, self.L_kld_valid, self.L_mmd_valid = [], [], [], []
 
        # track the whole training history
        self.L_val_list.append(L.item())
        self.L_val_nll_list.append(L_nll.item())
        self.L_val_kld_list.append(L_kld.item())
        self.L_val_mmd_list.append(L_mmd.item())
             
        self.log('L_valid_epoch', L.item(), on_epoch = True, prog_bar = True, logger = True)
        self.log('L_nll_valid_epoch', L_nll.item(), on_epoch = True, prog_bar = True, logger = True)
        self.log('L_kld_valid_epoch', L_kld.item(), on_epoch = True, prog_bar = True, logger = True)
        self.log('L_mmd_valid_epoch', L_mmd.item(), on_epoch = True, prog_bar = True, logger = True)
      
    


# Semi-supervised InfoVAE for the wavenet generator

class Lit_SSInfoVAE(pl.LightningModule):
    
    def __init__(
              self,
              DEVICE: str,
              SS_InfoVAE: any,
              xi_weight: float=1.0,
              alpha_weight: float=0.95,
              lambda_weight: float=2.0,
              gamma_weight: float=1.0,
              lr: float=1e-4,
              z_dim: int=10
        ):
        super().__init__()
        
        # device
        self.DEVICE=DEVICE

        # model    
        self.model=SS_InfoVAE
 
         # prefactor weights
        self.xi_weight=xi_weight
        self.alpha_weight=alpha_weight
        self.lambda_weight=lambda_weight
        self.gamma_weight=gamma_weight
        
        # learning rate
        self.lr=lr 
        # latent space
        self.z_dim=z_dim
         
        # log containers
        self.L_train, self.L_valid = [], [] # total loss values
        self.L_nll_train, self.L_nll_valid = [], [] # nll loss values
        self.L_kld_train, self.L_kld_valid = [], [] # kld loss values
        self.L_mmd_train, self.L_mmd_valid = [], [] # mmd loss values
        self.L_pheno_train, self.L_pheno_valid = [], [] # phenotype loss values
        
        # log container of the results at each epoch
        self.L_train_list, self.L_val_list = [], []
        self.L_train_nll_list, self.L_val_nll_list = [], []
        self.L_train_kld_list, self.L_val_kld_list = [], []
        self.L_train_mmd_list, self.L_val_mmd_list = [], []
        self.L_train_pheno_list, self.L_val_pheno_list = [], []
        # track metric performance
        self.train_precision_list, self.val_precision_list = [], []
        self.train_recall_list, self.val_recall_list = [], []
        self.train_f1_list, self.val_f1_list = [], []

        self.save_hyperparameters("lr", "z_dim", "xi_weight", "alpha_weight", "lambda_weight", "gamma_weight")
        
    def forward(self,x: torch.FloatTensor) -> (
            torch.FloatTensor,
            torch.FloatTensor,
            torch.FloatTensor,
            torch.FloatTensor,
            torch.FloatTensor,
            torch.FloatTensor
        ):
     
        # data properties 
        batch_size, protein_len, aa_vars = x.size() 
        
        # forward pass
        logits_xrc, y_pred_R, y_pred_C, z_pred, z_mu, z_var = self.model(x) 
       
        return (
                logits_xrc, 
                y_pred_R, 
                y_pred_C, 
                z_pred, 
                z_mu, 
                z_var
        )
   
    def configure_optimizers(self,):
        return torch.optim.Adam(self.parameters(), lr = self.lr)

    def compute_task_metrics(
            self,
            y_pred: torch.FloatTensor,
            y_true: torch.FloatTensor
        ) -> (
                float,
                float,
                float
        ):
        """
        using sklearn metrics, which means that we will have to convert between torch-numpy and GPU         -CPU.
        """
        # temp:  convert torch tensor to numpy
        y_pred = y_pred.cpu().detach().numpy().squeeze(1)
        y_true = y_true.cpu().detach().numpy().squeeze(1)
        
        # convert to binary predictions:
        y_pred = (y_pred > 0.5)*1.0        


        # compute metric scores ...
        precision = precision_score(y_pred, y_true )
        recall = recall_score(y_pred, y_true )
        f1 = f1_score(y_pred, y_true)

        return precision, recall, f1


    def training_step(
            self,
            batch: torch.FloatTensor,
            batch_idx: any
        ) -> dict:
        

        # return onehot encoded features, regression predictions, and accepted flow indexes
        x_onehot, y_pheno_R, y_pheno_C, accept_flow_idxes = batch
     
        # change nan to values to 0 (note: we will not account for these values in loss calculations).        
        y_pheno_C = torch.nan_to_num(y_pheno_C, nan = 0).to(torch.float32)
        y_pheno_R = torch.nan_to_num(y_pheno_R, nan = 0).to(torch.float32)

        # forward pass
        logits_xr, y_pred_R, y_pred_C, z_pred, z_mu, z_var = self(x_onehot)


 	# sample from some noise:
        # normal distribution:
        if self.DEVICE == 'cuda':
           z_true_samples = Variable(torch.randn((len(x_onehot), self.z_dim)).to(self.DEVICE))
        else:
           z_true_samples = torch.randn((len(x_onehot), self.z_dim))


        # prepare predictions
        # note: drop any samples with ground truth r.e. scores
        # ----------------------------------------------------
        # mask classification predictions 
        y_pheno_C = torch.masked_select(y_pheno_C, accept_flow_idxes == 1., out = None).unsqueeze(1)
        y_pred_C = torch.masked_select(y_pred_C, accept_flow_idxes == 1., out = None).unsqueeze(1)
        # mask regression predictions
        y_pheno_R = torch.masked_select(y_pheno_R, accept_flow_idxes == 1., out = None).unsqueeze(1)
        y_pred_R = torch.masked_select(y_pred_R, accept_flow_idxes == 1., out = None).unsqueeze(1)    


        # compute loss
        loss_nll, loss_kld, loss_mmd, loss_pheno = self.model.compute_loss(
                                                                    logits_xr,
                                                                    x_onehot,
                                                                    y_pred_R,
                                                                    y_pheno_R,
                                                                    y_pred_C,
                                                                    y_pheno_C,
                                                                    z_pred,
                                                                    z_true_samples,
                                                                    z_mu, 
                                                                    z_var
        )
        
        
        # stop track these variables 
        z_true_samples = z_true_samples.detach().cpu()

        # compute total loss
        loss = self.xi_weight * loss_nll + ( 1 - self.alpha_weight) * loss_kld + (self.alpha_weight + self.lambda_weight - 1) * loss_mmd + self.gamma_weight * loss_pheno

        # track all of loss values at each batch iteration in single epochs
        self.L_train.append(loss.item())
        self.L_nll_train.append(loss_nll.item())
        self.L_kld_train.append(loss_kld.item())
        self.L_mmd_train.append(loss_mmd.item())
        self.L_pheno_train.append(loss_pheno.item())
        
        # for now.. only track classification predictions
        return {'loss': loss,'y_pred': y_pred_C, 'y_true': y_pheno_C}

    def training_epoch_end(self, outputs: any):
        """
        Function recieves outputs from training_step().

        (*) outputs is a list/dict which contains the following simple example:
               [{'pred':x_pred,'target':x_target, 'loss':L}, ...]
              
        """
        
        # option 1
        # unfold the outputs and calculate the accuracy for each batch; then, we can 
        # average the mean each batch
        
        # initialize predicted and ground truth regression predictions
        y_pred = torch.empty((0,1))
        y_true = torch.empty((0,1))
        
        # conactenate ground truth and predcited values
        for out in outputs:
            try:
                y_pred = torch.cat((y_pred, out['y_pred'].cpu().detach()), dim = 0)
                y_true = torch.cat((y_true, out['y_true'].cpu().detach()), dim = 0)
            except RuntimeError:
                pass
         
        # -compute metrics-
        # both modes:
        try:
            precision, recall, f1 = self.compute_task_metrics(y_pred,  y_true)
            # save the metric
            self.log('Train_precision_epoch', precision.item(), on_epoch = True, prog_bar = True, logger = True)
            self.log('Train_recall_epoch', recall.item(), on_epoch = True,  prog_bar = True, logger = True)
            self.log('Train_f1_epoch', f1.item(), on_epoch = True, prog_bar = True, logger = True)
        
        except ValueError:
            # set the metrics to 0
            precision, recall, f1 = torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)
            # save the metric
            self.log('Train_precision_epoch', precision.item(), on_epoch = True, prog_bar = True, logger = True)
            self.log('Train_recall_epoch', recall.item(), on_epoch = True,  prog_bar = True, logger = True)
            self.log('Train_f1_epoch', f1.item(), on_epoch = True, prog_bar = True, logger = True)
      
        
        # compute loss value
        L = np.mean(self.L_train)
        L_nll = np.mean(self.L_nll_train)
        L_kld = np.mean(self.L_kld_train)
        L_mmd = np.mean(self.L_mmd_train)
        L_pheno = np.mean(self.L_pheno_train)
        
        # reset lists for next epoch
        self.L_train, self.L_nll_train, self.L_kld_train, self.L_mmd_train, self.L_pheno_train  = [], [], [], [], []
      
        # track the whole training history
        self.L_train_list.append(L.item())
        self.L_train_nll_list.append(L_nll.item())
        self.L_train_kld_list.append(L_kld.item())
        self.L_train_mmd_list.append(L_mmd.item())
        self.L_train_pheno_list.append(L_pheno.item())
        # track metric properties
        self.train_precision_list.append(precision.item())
        self.train_recall_list.append(recall.item())
        self.train_f1_list.append(f1.item())

        # save the metrics and losses at each epoch
        self.log('L_train_epoch', L.item(), on_epoch = True, prog_bar = True, logger = True)
        self.log('L_nll_train_epoch', L_nll.item(), on_epoch = True, prog_bar = True, logger = True)
        self.log('L_kld_train_epoch', L_kld.item(), on_epoch = True, prog_bar = True, logger = True)
        self.log('L_mmd_train_epoch', L_mmd.item(), on_epoch = True, prog_bar = True, logger = True)
        self.log('L_pheno_train_epoch', L_pheno.item(), on_epoch = True, prog_bar = True, logger = True)
        

    def validation_step(
            self,
            batch: torch.FloatTensor,
            batch_idx: any
        ) -> dict:
        
        # get data tensors from dataloader for current batch
        x_onehot, y_pheno_R, y_pheno_C, accept_flow_idxes = batch
     
        # change nan values to 0 (note: we will not account for these values in loss calculations)
        y_pheno_C = torch.nan_to_num(y_pheno_C, nan = 0).to(torch.float32) 
        y_pheno_R = torch.nan_to_num(y_pheno_R, nan = 0).to(torch.float32) 
 
        # forward pass
        logits_xr, y_pred_R, y_pred_C, z_pred, z_mu, z_var = self.forward(x_onehot)
  
       
        # sample from some noise:
        # normal distribution:

        if self.DEVICE == 'cuda':
           z_true_samples = torch.randn((len(x_onehot), self.z_dim)).to(self.DEVICE)
        else:
           z_true_samples = torch.randn((len(x_onehot), self.z_dim))

        # prepare predictions
        # note: drop any samples with ground truth r.e. scores
        # ----------------------------------------------------

        # classification:
        y_pheno_C = torch.masked_select(y_pheno_C, accept_flow_idxes == 1., out = None).unsqueeze(1)
        y_pred_C = torch.masked_select(y_pred_C, accept_flow_idxes == 1., out = None).unsqueeze(1)

        # regression:
        y_pheno_R = torch.masked_select(y_pheno_R, accept_flow_idxes == 1., out = None).unsqueeze(1)
        y_pred_R = torch.masked_select(y_pred_R, accept_flow_idxes == 1., out = None).unsqueeze(1)

      
        # loss
        loss_nll, loss_kld, loss_mmd, loss_pheno = self.model.compute_loss(
                                                                    xr=logits_xr,
                                                                    x=x_onehot,
                                                                    y_pred_R=y_pred_R,
                                                                    y_true_R=y_pheno_R,
                                                                    y_pred_C=y_pred_C, 
                                                                    y_true_C=y_pheno_C,
                                                                    z_pred=z_pred, 
                                                                    true_samples=z_true_samples,
                                                                    z_mu=z_mu, 
                                                                    z_var=z_var
        )
        
        # stop tracking these variables
        z_true_samples = z_true_samples.detach().cpu()
 
        # compute total loss
        loss = self.xi_weight * loss_nll + ( 1 - self.alpha_weight) * loss_kld + (self.alpha_weight + self.lambda_weight - 1)*loss_mmd + self.gamma_weight * loss_pheno

        # track all of loss values at each batch iteration in single epochs
        self.L_valid.append(loss.item())
        self.L_nll_valid.append(loss_nll.item())
        self.L_kld_valid.append(loss_kld.item())
        self.L_mmd_valid.append(loss_mmd.item())
        self.L_pheno_valid.append(loss_pheno.item())
               
        return {'val_loss': loss, 'val_y_pred': y_pred_C, 'val_y_true': y_pheno_C}

   
    def validation_epoch_end(self, outputs: any):
        """
        Function recvieves outputs from validation_step().

        (*) outputs is a list/dict which contains the following simple example:
               [{'pred':x_pred,'target':x_target, 'loss':L}, ...]
              
        """
        
        
        # option 1
        # unfold the outputs and calculate the accuracy for each batch; then, we can 
        # average the mean each batch
        
        # initialize predicted and ground truth regression predictions
        y_pred = torch.empty((0,1))
        y_true = torch.empty((0,1))
        
        # conactenate ground truth and predcited values
        for out in outputs:
            try:
                y_pred = torch.cat((y_pred, out['val_y_pred'].cpu().detach()), dim = 0)
                y_true = torch.cat((y_true, out['val_y_true'].cpu().detach()), dim = 0)
            except RuntimeError:
                pass
        
        # -compute metrics-
        # both modes:
        try:
            precision, recall, f1 = self.compute_task_metrics(y_pred,  y_true)
            # save the metric
            self.log('val_precision_epoch', precision.item(), on_epoch = True, prog_bar = True, logger = True)
            self.log('val_recall_epoch', recall.item(), on_epoch = True,  prog_bar = True, logger = True)
            self.log('val_f1_epoch', f1.item(), on_epoch = True, prog_bar = True, logger = True)
        
        except ValueError:
            precision, recall, f1 = torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0) 
            # save the metric
            self.log('val_precision_epoch', precision.item(), on_epoch = True, prog_bar = True, logger = True)
            self.log('val_recall_epoch', recall.item(), on_epoch = True,  prog_bar = True, logger = True)
            self.log('val_f1_epoch', f1.item(), on_epoch = True, prog_bar = True, logger = True)
 
        # compute loss value
        L = np.mean(self.L_valid)
        L_nll = np.mean(self.L_nll_valid)
        L_kld = np.mean(self.L_kld_valid)
        L_mmd = np.mean(self.L_mmd_valid)
        L_pheno = np.mean(self.L_pheno_valid)
        
        # reset lists for next epoch
        self.L_valid, self.L_nll_valid, self.L_kld_valid, self.L_mmd_valid, self.L_pheno_valid  = [], [], [], [], []
 
        # track the whole training history
        self.L_val_list.append(L.item())
        self.L_val_nll_list.append(L_nll.item())
        self.L_val_kld_list.append(L_kld.item())
        self.L_val_mmd_list.append(L_mmd.item())
        self.L_val_pheno_list.append(L_pheno.item())
        # track metric properties
        self.val_precision_list.append(precision.item())
        self.val_recall_list.append(recall.item())
        self.val_f1_list.append(f1.item())

        self.log('L_valid_epoch', L.item(), on_epoch = True, prog_bar = True, logger = True)
        self.log('L_nll_valid_epoch', L_nll.item(), on_epoch = True, prog_bar = True, logger = True)
        self.log('L_kld_valid_epoch', L_kld.item(), on_epoch = True, prog_bar = True, logger = True)
        self.log('L_mmd_valid_epoch', L_mmd.item(), on_epoch = True, prog_bar = True, logger = True)
        self.log('L_pheno_valid_epoch', L_pheno.item(), on_epoch = True, prog_bar = True, logger = True)
      
    

