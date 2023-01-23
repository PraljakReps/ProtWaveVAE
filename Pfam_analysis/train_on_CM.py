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

import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import CSVLogger

import source.preprocess as prep
import source.pfam_preprocess as pfam_prep
import source.wavenet_decoder as wavenet
import source.model_components as model_comps
import source.PL_wrapper as PL_mod
import train_on_pfam as train_sess
import utils.tools as util_tools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import sys
import argparse
from tqdm import tqdm

from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import train_test_split


def call_SS_model(
        args: any,
        protein_len: int,
    ) -> pl.LightningModule:
     
    
    
    # inference model
    encoder = model_comps.GatedCNN_encoder(
                                protein_len=protein_len,
                                class_labels=args.class_labels,
                                z_dim=args.z_dim,
                                num_rates=args.encoder_rates,
                                C_in=args.C_in,
                                C_out=args.C_out,
                                alpha=args.alpha,
                                kernel=args.enc_kernel,
                                num_fc=args.num_fc
    )
    

    # define regression model:
    decoder_re = model_comps.Decoder_re(
                                    num_layers=args.disc_num_layers,
                                    hidden_width=args.hidden_width,
                                    z_dim=args.z_dim,
                                    num_classes=args.num_classes,
                                    p=args.p
    )
  



    # generator: p(x_t|x_1, ..., x_t=1, z)
    # wave_hidden_state: WaveNet width
    # head_hidden_state: WaveNet head classifier width
    # num_dil_rates:  dilation rates
    # kernel_size: kernel size
    decoder_wave = wavenet.Wave_generator(
                                protein_len=protein_len,
                                class_labels=args.class_labels,
                                DEVICE=args.DEVICE,
                                wave_hidden_state=args.wave_hidden_state,
                                head_hidden_state=args.head_hidden_state,
                                num_dil_rates=args.num_dil_rates,
                                kernel_size=args.dec_kernel_size
    )
    
    cond_mapper = wavenet.CondNet(
            z_dim=args.z_dim,
            output_shape=(1, protein_len)
    )
   
    # define final model configuration ...
    # torch model
    SS_model = model_comps.SS_InfoVAE(
            DEVICE=args.DEVICE,
            encoder=encoder,
            decoder_recon=decoder_wave,
            cond_mapper=cond_mapper,
            decoder_pheno=decoder_re,
            z_dim=args.z_dim
    )

    # pytorch lightning model
    PL_model = PL_mod.Lit_SSInfoVAE(
            DEVICE=args.DEVICE,
            SS_InfoVAE=SS_model,
            xi_weight=args.xi_weight,
            alpha_weight=args.alpha_weight,
            lambda_weight=args.lambda_weight,
            gamma_weight=args.gamma_weight,
            lr=args.lr,
            z_dim=args.z_dim
    ).to(args.DEVICE)


    return PL_model



def get_SS_args(parser):

    # general variables
    parser.add_argument('--learning_option', dest='learning_option', default='semi-supervised', type=str)
  
    # decoder regression:
    parser.add_argument('--disc_num_layers', dest='disc_num_layers', default=2, type=int)
    parser.add_argument('--hidden_width', dest='hidden_width', default=10, type=int)
    parser.add_argument('--num_classes', dest='num_classes', default=1, type=int)
    parser.add_argument('--p', dest='p', default=0.3, type=float)
  
    # loss values
    parser.add_argument('--gamma_weight', dest='gamma_weight', default=1.0, type=float)
  
  
def train_SS_model(
        args: any,
        PL_model: pl.LightningModule,
        train_dataloader: DataLoader,
        test_dataloader: any
    ) -> (
            any, 
            pd.Series
    ):
        trainer = pl.Trainer(
                max_epochs=args.epochs,
                gpus = 1 if torch.cuda.is_available() else None,
        )
        
        if args.dataset_split == 1:
            print('\nTrain/Valid split training\n')
            trainer.fit(PL_model, train_dataloaders=train_dataloader, val_dataloaders=test_dataloader)
            
            
            # track losses
            # training: 
            train_L = trainer.callback_metrics['L_train_epoch'].item()
            train_NLL = trainer.callback_metrics['L_nll_train_epoch'].item()
            train_kld = trainer.callback_metrics['L_kld_train_epoch'].item()
            train_mmd = trainer.callback_metrics['L_mmd_train_epoch'].item()
            
            # validation
            valid_L = trainer.callback_metrics['L_valid_epoch'].item()
            valid_NLL = trainer.callback_metrics['L_nll_valid_epoch'].item()
            valid_kld = trainer.callback_metrics['L_kld_valid_epoch'].item()
            valid_mmd = trainer.callback_metrics['L_mmd_valid_epoch'].item()
         
            # track whole loss history
            # training metrics
            all_epochs_losses = {}           
        
            all_epochs_losses['train_L'] = [float('nan')] + PL_model.L_train_list
            all_epochs_losses['train_nll'] = [float('nan')] + PL_model.L_train_nll_list
            all_epochs_losses['train_kld'] = [float('nan')] + PL_model.L_train_kld_list
            all_epochs_losses['train_mmd'] = [float('nan')] + PL_model.L_train_mmd_list
            all_epochs_losses['val_L'] = PL_model.L_val_list
            all_epochs_losses['val_nll'] = PL_model.L_val_nll_list
            all_epochs_losses['val_kld'] = PL_model.L_val_kld_list
            all_epochs_losses['val_mmd'] = PL_model.L_val_mmd_list
            
            if args.learning_option == 'semi-supervised':

                train_pheno = trainer.callback_metrics['L_pheno_train_epoch'].item()
                train_MSE_epoch = trainer.callback_metrics['Train_MSE_epoch'].item()
                train_pearson_epoch = trainer.callback_metrics['Train_pearson_R_epoch'].item()
                train_spearman_epoch = trainer.callback_metrics['Train_spearman_rho_epoch'].item()
                valid_pheno = trainer.callback_metrics['L_pheno_valid_epoch'].item()
                valid_MSE_epoch = trainer.callback_metrics['val_MSE_epoch'].item()
                valid_pearson_epoch = trainer.callback_metrics['val_pearson_R_epoch'].item()
                valid_spearman_epoch = trainer.callback_metrics['val_spearman_rho_epoch'].item()
              
                all_epochs_losses['train_pheno'] = [float('nan')] + PL_model.L_train_pheno_list
                all_epochs_losses['train_MSE'] = [float('nan')] + PL_model.train_MSE_list
                all_epochs_losses['train_pearson'] = [float('nan')] + PL_model.train_pearson_list
                all_epochs_losses['train_spearman'] = [float('nan')] + PL_model.train_spearman_list
                all_epochs_losses['val_pheno'] = PL_model.L_val_pheno_list
                all_epochs_losses['val_MSE'] = PL_model.val_MSE_list
                all_epochs_losses['val_pearson'] = PL_model.val_pearson_list
                all_epochs_losses['val_spearman'] = PL_model.val_spearman_list


     
            elif args.learning_option == 'unsupervised':


                train_pheno = float('nan')
                train_MSE_epoch = float('nan')
                train_pearson_epoch = float('nan')
                train_spearman_epoch = float('nan')
                valid_pheno = float('nan')
                valid_MSE_epoch = float('nan')
                valid_pearson_epoch = float('nan')
                valid_spearman_epoch = float('nan')
      

            final_epoch_results = [
                    train_L,
                    train_NLL,
                    train_kld,
                    train_mmd,
                    train_pheno,
                    train_MSE_epoch,
                    train_pearson_epoch,
                    train_spearman_epoch,
                    valid_L,
                    valid_NLL,
                    valid_kld,
                    valid_mmd,
                    valid_pheno,
                    valid_MSE_epoch,
                    valid_pearson_epoch,
                    valid_spearman_epoch
            ]
            
    
        else:
            print('\nTrain on the whole data\n')
            trainer.fit(PL_model, train_dataloader)
            
            
            # track losses
            train_L = trainer.callback_metrics['L_train_epoch'].item()
            train_NLL = trainer.callback_metrics['L_nll_train_epoch'].item()
            train_kld = trainer.callback_metrics['L_kld_train_epoch'].item()
            train_mmd = trainer.callback_metrics['L_mmd_train_epoch'].item()
            train_pheno = trainer.callback_metrics['L_pheno_train_epoch'].item()
            train_MSE_epoch = trainer.callback_metrics['Train_MSE_epoch'].item()
            train_pearson_epoch = trainer.callback_metrics['Train_pearson_R_epoch'].item()
            train_spearman_epoch = trainer.callback_metrics['Train_spearman_rho_epoch'].item()
       


            final_epoch_results = [
                    train_L,
                    train_NLL,
                    train_kld,
                    train_mmd,
                    train_pheno,
                    train_MSE_epoch,
                    train_pearson_epoch,
                    train_spearman_epoch
            ]
            
            # track whole loss history
            all_epochs_losses = {}           
            all_epochs_losses['train_L'] = PL_model.L_train_list
            all_epochs_losses['train_nll'] = PL_model.L_train_nll_list
            all_epochs_losses['train_kld'] = PL_model.L_train_kld_list
            all_epochs_losses['train_mmd'] = PL_model.L_train_mmd_list
            all_epochs_losses['train_pheno'] = PL_model.L_train_pheno_list
            all_epochs_losses['train_MSE'] = PL_model.L_train_pheno_list
            all_epochs_losses['train_pearson'] = PL_model.L_train_pearson_list
            all_epochs_losses['train_spearman'] = PL_model.L_train_spearman_list

            
        return (
                final_epoch_results,
                all_epochs_losses
        )

def save_results(
        args: any,
        PL_model: any,
        final_epoch_results: list,
        all_epoch_losses: dict
    ) -> None:

    # save results
    # final epoch losses
    
    if args.learning_option == 'semi-supervised': # with train/val split
        
        final_epoch_columns = [
            'train_L',
            'train_nll',
            'train_kld',
            'train_mmd',
            'train_pheno',
            'train_MSE',
            'train_pearson',
            'train_spearman',
            'val_L',
            'val_nll',
            'val_kld',
            'val_mmd',
            'val_pheno',
            'val_MSE',
            'val_pearson',
            'val_spearman'
        ]

    else:

        final_epoch_columns = [
            'train_L',
            'train_nll',
            'train_kld',
            'train_mmd',
            'train_pheno',
            'train_MSE',
            'train_pearson',
            'train_spearman',
            'val_L',
            'val_nll',
            'val_kld',
            'val_mmd',
            'val_pheno',
            'val_MSE',
            'val_pearson',
            'val_spearman'
        ]
 
    final_epoch_dict = dict(map(lambda column, data : (column, [data]), final_epoch_columns, final_epoch_results))
    final_epoch_df = pd.DataFrame(final_epoch_dict)
    final_epoch_df.to_csv(args.output_results_path, index = False)

  
    # save all epoch results
    all_epochs_df = pd.DataFrame(all_epoch_losses)
    all_epochs_df.to_csv(args.output_results_path.replace('.csv', '_all.csv'), index = False)
                  
    # save model
    torch.save(PL_model.model.state_dict(), args.model_output_path)

    return

def load_CM_data(args: any) -> (
        DataLoader,
        DataLoader,
        int
    ):

    if args.learning_option == 'semi-supervised':
        unsupervised_option = False

    elif args.learning_option == 'unsupervised':
        unsupervised_option = True
    else:
        quit()
    
    # prepare nautral homologs as training data
    train_num_X, train_OH_X, train_y = pfam_prep.prepare_CM_dataset(
                 data_path = args.train_path,
                 alignment = args.alignment
    )

    # prepare design homologs as testing data
    test_num_X, test_OH_X, test_y = pfam_prep.prepare_CM_dataset(
                 data_path=args.test_path,
                 alignment=args.alignment
    )
    

    # train dataset:
    train_dataset = pfam_prep.CM_dataset(
                                    num_inputs=train_num_X,
                                    onehot_inputs=train_OH_X,
                                    pheno_outputs=train_y,
                                    unsupervised_option=unsupervised_option
    )

    train_dataloader = DataLoader(
                                    dataset = train_dataset,
                                    batch_size = args.batch_size,
                                    num_workers = 4,
                                    shuffle = True
    )

    # test dataset:
    test_dataset = pfam_prep.CM_dataset(
                                        num_inputs = test_num_X,
                                        onehot_inputs = test_OH_X,
                                        pheno_outputs = test_y,
                                        unsupervised_option=unsupervised_option
    )

    test_dataloader = DataLoader(
                                    dataset = test_dataset,
                                    batch_size = args.batch_size,
                                    num_workers = 4,
                                    shuffle = False
    )
    
    _, protein_len, _ = train_OH_X.shape

    return (
            train_dataloader,
            test_dataloader,
            protein_len
    )
            

if __name__ == '__main__':
    
    # get variable arguments
    parser = argparse.ArgumentParser()
    train_sess.get_args(parser)
    get_SS_args(parser)
 
    # only use unaligned sequences
    args = parser.parse_args()
    args.alignment = False
    # reproducibility
    train_sess.set_SEED(args=args)
    
    # set GPU (cuda)
    args.DEVICE = train_sess.set_GPU(args=args)

    # load Data
    train_dataloader, test_dataloader, protein_len = load_CM_data(args=args)
   
    # call model
    if args.learning_option == 'unsupervised':
        
        PL_model = train_sess.call_model(
                args=args,
                protein_len=protein_len
        )
    
    elif args.learning_option == 'semi-supervised':
        
        PL_model = call_SS_model(
                args=args,
                protein_len=protein_len
        )

     
    print('Train model with alignment: ', args.alignment)
    # train model
    final_epoch_results, all_epoch_results = train_SS_model(
                                                args=args,
                                                PL_model=PL_model,
                                                train_dataloader=train_dataloader,
                                                test_dataloader=test_dataloader
    )

    # save results
    save_results(
            args=args,
            PL_model=PL_model,
            final_epoch_results=final_epoch_results,
            all_epoch_losses=all_epoch_results
    )





