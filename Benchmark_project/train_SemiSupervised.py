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
import source.PL_wrapper as PL_wrapper

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import sys
import argparse
from tqdm import tqdm
import random
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr, spearmanr

"""
Summary: train model session
"""

def get_args() -> any:

    # write output path name
    parser = argparse.ArgumentParser()
    
    # path varibles
    parser.add_argument('--data_path', default='./data/*.csv')
    parser.add_argument('--train_path', default='./data/*.csv')
    parser.add_argument('--valid_path', default='./data/*.csv')
    parser.add_argument('--test_path', default='./data/*.csv')
     
    parser.add_argument('--output_results_path', default='./outputs/benchmark_task/final_model/*.csv')
    parser.add_argument('--output_model_path', default='./outputs/benchmark_task/final_model/*.pth')
    parser.add_argument('--output_folder_path', default='./outputs/benchmark_task/*.pth')

    parser.add_argument('--protein', default='AAV')

    # model training variables
    parser.add_argument('--SEED', default=42, type=int, help='Random seed')
    parser.add_argument('--batch_size', default=512, type=int, help='Size of the batch.')
    parser.add_argument('--epochs', default=1000, type=int, help='Number of epochs')
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--DEVICE', default='cuda', help='Learning rate')
    parser.add_argument('--split_option', default=0, type=int, help='Choose whether to split into train/valid sets')
   
    # general architecture variables
    parser.add_argument('--z_dim', default=6, type=int, help='Latent space size')
    parser.add_argument('--num_classes', default=2, type=int, help='functional/nonfunctional labels')
    parser.add_argument('--aa_labels', default=21, type=int, help='AA plus pad gap (20+1) labels')
    
    # encoder hyperparameters
    parser.add_argument('--encoder_rates', default=5, type=int, help='dilation convolution depth')
    parser.add_argument('--C_in', default=21, type=int, help='input feature depth')
    parser.add_argument('--C_out', default=256, type=int, help='output feature depth')
    parser.add_argument('--alpha', default=0.1, type=float, help='leaky Relu hyperparameter (optional)')
    parser.add_argument('--enc_kernel', default=3, type=int, help='kernel filter size')
    parser.add_argument('--num_fc', default=1, type=int, help='number of fully connect layers')
      
    # top model (discriminative decoder) hyperparameters
    parser.add_argument('--disc_num_layers', default=2, type=int, help='depth of the discrim. top model')
    parser.add_argument('--hidden_width', default=10, type=int, help='width of top model')
    parser.add_argument('--p', default=0.3, type=float, help='top model dropout')

    # decoder wavenet hyperparameters
    parser.add_argument('--wave_hidden_state', default=256, type=int, help='no. filters for the dilated convolutions')
    parser.add_argument('--head_hidden_state', default=128, type=int, help='no. filters for the WaveNets top model')
    parser.add_argument('--num_dil_rates', default=8, type=int, help='depth of the WaveNet')
    parser.add_argument('--dec_kernel_size', default=3, type=int, help='WaveNet kernel size')

    # loss prefactor weights
    parser.add_argument('--nll_weight', default=1., type=float, help='NLL prefactor weight')
    parser.add_argument('--MI_weight', default=0.95, type=float, help='MI prefactor weight')
    parser.add_argument('--lambda_weight', default=2., type=float, help='MMD prefactor weight')
    parser.add_argument('--gamma_weight', default=1., type=float, help='discriminative prefactor weight')

    args = parser.parse_args()
    
    return args


def set_GPU() -> None:

    if torch.cuda.is_available():
        print('GPU available')
    else:
        print('Please enable GPU or use CUP')
        quit()

    USE_CUDA = True
    DEVICE = 'cuda' if USE_CUDA else 'cpu'
    return

def set_SEED(args: any) -> None:
    seed_everything(args.SEED, workers = True)
    return 

class prepare_data:


    def __init__(self, args: any):

        self.args = args


    def get_AAV(self, ) -> (
            DataLoader,
            DataLoader,
            DataLoader,
            int
        ):
        
        # create data
        train_num, train_OH, train_pheno, valid_num, valid_OH, valid_pheno, test_num, test_OH, test_pheno, max_seq_len = prep.prepare_AAV_datasets(
                df_path=self.args.data_path,
                split_option=self.args.split_option
        )
     
        ###########################
        ## Prepare Train dataset ##
        ###########################
         
        train_dataset = prep.AAV_dataset(
                num_inputs=train_num,
                onehot_inputs=train_OH,
                pheno_outputs=train_pheno
        )

        train_dataloader = DataLoader(train_dataset, batch_size=self.args.batch_size, num_workers=4, shuffle=True)

        ###########################
        ## Prepare Valid dataset ##
        ###########################

        valid_dataset = prep.AAV_dataset(
                num_inputs=valid_num,
                onehot_inputs=valid_OH,
                pheno_outputs=valid_pheno
        )

        valid_dataloader = DataLoader(valid_dataset, batch_size=self.args.batch_size, num_workers=4, shuffle=False)
        
        ###########################
        ## Prepare Test dataset ##
        ###########################

 
        test_dataset = prep.AAV_dataset(
                num_inputs=test_num,
                onehot_inputs=test_OH,
                pheno_outputs=test_pheno
        )

        test_dataloader = DataLoader(test_dataset, batch_size=self.args.batch_size, num_workers=4, shuffle=False)
               
        max_seq_len = train_num.shape[-1]
     
        return (
                train_dataloader,
                valid_dataloader,
                test_dataloader,
                max_seq_len
        )



    def get_GB1(self, ) -> (
            DataLoader,
            DataLoader,
            DataLoader,
            int
        ):

        # create data
        train_num, train_OH, train_pheno, valid_num, valid_OH, valid_pheno, test_num, test_OH, test_pheno, max_seq_len = prep.prepare_GB1_datasets(
                GB1_path=self.args.data_path,
                split_option=self.args.split_option
        )
     
        ###########################
        ## Prepare Train dataset ##
        ###########################
         
        train_dataset = prep.GB1_dataset(
                num_inputs=train_num,
                onehot_inputs=train_OH,
                pheno_outputs=train_pheno
        )

        train_dataloader = DataLoader(train_dataset, batch_size=self.args.batch_size, num_workers=4, shuffle=True)

        ###########################
        ## Prepare Valid dataset ##
        ###########################

        valid_dataset = prep.GB1_dataset(
                num_inputs=valid_num,
                onehot_inputs=valid_OH,
                pheno_outputs=valid_pheno
        )

        valid_dataloader = DataLoader(valid_dataset, batch_size=self.args.batch_size, num_workers=4, shuffle=False)
        
        ###########################
        ## Prepare Test dataset ##
        ###########################

 
        test_dataset = prep.GB1_dataset(
                num_inputs=test_num,
                onehot_inputs=test_OH,
                pheno_outputs=test_pheno
        )

        test_dataloader = DataLoader(test_dataset, batch_size=self.args.batch_size, num_workers=4, shuffle=False)
               
        max_seq_len = train_num.shape[-1]
     
        return (
                train_dataloader,
                valid_dataloader,
                test_dataloader,
                max_seq_len
        )




    def get_GFP(self, ) -> (
            DataLoader,
            DataLoader,
            DataLoader,
            int
        ):
        
        

        # create data
        train_num, train_OH, train_pheno, valid_num, valid_OH, valid_pheno, test_num, test_OH, test_pheno, max_seq_len = prep.prepare_GFP_datasets(
                train_path=self.args.train_path,
                valid_path=self.args.valid_path,
                test_path=self.args.test_path
        )
     
        ###########################
        ## Prepare Train dataset ##
        ###########################
         
        train_dataset = prep.GFP_dataset(
                num_inputs=train_num,
                onehot_inputs=train_OH,
                pheno_outputs=train_pheno
        )

        train_dataloader = DataLoader(train_dataset, batch_size=self.args.batch_size, num_workers=4, shuffle=True)

        ###########################
        ## Prepare Valid dataset ##
        ###########################

        valid_dataset = prep.GFP_dataset(
                num_inputs=valid_num,
                onehot_inputs=valid_OH,
                pheno_outputs=valid_pheno
        )

        valid_dataloader = DataLoader(valid_dataset, batch_size=self.args.batch_size, num_workers=4, shuffle=False)
        
        ###########################
        ## Prepare Test dataset ##
        ###########################

 
        test_dataset = prep.GFP_dataset(
                num_inputs=test_num,
                onehot_inputs=test_OH,
                pheno_outputs=test_pheno
        )

        test_dataloader = DataLoader(test_dataset, batch_size=self.args.batch_size, num_workers=4, shuffle=False)
               
        max_seq_len = train_num.shape[-1]
        
        print('Max sequence length:', max_seq_len)
        return (
                train_dataloader,
                valid_dataloader,
                test_dataloader,
                max_seq_len
        )

    def get_stability(self,) -> (
            DataLoader,
            DataLoader,
            DataLoader,
            int):

        # create data
        train_num, train_OH, train_pheno, valid_num, valid_OH, valid_pheno, test_num, test_OH, test_pheno, max_seq_len = prep.prepare_stability_datasets(
                train_path=self.args.train_path,
                valid_path=self.args.valid_path,
                test_path=self.args.test_path
        )
        

        ###########################
        ## Prepare Train dataset ##
        ###########################
         
        train_dataset = prep.stability_dataset(
                num_inputs=train_num,
                onehot_inputs=train_OH,
                pheno_outputs=train_pheno
        )

        train_dataloader = DataLoader(train_dataset, batch_size=self.args.batch_size, num_workers=4, shuffle=True)

        ###########################
        ## Prepare Valid dataset ##
        ###########################

        valid_dataset = prep.stability_dataset(
                num_inputs=valid_num,
                onehot_inputs=valid_OH,
                pheno_outputs=valid_pheno
        )

        valid_dataloader = DataLoader(valid_dataset, batch_size=self.args.batch_size, num_workers=4, shuffle=False)
        
        ###########################
        ## Prepare Test dataset ##
        ###########################

 
        test_dataset = prep.stability_dataset(
                num_inputs=test_num,
                onehot_inputs=test_OH,
                pheno_outputs=test_pheno
        )

        test_dataloader = DataLoader(test_dataset, batch_size=self.args.batch_size, num_workers=4, shuffle=False)
               
        max_seq_len = train_num.shape[-1]
     
        return (
                train_dataloader,
                valid_dataloader,
                test_dataloader,
                max_seq_len
        )



    
def get_data(args: any) -> (
        DataLoader,
        DataLoader,
        DataLoader,
        int
    ):

    benchmark_datasets = prepare_data(args=args)

    if args.protein == 'AAV':
        train_dataloader, valid_dataloader, test_dataloader, max_seq_len = benchmark_datasets.get_AAV()
    
    elif args.protein == 'GB1':
        train_dataloader, valid_dataloader, test_dataloader, max_seq_len = benchmark_datasets.get_GB1()

    elif args.protein == 'GFP':
        train_dataloader, valid_dataloader, test_dataloader, max_seq_len = benchmark_datasets.get_GFP()
    
    elif args.protein == 'stability':
        train_dataloader, valid_dataloader, test_dataloader, max_seq_len = benchmark_datasets.get_stability()

    return (
            train_dataloader,
            valid_dataloader,
            test_dataloader,
            max_seq_len
    )




def get_model(
        args: any,
        protein_len: int
    ) -> any:

        

    # define inference model:
    encoder = model_comps.GatedCNN_encoder(
                                  protein_len=protein_len,
                                  class_labels=args.aa_labels,
                                  z_dim=args.z_dim,
                                  num_rates=args.encoder_rates,
                                  C_in=args.C_in,
                                  C_out=args.C_out,
                                  alpha=args.alpha,
                                  kernel=args.enc_kernel,
                                  num_fc=args.num_fc,
    )
    # define regression model:
    decoder_re = model_comps.Decoder_re(
                                    num_layers=args.disc_num_layers,
                                    hidden_width=args.hidden_width,
                                    z_dim=args.z_dim,
                                    num_classes=args.num_classes,
                                    p=args.p
    )
  
    # define generator model:
    decoder_wave = wavenet.Wave_generator(
                                protein_len=protein_len,
                                class_labels=args.aa_labels,
                                DEVICE=args.DEVICE,
                                wave_hidden_state=args.wave_hidden_state,
                                head_hidden_state=args.head_hidden_state,
                                num_dil_rates=args.num_dil_rates,
                                kernel_size=args.dec_kernel_size
    )

    # latent global conditioning
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
    
    # pytorch model
    PL_model = PL_wrapper.Lit_SSInfoVAE(
                            DEVICE=args.DEVICE,
                            SS_InfoVAE=SS_model,
                            xi_weight=args.nll_weight,
                            alpha_weight=args.MI_weight,
                            lambda_weight=args.lambda_weight,
                            gamma_weight=args.gamma_weight,
                            lr=args.lr,
                            z_dim=args.z_dim
    )

    return PL_model




def train_model(
        args: any,
        PL_model: pl.LightningModule,
        train_dataloader: DataLoader,
        valid_dataloader: DataLoader
    ) -> (
            any,
            pd.Series
    ):
        
        trainer = pl.Trainer(
         logger=False,
         callbacks=None,
         max_epochs=args.epochs,
         gpus = 1 if torch.cuda.is_available() else None,
         )      
        
        trainer.fit(PL_model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)  
        
        # training metrics
        train_L = trainer.callback_metrics['L_train_epoch'].item()
        train_NLL = trainer.callback_metrics['L_nll_train_epoch'].item()
        train_kld = trainer.callback_metrics['L_kld_train_epoch'].item()
        train_mmd = trainer.callback_metrics['L_mmd_train_epoch'].item()
        train_pheno = trainer.callback_metrics['L_pheno_train_epoch'].item()
        train_MSE_epoch = trainer.callback_metrics['Train_MSE_epoch'].item()
        train_pearson_epoch = trainer.callback_metrics['Train_pearson_R_epoch'].item()
        train_spearman_epoch = trainer.callback_metrics['Train_spearman_rho_epoch'].item()
     
        # validation metrics
        val_L = trainer.callback_metrics['L_valid_epoch'].item()
        val_NLL = trainer.callback_metrics['L_nll_valid_epoch'].item()
        val_kld = trainer.callback_metrics['L_kld_valid_epoch'].item()
        val_mmd = trainer.callback_metrics['L_mmd_valid_epoch'].item()
        val_pheno = trainer.callback_metrics['L_pheno_valid_epoch'].item()
        val_MSE_epoch = trainer.callback_metrics['val_MSE_epoch'].item()
        val_pearson_epoch = trainer.callback_metrics['val_pearson_R_epoch'].item()
        val_spearman_epoch = trainer.callback_metrics['val_spearman_rho_epoch'].item()
     
       


        final_epoch_results = [
            train_L,
            train_NLL,
            train_kld,
            train_mmd,
            train_pheno,
            train_MSE_epoch,
            train_pearson_epoch,
            train_spearman_epoch,
            val_L,
            val_NLL,
            val_kld,
            val_mmd,
            val_pheno,
            val_MSE_epoch,
            val_pearson_epoch,
            val_spearman_epoch
        ]
        
        all_epochs_losses = {
            'train_L': list(),
            'train_nll': list(),
            'train_kld': list(),
            'train_mmd': list(),
            'train_pheno': list(),
            'train_MSE': list(),
            'train_pearson': list(),
            'train_spearman': list(),
            'val_L': list(),
            'val_nll': list(),
            'val_kld': list(),
            'val_mmd': list(),
            'val_pheno': list(),
            'val_MSE': list(),
            'val_pearson': list(),
            'val_spearman': list()
        }
            
        # allocate losses from all of the epochs:
        # training metrics
        all_epochs_losses['train_L'] = [float('nan')] + PL_model.L_train_list
        all_epochs_losses['train_nll'] = [float('nan')] + PL_model.L_train_nll_list
        all_epochs_losses['train_kld'] = [float('nan')] + PL_model.L_train_kld_list
        all_epochs_losses['train_mmd'] = [float('nan')] + PL_model.L_train_mmd_list
        all_epochs_losses['train_pheno'] = [float('nan')] + PL_model.L_train_pheno_list
        all_epochs_losses['train_MSE'] = [float('nan')] + PL_model.train_MSE_list
        all_epochs_losses['train_pearson'] = [float('nan')] + PL_model.train_pearson_list
        all_epochs_losses['train_spearman'] = [float('nan')] + PL_model.train_spearman_list
        # validation metrics
        all_epochs_losses['val_L'] = PL_model.L_val_list
        all_epochs_losses['val_nll'] = PL_model.L_val_nll_list
        all_epochs_losses['val_kld'] = PL_model.L_val_kld_list
        all_epochs_losses['val_mmd'] = PL_model.L_val_mmd_list
        all_epochs_losses['val_pheno'] = PL_model.L_val_pheno_list
        all_epochs_losses['val_MSE'] = PL_model.val_MSE_list
        all_epochs_losses['val_pearson'] = PL_model.val_pearson_list
        all_epochs_losses['val_spearman'] = PL_model.val_spearman_list


        return (
                trainer,
                PL_model,
                final_epoch_results,
                all_epochs_losses
        )


@torch.no_grad()
def test_model(
        args: any,
        trainer: any,
        PL_model: pl.LightningModule,
        test_dataloader: DataLoader
        ) -> (
                None
    ): 


    trainer.test(dataloaders=test_dataloader)    

    test_MSE_epoch = trainer.callback_metrics['test_MSE_epoch'].item()
    test_pearson_epoch = trainer.callback_metrics['test_pearson_R_epoch'].item()
    test_spearman_epoch = trainer.callback_metrics['test_spearman_rho_epoch'].item()
     

    # create test dataframe
    test_dict = {}

    if isinstance(test_MSE_epoch, list):
        test_dict['test_MSE'] = test_MSE_epoch
        test_dict['test_pearson'] = test_pearson_epoch
        test_dict['test_spearman'] = test_spearman_epoch
    else:
        test_dict['test_MSE'] = [test_MSE_epoch]
        test_dict['test_pearson'] = [test_pearson_epoch]
        test_dict['test_spearman'] = [test_spearman_epoch]
    
    test_df = pd.DataFrame(test_dict)
    
    return test_df


def save_results(
        args: any,
        PL_model: any,
        final_epoch_results: list,
        all_epochs_losses: dict,
        test_df: pd.Series
    ) -> None:
    
    # save results
    # final epoch losses
    final_epoch_columns = [
                    'train_L',
                    'train_NLL',
                    'train_kld',
                    'train_mmd',
                    'train_pheno',
                    'train_MSE',
                    'train_pearson',
                    'train_spearman',
                    'val_L',
                    'val_NLL',
                    'val_kld',
                    'val_mmd',
                    'val_pheno',
                    'val_MSE',
                    'val_pearson',
                    'val_spearman',
    ]

    final_epoch_dict = dict(map(lambda column, data : (column, [data]), final_epoch_columns, final_epoch_results))
    final_epoch_df = pd.DataFrame(final_epoch_dict)
    
    final_epoch_df.to_csv(args.output_results_path.replace('.csv', '_history.csv'), index = False)
    
    # save all epoch results
    all_epochs_df = pd.DataFrame(all_epochs_losses)
    all_epochs_df.to_csv(args.output_results_path.replace('.csv', '_history_all.csv'), index = False)
    
    # save model
    torch.save(PL_model.model.state_dict(), args.output_model_path)

    # testing dataset
    test_df.to_csv(args.output_results_path.replace('.csv', '_testset.csv'), index = False)

    return



if __name__ == '__main__':


    # inpurt parameters, variables, and paths
    args = get_args()
    # make output folder directory
    os.makedirs(args.output_folder_path, exist_ok=True)
    # set GPU
    set_GPU()
    # set seed for reproducibility 
    set_SEED(args=args)
    # acquire data
    train_dataloader, valid_dataloader, test_dataloader, protein_len = get_data(args=args)
    # acquire model
    PL_model = get_model(
                  args=args,
                  protein_len=protein_len
    )
    print('Start training !')
    # train model
    trainer, PL_model, final_epoch_results, all_epochs_losses = train_model(
                            args=args,
                            PL_model=PL_model,
                            train_dataloader=train_dataloader,
                            valid_dataloader=valid_dataloader
    )
    print('Finished training !')
    
    test_df = test_model(
            args=args,
            trainer=trainer,
            PL_model=PL_model,
            test_dataloader=test_dataloader
    )

    # save models and spreadsheet
    save_results(
        args=args,
        PL_model=PL_model,
        final_epoch_results=final_epoch_results,
        all_epochs_losses=all_epochs_losses,
        test_df=test_df
    )
    print("Save model ...") 
