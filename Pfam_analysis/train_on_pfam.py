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


def call_model(
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
    model = model_comps.InfoVAE(
            DEVICE=args.DEVICE,
            encoder=encoder,
            decoder_recon=decoder_wave,
            cond_mapper=cond_mapper,
            z_dim=args.z_dim
    )

    # pytorch lightning model
    Lit_InfoVAE = PL_mod.Lit_InfoVAE(
            DEVICE=args.DEVICE,
            model=model,
            lr=args.lr,
            xi_weight=args.xi_weight,
            alpha_weight=args.alpha_weight,
            lambda_weight=args.lambda_weight,
            z_dim=args.z_dim
    ).to(args.DEVICE)


    return Lit_InfoVAE



def load_data(
        args: any
    ) -> (
            DataLoader,
            DataLoader,
            any,
            int
    ):
        
        #load hyperparameters
        
        def split_data(
                args: any,
                X: torch.FloatTensor
            ) -> (
                    torch.FloatTensor,
                    torch.FloatTensor
            ):

                train_X, valid_X = train_test_split(
                        X,
                        test_size=args.test_size,
                        random_state=args.SEED
                )

                return (
                    train_X,
                    valid_X
                )
        
        if args.homolog_option == 0:
            # CM homologs
    
            # prepare natural homologs as training data
            train_num_X, train_OH_X, train_y = pfam_prep.prepare_CM_dataset(
                 data_path = args.train_path,
                 alignment = args.alignment
            )

            # prepare design homologs as testing data
            test_num_X, test_OH_X, test_y = pfam_prep.prepare_CM_dataset(
                 data_path=args.test_path,
                 alignment=args.alignment
            )
            

            if args.dataset_split == 1:
                # split data into train/valid
                train_num_X, valid_num_X = split_data(
                        args=args,
                        X=train_num_X
                )
                
                train_OH_X, valid_OH_X = split_data(
                        args=args,
                        X=train_OH_X
                )
            
                train_y, valid_y = split_data(
                        args=args,
                        X=train_y
                )
  
                # valid dataset:
                valid_dataset = pfam_prep.CM_dataset(
                                            num_inputs = valid_num_X,
                                            onehot_inputs = valid_OH_X,
                                            pheno_outputs = valid_y
                )

                valid_dataloader = DataLoader(
                                        dataset = valid_dataset,
                                        batch_size = args.batch_size,
                                        num_workers = 4,
                                        shuffle = False
                )


                
            # train dataset:
            train_dataset = pfam_prep.CM_dataset(
                                    num_inputs = train_num_X,
                                    onehot_inputs = train_OH_X,
                                    pheno_outputs = train_y
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
                                        pheno_outputs = test_y
            )

            test_dataloader = DataLoader(
                                    dataset = test_dataset,
                                    batch_size = args.batch_size,
                                    num_workers = 4,
                                    shuffle = False
            )


        elif args.homolog_option == 1:
            
            
            # S1A serine protease family:
     
            # prepare natural homologs as training data
            train_num_X, train_OH_X, _, _, _, _, _, _, _ = pfam_prep.prepare_S1A_dataset(
                     data_path = args.data_path,
                     alignment = args.alignment
            )
 
            if args.dataset_split == 1:
                # split data into train/valid
                train_num_X, valid_num_X = split_data(
                        args=args,
                        X=train_num_X
                )
                
                train_OH_X, valid_OH_X = split_data(
                        args=args,
                        X=train_OH_X
                )
                
                #valid dataset:
                valid_dataset = pfam_prep.S1A_dataset(
                                            num_inputs = valid_num_X,
                                            onehot_inputs = valid_OH_X,
                )

                valid_dataloader = DataLoader(
                                        dataset = valid_dataset,
                                        batch_size = args.batch_size,
                                        num_workers = 4,
                                        shuffle = False
                )

            else:
                valid_dataloader = None

       
            # train dataset:
            train_dataset = pfam_prep.S1A_dataset(
                                        num_inputs = train_num_X,
                                        onehot_inputs = train_OH_X,
            )

            train_dataloader = DataLoader(
                                    dataset = train_dataset,
                                    batch_size = args.batch_size,
                                    num_workers = 4,
                                    shuffle = True
            )


            test_dataloader = None

        elif args.homolog_option == 2:
            # lactamase protein family:
     
            # prepare natural homologs as training data
            train_num_X, train_OH_X = pfam_prep.prepare_lactamase_dataset(
                 data_path = args.data_path,
                 alignment = args.alignment
            )
     
            
            if args.dataset_split == 1:
                # split data into train/valid
                train_num_X, valid_num_X = split_data(
                        args=args,
                        X=train_num_X
                )
                
                train_OH_X, valid_OH_X = split_data(
                        args=args,
                        X=train_OH_X
                )
                
                #valid dataset:
                valid_dataset = pfam_prep.lactamase_dataset(
                                            num_inputs = valid_num_X,
                                            onehot_inputs = valid_OH_X,
                )

                valid_dataloader = DataLoader(
                                        dataset = valid_dataset,
                                        batch_size = args.batch_size,
                                        num_workers = 4,
                                        shuffle = False
                )

            else:
                valid_dataloader = None


            # train dataset:
            train_dataset = pfam_prep.lactamase_dataset(
                                    num_inputs = train_num_X,
                                    onehot_inputs = train_OH_X,
            )

            train_dataloader = DataLoader(
                                dataset = train_dataset,
                                batch_size = args.batch_size,
                                num_workers = 4,
                                shuffle = True
            )


            test_dataloader = None

        elif args.homolog_option == 3:
            
            # G-protein family:
      
            # prepare natural homologs as training data
            train_num_X, train_OH_X = pfam_prep.prepare_Gprotein_dataset(
                 data_path = args.data_path,
                 alignment = args.alignment
            )
     
            if args.dataset_split == 1:
                # split data into train/valid
                train_num_X, valid_num_X = split_data(
                        args=args,
                        X=train_num_X
                )
                
                train_OH_X, valid_OH_X = split_data(
                        args=args,
                        X=train_OH_X
                )
                
                #valid dataset:
                valid_dataset = pfam_prep.Gprotein_dataset(
                                            num_inputs = valid_num_X,
                                            onehot_inputs = valid_OH_X,
                )

                valid_dataloader = DataLoader(
                                        dataset = valid_dataset,
                                        batch_size = args.batch_size,
                                        num_workers = 4,
                                        shuffle = False
                )

            else:
                valid_dataloader = None

            # train dataset:
            train_dataset = pfam_prep.Gprotein_dataset(
                                    num_inputs = train_num_X,
                                    onehot_inputs = train_OH_X,
            )

            train_dataloader = DataLoader(
                                dataset = train_dataset,
                                batch_size = args.batch_size,
                                num_workers = 4,
                                shuffle = False
            )

            test_dataloader = None

        elif args.homolog_option == 4:
            
            
            # DHFR protein family:
       
            # prepare natural homologs as training data
            train_num_X, train_OH_X = pfam_prep.prepare_DHFR_dataset(
                 data_path = args.data_path,
                 alignment = args.alignment
            )
     
            if args.dataset_split == 1:
                # split data into train/valid
                train_num_X, valid_num_X = split_data(
                        args=args,
                        X=train_num_X
                )
                
                train_OH_X, valid_OH_X = split_data(
                        args=args,
                        X=train_OH_X
                )
                
                #valid dataset:
                valid_dataset = pfam_prep.DHFR_dataset(
                                            num_inputs = valid_num_X,
                                            onehot_inputs = valid_OH_X,
                )

                valid_dataloader = DataLoader(
                                        dataset = valid_dataset,
                                        batch_size = args.batch_size,
                                        num_workers = 4,
                                        shuffle = False
                )

            else:
                valid_dataloader = None

          
            # train dataset:
            train_dataset = pfam_prep.DHFR_dataset(
                                        num_inputs = train_num_X,
                                        onehot_inputs = train_OH_X,
            )

            train_dataloader = DataLoader(
                                    dataset = train_dataset,
                                    batch_size = args.batch_size,
                                    num_workers = 4,
                                    shuffle = True
            )



            test_dataloader = None
        else:
            pass

        _, protein_len, _ = train_OH_X.shape

        return (
            train_dataloader,
            valid_dataloader,
            test_dataloader,
            protein_len
        )


def get_args():


    # write output path name
    parser = argparse.ArgumentParser()
    parser.add_argument('--SEED', dest = 'SEED', default = 42, type = int, help = 'Flag: andom seed.')
    parser.add_argument('--homolog_option', dest = 'homolog_option', default = 0, type = int, help = 'Flag: choose which homolog protien task to work on.')
    parser.add_argument('--data_path', dest='data_path', default='.././data/protein_families/DHFR/pfam_DHFR.csv')
    parser.add_argument('--train_path', dest='train_path', default='.././data/protein_families/DHFR/pfam_DHFR.csv')
    parser.add_argument('--test_path', dest='test_path', default='.././data/protein_families/DHFR/pfam_DHFR.csv')
    parser.add_argument('--epochs', dest = 'epochs', default =  1, type = int, help = 'Flag: max number of epochs for training')    
    parser.add_argument('--alignment', dest = 'alignment', default =  False, type = bool, help = 'Flag: Choose whether to use alignment or not.')    
    parser.add_argument('--output_results_path', dest = 'output_results_path', default = './output/train_sess', type = str, help = 'Flag: Choose directory path for csv logger')
    parser.add_argument('--model_output_path', dest = 'model_output_path', default = './output/train_sess', type = str, help = 'Flag: Choose directory path for model')
    parser.add_argument('--DEVICE', dest='DEVICE', default='CUDA',type=str)
    parser.add_argument('--batch_size', dest='batch_size', default=512)
    parser.add_argument('--dataset_split', dest='dataset_split', default=1, type=int)   
    parser.add_argument('--test_size', dest='test_size', default=0.2, type=float)   
    

    # model hyperparameters
    parser.add_argument('--z_dim', dest='z_dim', default=6, type=int)
    parser.add_argument('--class_labels', dest='class_labels', default=21, type=int)

    # encoder
    parser.add_argument('--encoder_rates', dest='encoder_rates', default=0, type=int)
    parser.add_argument('--C_in', dest='C_in', default=21, type=int)
    parser.add_argument('--C_out', dest='C_out', default=32, type=int)
    parser.add_argument('--alpha', dest='alpha', default=0.1, type=float)
    parser.add_argument('--num_fc', dest='num_fc', default=2, type=int)
    parser.add_argument('--enc_kernel', dest='enc_kernel', default=3, type=int)

    # generator
    parser.add_argument('--wave_hidden_state', dest='wave_hidden_state', default=128, type=int)
    parser.add_argument('--head_hidden_state', dest='head_hidden_state', default=128, type=int)
    parser.add_argument('--num_dil_rates', dest='num_dil_rates', default=256, type=int)
    parser.add_argument('--dec_kernel_size', dest='dec_kernel_size', default=3, type=int)
    parser.add_argument('--aa_labels', dest='aa_labels', default=21, type=int)
   
    # loss values
    parser.add_argument('--xi_weight', dest='xi_weight', default=1.0, type=float)
    parser.add_argument('--alpha_weight', dest='alpha_weight', default=0.99, type=float)
    parser.add_argument('--lambda_weight', dest='lambda_weight', default=10, type=float)
    parser.add_argument('--lr', dest='lr', default=1e-4, type=float)

    args = parser.parse_args()
 
    return args

  
def train_model(
        args: any,
        PL_model: pl.LightningModule,
        train_dataloader: DataLoader,
        valid_dataloader: any
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
            trainer.fit(PL_model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)
            
            
            # track losses
            train_L = trainer.callback_metrics['L_train_epoch'].item()
            train_NLL = trainer.callback_metrics['L_nll_train_epoch'].item()
            train_kld = trainer.callback_metrics['L_kld_train_epoch'].item()
            train_mmd = trainer.callback_metrics['L_mmd_train_epoch'].item()
            valid_L = trainer.callback_metrics['L_valid_epoch'].item()
            valid_NLL = trainer.callback_metrics['L_nll_valid_epoch'].item()
            valid_kld = trainer.callback_metrics['L_kld_valid_epoch'].item()
            valid_mmd = trainer.callback_metrics['L_mmd_valid_epoch'].item()
        
            final_epoch_results = [
                    train_L,
                    train_NLL,
                    train_kld,
                    train_mmd,
                    valid_L,
                    valid_NLL,
                    valid_kld,
                    valid_mmd
            ]
            
            # track whole loss history
            all_epochs_losses = {}           
            all_epochs_losses['train_L'] = [float('nan')] + PL_model.L_train_list
            all_epochs_losses['train_nll'] = [float('nan')] + PL_model.L_train_nll_list
            all_epochs_losses['train_kld'] = [float('nan')] + PL_model.L_train_kld_list
            all_epochs_losses['train_mmd'] = [float('nan')] + PL_model.L_train_mmd_list
 
            all_epochs_losses['val_L'] = PL_model.L_val_list
            all_epochs_losses['val_nll'] = PL_model.L_val_nll_list
            all_epochs_losses['val_kld'] = PL_model.L_val_kld_list
            all_epochs_losses['val_mmd'] = PL_model.L_val_mmd_list


        else:
            print('\nTrain on the whole data\n')
            trainer.fit(PL_model, train_dataloader)
            
            
            # track losses
            train_L = trainer.callback_metrics['L_train_epoch'].item()
            train_NLL = trainer.callback_metrics['L_nll_train_epoch'].item()
            train_kld = trainer.callback_metrics['L_kld_train_epoch'].item()
            train_mmd = trainer.callback_metrics['L_mmd_train_epoch'].item()
 
            final_epoch_results = [
                    train_L,
                    train_NLL,
                    train_kld,
                    train_mmd
            ]
            
            # track whole loss history
            all_epochs_losses = {}           
            all_epochs_losses['train_L'] = PL_model.L_train_list
            all_epochs_losses['train_nll'] = PL_model.L_train_nll_list
            all_epochs_losses['train_kld'] = PL_model.L_train_kld_list
            all_epochs_losses['train_mmd'] = PL_model.L_train_mmd_list
 
              
        return (
                final_epoch_results,
                all_epochs_losses
        )

def set_SEED(args:any):

    return seed_everything(args.SEED)

def set_GPU(args:any) -> str:
 
    # activate GPU
    # ----------------------------
    torch.backends.cudnn.enabled
    
    USE_CUDA = False

    if torch.cuda.is_available():
        USE_CUDA = True    
        print('GPU available')
    else:
       print('Please enable GPU or use CPU')
       #quit()
  
    DEVICE = 'cuda' if USE_CUDA else 'cpu'

    return DEVICE

def save_results(
        args: any,
        PL_model: any,
        final_epoch_results: list,
        all_epoch_losses: dict
    ) -> None:

    # save results
    # final epoch losses
    
    if args.dataset_split == 1: # with train/val split
        
        final_epoch_columns = [
            'train_L',
            'train_nll',
            'train_kld',
            'train_mmd',
            'val_L',
            'val_nll',
            'val_kld',
            'val_mmd',
        ]

    else:

        final_epoch_columns = [
            'train_L',
            'train_nll',
            'train_kld',
            'train_mmd',
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



if __name__ == '__main__':

    args = get_args()
  
    # reproducibility
    set_SEED(args=args)
    
    # set GPU (cuda)
    args.DEVICE = set_GPU(args=args)


    # load Data
    train_dataloader, valid_dataloader, test_dataloader, protein_len = load_data(args=args)
   
    # call model
    PL_model = call_model(
           args=args,
           protein_len = protein_len,
    )

    # train model
    final_epoch_results, all_epoch_results = train_model(
                                                args=args,
                                                PL_model=PL_model,
                                                train_dataloader=train_dataloader,
                                                valid_dataloader=valid_dataloader
    )
    
    # save results
    save_results(
            args=args,
            PL_model=PL_model,
            final_epoch_results=final_epoch_results,
            all_epoch_losses=all_epoch_results
    )
