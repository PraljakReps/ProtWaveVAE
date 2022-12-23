
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.autograd import Variable
from torch.nn import functional as F

import optuna
from optuna.integration import PyTorchLightningPruningCallback

import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping

import source.preprocess as prep
import source.model_components as model_comps
import source.wavenet_decoder as wavenet
import source.PL_wrapper as PL_wrapper

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import sys
import argparse
from tqdm import tqdm
import random

from sklearn.model_selection import train_test_split





class Objective(object):

    def __init__(
            self,
            args,
            max_seq_len,
            train_dataloader,
            valid_dataloader,
            z_dim,
            encoder_rates,
            C_out,
            enc_kernel,
            disc_num_layers,
            hidden_width,
            p,
            num_classes,
            wave_hidden_state,
            head_hidden_state,
            num_dil_rates,
            dec_kernel_size,
            nll_weight,
            MI_weight,
            lambda_weight,
            gamma_weight
        ):


        self.args = args
        self.max_seq_len = max_seq_len
        self.DEVICE = args.DEVICE
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        
        # encoder hyperparameters
        self.z_dim = z_dim
        self.encoder_rates = encoder_rates
        self.C_out = C_out
        self.enc_kernel = enc_kernel

        # discriminative top model hyperparameters
        self.disc_num_layers = disc_num_layers
        self.hidden_width = hidden_width
        self.p = p
        self.num_classes = num_classes

        # autoregressive decoder hyperparameters
        self.wave_hidden_state = wave_hidden_state
        self.head_hidden_state = head_hidden_state
        self.num_dil_rates = num_dil_rates
        self.dec_kernel_size = dec_kernel_size

        # prefactor loss weights
        self.nll_weight = nll_weight
        self.MI_weight = MI_weight
        self.lambda_weight = lambda_weight
        self.gamma_weight = gamma_weight



    def __call__(self, trial):
        # calculate an objective value by using the extra arguments ...

              
              
        # encoder hyperparameters
        z_dim = trial.suggest_categorical('z_dim', self.z_dim)
        encoder_rates = trial.suggest_categorical('encoder_rates', self.encoder_rates)
        C_out = trial.suggest_categorical('C_out', self.C_out)
        enc_kernel = trial.suggest_categorical('enc_kernel', self.enc_kernel)

        # discriminative top model hyperparameters
        disc_num_layers = trial.suggest_categorical('disc_num_layers', self.disc_num_layers)
        hidden_width = trial.suggest_categorical('hidden_width', self.hidden_width)
        p = trial.suggest_categorical('p', self.p)

        # autoregressive decoder hyperparameters
        wave_hidden_state = trial.suggest_categorical('wave_hidden_state', self.wave_hidden_state)
        head_hidden_state = trial.suggest_categorical('head_hidden_state', self.head_hidden_state)
        num_dil_rates = trial.suggest_categorical('num_dil_rates', self.num_dil_rates)
        dec_kernel_size = trial.suggest_categorical('dec_kernel_size', self.dec_kernel_size)

        # loss hps
        nll_weight = trial.suggest_categorical('nll_weight', self.nll_weight)
        MI_weight = trial.suggest_categorical('MI_weight', self.MI_weight)
        lambda_weight = trial.suggest_categorical('lambda_weight', self.lambda_weight)
        gamma_weight = trial.suggest_categorical('gamma_weight', self.gamma_weight)
        
        
        # define inference model:
        encoder = model_comps.GatedCNN_encoder(
                                  protein_len=self.max_seq_len,
                                  z_dim=z_dim,
                                  num_rates=encoder_rates,
                                  C_in=args.C_in,
                                  C_out=C_out,
                                  alpha=args.alpha,
                                  kernel=enc_kernel,
                                  pad_option="valid"
        )
    
        
        # define regression model:
        decoder_re = model_comps.Decoder_re(
                                    num_layers=disc_num_layers,
                                    hidden_width=hidden_width,
                                    z_dim=z_dim,
                                    num_classes=args.num_classes,
                                    p=p
        )

        # define generator model:
        decoder_wave = wavenet.Wave_generator(
                                protein_len=self.max_seq_len,
                                class_labels=args.aa_labels,
                                DEVICE=args.DEVICE,
                                wave_hidden_state=wave_hidden_state,
                                head_hidden_state=head_hidden_state,
                                num_dil_rates=num_dil_rates,
                                kernel_size=dec_kernel_size
        )

        # latent global conditioning
        cond_mapper = wavenet.CondNet(
                                z_dim=z_dim,
                                output_shape=(1, self.max_seq_len)
        )

        # define final model configuration ...
        # torch model
        SS_model = model_comps.SS_InfoVAE(
                             DEVICE=args.DEVICE,
                             encoder=encoder,
                             decoder_recon=decoder_wave,
                             cond_mapper=cond_mapper,
                             decoder_pheno=decoder_re,
                             z_dim=z_dim
        )

        # pytorch model
        PL_model = PL_wrapper.Lit_SSInfoVAE(
                            DEVICE=args.DEVICE,
                            SS_InfoVAE=SS_model,
                            xi_weight=nll_weight,
                            alpha_weight=MI_weight,
                            lambda_weight=lambda_weight,
                            gamma_weight=gamma_weight,
                            lr=args.lr,
                            z_dim=z_dim
        )

        trainer = pl.Trainer(
                logger = False,
                max_epochs = args.epochs,
                gpus = 1 if torch.cuda.is_available() else None,
                progress_bar_refresh_rate = False
        )

        trainer.fit(
                PL_model,
                train_dataloaders = self.train_dataloader,
                val_dataloaders = self.valid_dataloader
        )


        train_L = trainer.callback_metrics['L_train_epoch'].item()
        train_NLL = trainer.callback_metrics['L_nll_train_epoch'].item()
        train_kld = trainer.callback_metrics['L_kld_train_epoch'].item()
        train_mmd = trainer.callback_metrics['L_mmd_train_epoch'].item()
        train_pheno = trainer.callback_metrics['L_pheno_train_epoch'].item()
        train_precision_epoch = trainer.callback_metrics['Train_precision_epoch'].item()
        train_recall_epoch = trainer.callback_metrics['Train_recall_epoch'].item()
        train_f1_epoch = trainer.callback_metrics['Train_f1_epoch'].item()

        val_L = trainer.callback_metrics['L_valid_epoch'].item()
        val_NLL = trainer.callback_metrics['L_nll_valid_epoch'].item()
        val_kld = trainer.callback_metrics['L_kld_valid_epoch'].item()
        val_mmd = trainer.callback_metrics['L_mmd_valid_epoch'].item()
        val_pheno = trainer.callback_metrics['L_pheno_valid_epoch'].item()
        val_precision_epoch = trainer.callback_metrics['val_precision_epoch'].item()
        val_recall_epoch = trainer.callback_metrics['val_recall_epoch'].item()
        val_f1_epoch = trainer.callback_metrics['val_f1_epoch'].item()

        final_epoch_results = [
            train_L,
            val_L,
            train_NLL,
            train_kld,
            train_mmd,
            train_pheno,
            val_NLL,
            val_kld,
            val_mmd,
            val_pheno,
        ]

        all_epochs_losses = {
            'train_L': list(),
            'train_nll': list(),
            'train_kld': list(),
            'train_mmd': list(),
            'train_pheno': list(),
            'train_precision': list(),
            'train_recall': list(),
            'train_f1': list(),
            'valid_L': list(),
            'valid_nll': list(),
            'valid_kld': list(),
            'valid_mmd': list(),
            'valid_pheno': list(),
            'valid_precision': list(),
            'valid_recall': list(),
            'valid_f1': list()
        }

        # allocate losses from all of the epochs:
        all_epochs_losses['train_L'] = PL_model.L_train_list + [float('nan')]
        all_epochs_losses['train_nll'] = PL_model.L_train_nll_list + [float('nan')]
        all_epochs_losses['train_kld'] = PL_model.L_train_kld_list + [float('nan')]
        all_epochs_losses['train_mmd'] = PL_model.L_train_mmd_list + [float('nan')]
        all_epochs_losses['train_pheno'] = PL_model.L_train_pheno_list + [float('nan')]
        all_epochs_losses['train_precision'] = PL_model.train_precision_list + [float('nan')]
        all_epochs_losses['train_recall'] = PL_model.train_recall_list + [float('nan')]
        all_epochs_losses['train_f1'] = PL_model.train_f1_list + [float('nan')]

        all_epochs_losses['valid_L'] = PL_model.L_val_list
        all_epochs_losses['valid_nll'] = PL_model.L_val_nll_list
        all_epochs_losses['valid_kld'] = PL_model.L_val_kld_list
        all_epochs_losses['valid_mmd'] = PL_model.L_val_mmd_list
        all_epochs_losses['valid_pheno'] = PL_model.L_val_pheno_list
        all_epochs_losses['valid_precision'] = PL_model.val_precision_list
        all_epochs_losses['valid_recall'] = PL_model.val_recall_list
        all_epochs_losses['valid_f1'] = PL_model.val_f1_list



        return final_epoch_results


def prepare_data(
        args: any,
        train: any,
        test: any,
        num_workers: int = 4,
        SEED: int = 42
    ) -> (
            DataLoader,
            DataLoader,
            int,
            Dataset,
            Dataset
    ):
        
        train_X, train_pheno, train_C, train_accept = train
        test_X, test_pheno, test_C, test_accept = test

        # train dataset
        train_dataset = prep.SH3_dataset(
                                onehot_inputs=train_X,
                                re_inputs=train_pheno,
                                C_inputs=train_C,
                                accept_inputs=train_accept
        )


        train_dataloader = DataLoader(
                            train_dataset,
                            batch_size = args.batch_size,
                            num_workers = 8,
                            shuffle = True
        )

        # test dataset
        test_dataset = prep.SH3_dataset(
                                onehot_inputs=test_X,
                                re_inputs=test_pheno,
                                C_inputs=test_C,
                                accept_inputs=test_accept
        )


        test_dataloader = DataLoader(
                            test_dataset,
                            batch_size = args.batch_size,
                            num_workers = 8,
                            shuffle = True
        )


        _, max_protein_len, _ = train_X.shape

        return (
            train_dataloader,
            test_dataloader,
            max_protein_len,
            train_dataset,
            test_dataset
        )

def CV_train(
        args,
    ):


    # encoder variables
    z_dim = [int(item) for item in args.z_dim.split(',')] # size of latent space
    encoder_rates = [int(item) for item in args.encoder_rates.split(',')] # depth of dilated encoder
    C_out = [int(item) for item in args.C_out.split(',')] # conv no. filters
    enc_kernel = [int(item) for item in args.enc_kernel.split(',')] # size of the encoder kernel
    
    # discriminative top model variables
    disc_num_layers = [int(item) for item in args.disc_num_layers.split(',')] # number of disc layers
    hidden_width = [int(item) for item in args.hidden_width.split(',')] # size of the MLP
    p = [float(item) for item in args.p.split(',')] # dropout hp
    
    # autoregressive decoder variables
    wave_hidden_state = [int(item) for item in args.wave_hidden_state.split(',')] # number of kernel filters for wavenet
    head_hidden_state = [int(item) for item in args.head_hidden_state.split(',')] # number of kernel filters for wavenet's top model
    num_dil_rates = [int(item) for item in args.num_dil_rates.split(',')] # depth of the wavenet model
    dec_kernel_size = [int(item) for item in args.dec_kernel_size.split(',')] # kernel size of the decoder

    # learning rate variables 
    nll_weight = [float(item) for item in args.nll_weight.split(',')] # NLL prefactor weight
    MI_weight = [float(item) for item in args.MI_weight.split(',')] # MI prefactor weight
    lambda_weight = [float(item) for item in args.lambda_weight.split(',')] # lambda prefactor weight
    gamma_weight = [float(item) for item in args.gamma_weight.split(',')] # gamma prefactor weight  

    
    hp_dict = {
            'z_dim':z_dim,
            'encoder_rates':encoder_rates,
            'C_out':C_out,
            'enc_kernel':enc_kernel,
            'disc_num_layers':disc_num_layers,
            'hidden_width':hidden_width,
            'p':p,
            'wave_hidden_state':wave_hidden_state,
            'head_hidden_state':head_hidden_state,
            'num_dil_rates':num_dil_rates,
            'dec_kernel_size':dec_kernel_size,
            'nll_weight':nll_weight,
            'MI_weight':MI_weight,
            'lambda_weight':lambda_weight,
            'gamma_weight':gamma_weight
    }


    from sklearn.model_selection import KFold
    kf = KFold(n_splits = args.K)


    df = pd.read_csv(args.dataset_path)
    max_seq_len = max([len(seq) for seq in df.Sequences_unaligned.values])


    X_OH, y_pheno, y_C, y_accept = prep.prepare_SH3_data(
                                df=df,
                                max_seq_len=max_seq_len,
    )


    for ii, (train_index, test_index) in enumerate(kf.split(X_OH)):

        # split data into train/test
        X_train, X_test = X_OH[train_index], X_OH[test_index]
        y_pheno_train, y_pheno_test = y_pheno[train_index], y_pheno[test_index]
        y_C_train, y_C_test = y_C[train_index], y_C[test_index]
        y_accept_train, y_accept_test = y_accept[train_index], y_accept[test_index]


        train_dataloader, test_dataloader, max_seq_len, train_dataset, test_dataset  = prepare_data(
                args=args,
                train=(X_train, y_pheno_train, y_C_train, y_accept_train),
                test=(X_test,y_pheno_test,y_C_test,y_accept_test),
                num_workers=4,
                SEED=args.SEED
        )
    
        # optuna hp optimization:
        search_space = {args.search_variable: hp_dict[args.search_variable]}
        
        study = optuna.create_study(
                sampler = optuna.samplers.GridSampler(search_space),
                directions = [
                    'minimize',
                    'minimize',
                    'minimize',
                    'minimize',
                    'minimize',
                    'minimize',
                    'minimize',
                    'minimize',
                    'minimize',
                    'minimize',
                    ]
        )

        study.optimize(
                Objective(
                    args=args,
                    max_seq_len=max_seq_len,
                    train_dataloader=train_dataloader,
                    valid_dataloader=test_dataloader,
                    z_dim=z_dim,
                    encoder_rates=encoder_rates,
                    C_out=C_out,
                    enc_kernel=enc_kernel,
                    disc_num_layers=disc_num_layers,
                    hidden_width=hidden_width,
                    p=p,
                    num_classes=args.num_classes,
                    wave_hidden_state=wave_hidden_state,
                    head_hidden_state=head_hidden_state,
                    num_dil_rates=num_dil_rates,
                    dec_kernel_size=dec_kernel_size,
                    nll_weight=nll_weight,
                    MI_weight=MI_weight,
                    lambda_weight=lambda_weight,
                    gamma_weight=gamma_weight
                ),
        n_trials=args.n_trials,
        timeout = None
        )
              
        df = study.trials_dataframe()
        df.to_csv(f'{args.output_results_path}_{args.search_variable}_k{ii}.csv', index = False)


    return print('Finished')


def get_args() -> any:


    # write output path name
    parser = argparse.ArgumentParser()

    
    # path variables
    parser.add_argument('--dataset_path', default = './data/ACS_SynBio_SH3_dataset.csv', help = 'flag: choose output path')
    parser.add_argument('--output_results_path', default = './outputs/SH3_task/hp_optim/ProtWaveVAE', help = 'flag: choose output path', type = str)
  
    # model training variables
    parser.add_argument('--SEED', default = 42, help = 'flag: random seed', type = int)
    parser.add_argument('--batch_size', default = 512, help = 'flag: batch size', type = int)
    parser.add_argument('--epochs', default = 1, help = 'flag: number of training epochs', type = int)
    parser.add_argument('--lr', default = 1e-4, help = 'flag: learning rate', type = float)
    parser.add_argument('--DEVICE', default = 'cuda', help = 'flag: setup GPU', type = str)
   
    # general architecture variables
    parser.add_argument('--z_dim', default='6', type=str, help='Latent space size')
    parser.add_argument('--num_classes', default=1, type=int, help='functional/nonfunctional labels')
    parser.add_argument('--aa_labels', default=21, type=int, help='AA plus pad gap (20+1) labels')

    # encoder hyperparameters
    parser.add_argument('--encoder_rates', default='5', type=str, help='dilation convolution depth')
    parser.add_argument('--C_in', default=21, type=int, help='input feature depth')
    parser.add_argument('--C_out', default='256', type=str, help='output feature depth')
    parser.add_argument('--alpha', default=0.1, type=float, help='leaky Relu hyperparameter (optional)')
    parser.add_argument('--enc_kernel', default='3', type=str, help='kernel filter size')

    # top model (discriminative decoder) hyperparameters
    parser.add_argument('--disc_num_layers', default='2', type=str, help='depth of the discrim. top model')
    parser.add_argument('--hidden_width', default='10', type=str, help='width of top model')
    parser.add_argument('--p', default='0.3', type=str, help='top model dropout')

    # decoder wavenet hyperparameters
    parser.add_argument('--wave_hidden_state', default='256', type=str, help='no. filters for the dilated convolutions')
    parser.add_argument('--head_hidden_state', default='128', type=str, help='no. filters for the WaveNets top model')
    parser.add_argument('--num_dil_rates', default='8', type=str, help='depth of the WaveNet')
    parser.add_argument('--dec_kernel_size', default='3', type=str, help='WaveNet kernel size')

    # loss prefactor weights
    parser.add_argument('--nll_weight', default='1.', type=str, help='NLL prefactor weight')
    parser.add_argument('--MI_weight', default='0.95', type=str, help='MI prefactor weight')
    parser.add_argument('--lambda_weight', default='2.', type=str, help='MMD prefactor weight')
    parser.add_argument('--gamma_weight', default='1.', type=str, help='discriminative prefactor weight')

 
    parser.add_argument('--search_variable', default = 'z_dim', help = 'Flag: choose hyperparameter variable to grid search.', type = str)
    parser.add_argument('--n_trials', default = 1, help = 'Flag: choose number of trials.', type = int)
    parser.add_argument('--K', default = 5, help = 'Flag: Cross-validation', type = int)
    
    
    args = parser.parse_args()

    return args


def set_GPU() -> None:


    if torch.cuda.is_available():
        print('GPU available')

    else:
        print('Please enable GPU')
        quit()
    
    USE_CUDA = True
    DEVICE = 'cuda' if USE_CUDA else 'cpu'

    return 

def set_SEED(args: any) -> None:

    seed_everything(args.SEED, workers=True)
    return


if __name__ == '__main__':


    # get argument variables:
    # ---------------
    args = get_args()

    # activate GPU:
    # ---------------
    set_GPU()

    # set seed for reproducibility
    set_SEED(args)


    CV_train(
            args=args,
    )
