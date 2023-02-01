
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.autograd import Variable
from torch.nn import functional as F

# Pytorch lightning
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping

import source.preprocess as prep
import source.wavenet_decoder as wavenet
import source.model_components as model_comps
import source.PL_wrapper as PL_wrapper
import train_ProtWaveVAE as ProtWaveVAE
import generate_proteins as gen_tools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import sys
import argparse
from tqdm import tqdm
import random
import os



def get_data(args: any) -> (
        pd.Series,
        DataLoader,
        int
    ):

    df = pd.read_csv(args.dataset_path)

    max_seq_len = max([len(seq) for seq in df.Sequences_unaligned.values])
    
    OH, pheno, C, accept = prep.prepare_SH3_data(
            df=df,
            max_seq_len=max_seq_len,
    )

    # train dataset
    dataset = prep.SH3_dataset(
            onehot_inputs=OH,
            re_inputs=pheno,
            C_inputs=C,
            accept_inputs=accept
    )

    dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=8,
            shuffle=False
    )

    # extract the sequence length
    _, protein_len, _ = OH.shape

    print('Size of the training dataset:', OH.shape[0])
    return (
            df,
            dataloader,
            protein_len
    )


@torch.no_grad()
def infer_latents(
        args: any,
        model: nn.Module,
        dataloader: DataLoader
    ) -> (
            torch.FloatTensor,
            torch.FloatTensor
    ):

   
    # evaluation mode
    model.eval()

    data_size = dataloader.dataset[0:][0].shape[0]

    # latent codes
    z = torch.zeros((data_size, args.z_dim))
    z_mean = torch.zeros((data_size, args.z_dim))

    for ii, batch in tqdm( enumerate(dataloader) ):

        X, _, _, _ = batch

        logits_xrc_temp, _, _, z_temp, z_mean_temp, z_var_temp  = model(X.to(args.DEVICE))
   
        # left and right indexes ... 
        left_idx = ii * args.batch_size
        right_idx = (ii+1) * args.batch_size

        # latent inference
        z[left_idx:right_idx, :] = z_temp
        z_mean[left_idx:right_idx, :]= z_mean_temp 

    return (
            z,
            z_mean
    )


def save_df(
        args: any,
        df: pd.Series,
        Zpred: torch.FloatTensor,
        Zpred_mode: torch.FloatTensor
    ) -> None:

    
    for z_axis in range(args.z_dim):

        df[f'z_{z_axis}'] = list(Zpred[:,z_axis].numpy()[:])
    for z_axis in range(args.z_dim):
    
        df[f'z[mode]_{z_axis}'] = list(Zpred_mode[:,z_axis].numpy()[:])

    df.to_csv(args.output_results_path, index=False)

    return 
    
if __name__ == '__main__':

    args = gen_tools.get_args()
    ProtWaveVAE.set_GPU() # set GPU
    ProtWaveVAE.set_SEED(args=args) # set SEED (reproducibility)
    
    os.makedirs(args.save_dir, exist_ok=True)

    # get data
    df, dataloader, max_seq_len = get_data(args=args)
    args.max_seq_len = max_seq_len 
    
    # get model
    PL_model = ProtWaveVAE.get_model(
                            args=args,
                            protein_len=max_seq_len
    ).to(args.DEVICE)
    model = PL_model.model
    model.load_state_dict(torch.load(args.output_model_path))
    
    print('Start inference')

    # infer latent embeddings using pretrained model
    Zpred, Zpred_mode = infer_latents(
                          args=args,
                          model=model,
                          dataloader=dataloader
    )
    
    save_df(
            args=args,
            df=df,
            Zpred=Zpred,
            Zpred_mode=Zpred_mode
    )
