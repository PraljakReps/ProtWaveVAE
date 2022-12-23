
import torch
from torch import nn
import torch.distributions as dist
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl


import source.preprocess as prep
import source.pfam_preprocess as pfam_prep
import source.wavenet_decoder as wavenet
import source.model_components as model_comps
import source.PL_wrapper as PL_mod
import train_on_pfam as train_sess
import utils.tools as util_tools
import train_on_CM as CM_train_sess

import numpy as np
import pandas as pd
import sys
import argparse
from tqdm import tqdm
import os

def sample_get_args(parser):

    parser.add_argument('--samples_output_path', dest='samples_output_path', default='./outputs/prediction', type=str, help='Flag: Choose directory path for the design sequence data')
    parser.add_argument('--weights_path', dest='weights_path', default='./outputs/prediction', type=str, help='Flag: Choose directory path for pretrained weights')
    parser.add_argument('--folder_path', dest='folder_path', default='./outputs/prediction', type=str, help='Flag: Choose directory path for folder')


def load_weights(
        args: any,
        model: nn.Module
    ) -> nn.Module:

    model.load_state_dict(torch.load(args.weights_path))
    model.eval()

    return model

def create_normal_dist(args):

    loc = torch.zeros(args.z_dim)
    scale = torch.ones(args.z_dim)
    normal = dist.normal.Normal(loc, scale)

    return normal

@torch.no_grad()
def infer_latents(
	args: any,
	model: nn.Module,
        protein_len: int,
        train_dataset: Dataset,
        valid_dataset: any,
        test_dataset: any,
    )-> (
            torch.FloatTensor,
            torch.FloatTensor
    ):

    # eval mode
    model.eval()
    
    if args.learning_option == 'semi-supervised':
        _, _, z_train_sample, z_train_mode, _ = model(train_dataset[0:][1].to(args.DEVICE))
    elif args.learning_option == 'unsupervised':
        _, z_train_sample, z_train_mode, _ = model(train_dataset[0:][1].to(args.DEVICE))
   
    return (
        z_train_sample,
        z_train_mode
    )

def append_to_df(
        args: any,
        df: pd.Series,
        z_sample: torch.FloatTensor,
        z_mode: torch.FloatTensor
    ) -> None:
    
    for z_axis in range(args.z_dim):

        df[f'z_{z_axis}'] = list(z_sample[:,z_axis].cpu().numpy())
    
    for z_axis in range(args.z_dim):

        df[f'z_{z_axis}_mode'] = list(z_mode[:,z_axis].cpu().numpy())
    
    # save dataframes
    df.to_csv(args.samples_output_path, index=False)

    return 
        

def get_df(args: any) -> pd.Series:

    return pd.read_csv(args.data_path)


if __name__ == '__main__':

    # get variable arguments
    parser = argparse.ArgumentParser()
    train_sess.get_args(parser)
    sample_get_args(parser)
    CM_train_sess.get_SS_args(parser)
    args = parser.parse_args()
    args.alignment = False
    os.makedirs(args.folder_path, exist_ok=True)

    # reprod.
    train_sess.set_SEED(args=args)

    # set GPU (cuda)
    args.DEVICE = train_sess.set_GPU(args=args)

    # load data ( get sequence length)
    train_dataloader, _, test_dataloader, protein_len = train_sess.load_data(args=args)

    # call model
    if args.learning_option == 'semi-supervised':
        PL_model = CM_train_sess.call_SS_model(
            args=args,
            protein_len=protein_len
        )

    elif args.learning_option == 'unsupervised':
        PL_model = train_sess.call_model(
            args=args,
            protein_len=protein_len
        )


    # load weights
    model = load_weights(
            args=args,
            model=PL_model.model
    )

    z_train_sample, z_train_mode = infer_latents(
	args=args,
	model=model,
        protein_len=protein_len,
        train_dataset=train_dataloader.dataset,
        valid_dataset=None,
        test_dataset=None
    )
    
    # get training dataset
    df = get_df(args=args)
    
    # add unaligned sequence column for CM family
    df['Unaligned_sequence'] = [seq.replace('-','') for seq in df.Sequence]

    # append latent inferred codes to training dataset
    append_to_df(
            args=args,
            df=df,
            z_sample=z_train_sample,
            z_mode=z_train_mode
    )




