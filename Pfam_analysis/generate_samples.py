
import torch
from torch import nn
import torch.distributions as dist

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
def sample_seqs(
	args: any,
	model: nn.Module,
        protein_len: int,
	n: int=100
    )-> (
            torch.FloatTensor,
            torch.FloatTensor
    ):


    # eval mode
    model.eval()

    # set up the sequence context
    X_context = torch.zeros((n,protein_len,21)).to(args.DEVICE)

    # setup sampling distribution
    normal_dist = create_normal_dist(args=args)
    
    # latent-conditional info.
    Z_context = normal_dist.sample((n,)).to(args.DEVICE)

    # generate sequences
    X_samples = model.sample(
        args=args,
        X_context=X_context,
        z=Z_context,
        option='categorical'
    ).cpu()


    return (
        X_samples,
        Z_context
    )

def create_df(
        args: any,
        X: torch.FloatTensor,
        z: torch.FloatTensor
    ) -> None:
    

    # create empty dict
    sample_dict = {}
    
    # create sequences
    aa_seqs = util_tools.create_seqs(X=X)
    ids = [f'id_{ii}' for ii in range(len(aa_seqs))]

    sample_dict['id'] = ids
    sample_dict['sequence'] = aa_seqs
    
    for z_axis in range(args.z_dim):

        sample_dict[f'z_{z_axis}'] = list(z[:,z_axis].cpu().numpy())


    sample_df = pd.DataFrame(sample_dict)
    colabfold_df = sample_df[['id', 'sequence']]

    # save dataframes
    sample_df.to_csv(args.samples_output_path, index=False)
    colabfold_df.to_csv(args.samples_output_path.replace('.csv','_colabfold.csv'), index=False)


    return 
        




if __name__ == '__main__':

    # get variable arguments
    parser = argparse.ArgumentParser()
    train_sess.get_args(parser)
    sample_get_args(parser)
    CM_train_sess.get_SS_args(parser)
    args = parser.parse_args()
    args.alignment = False # only unalignments 
    os.makedirs(args.folder_path, exist_ok=True)

    # reprod.
    train_sess.set_SEED(args=args)

    # set GPU (cuda)
    args.DEVICE = train_sess.set_GPU(args=args)

    # load data ( get sequence length)
    _, _, _, protein_len = train_sess.load_data(args=args)

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

    else:
        print('Only learning option is the Semi-supervised or unsupervised')

    # load weights
    model = load_weights(
            args=args,
            model=PL_model.model
    )

    X_samples, Z_samples = sample_seqs(
            args=args,
            model=model,
            protein_len=protein_len,
            n=100
    )
    

    create_df(
            args=args,
            X=X_samples,
            z=Z_samples
    )




