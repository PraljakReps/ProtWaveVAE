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
import generate_samples as generate_tools

import numpy as np
import pandas as pd
import sys
import argparse
from tqdm import tqdm
import os


def diversify_args(parser):
    
    
    parser.add_argument('--Cterminus_output_path', dest='Cterminus_output_path', default='./outputs/', type=str, help='Flag: path to save dataframe for Cterminus conditioning')
    parser.add_argument('--DeNovo_output_path', dest='DeNovo_output_path', default='./outputs/', type=str, help='Flag: path to save dataframe for de novo design')
    parser.add_argument('--N', dest='N', default=100, type=int, help='Flag: Number of samples to generate')
    
 

####################
# Design functions #
####################

def convert_list_to_tensor(
        seq_list: list,
        max_seq_len: int,
    ) -> torch.FloatTensor:


    padded_seq_list = prep.pad_ends(
            seqs=seq_list,
            max_seq_length=max_seq_len
    )
    num_seq_list = prep.create_num_seqs(padded_seq_list)
    onehot_transformer = torch.eye(21)
    x_onehot = onehot_transformer[num_seq_list]

    return x_onehot


@torch.no_grad()
def diversify_CM_gene(
    args: any,
    model: nn.Module,
    seq_list: list,
    max_seq_len: int,
    z_context: torch.FloatTensor,
    L: int
    ) -> torch.FloatTensor:
    
    # eval mode: 
    model.eval()
    
    # number of candidates 
    n = z_context.shape[0]
    
    # create torch tensor
    X = convert_list_to_tensor(
                seq_list=seq_list,
                max_seq_len=max_seq_len
    )
    

    if len(X.shape) == 2:
        X = X.unsqueeze(0).repeat(n,1,1)
    elif len(X.shape) != 3:
        quit
    else:
        pass
    
    # diversify sequence of interest
    X_diversify_samples = model.diversify(
            args=args,
            X_context=X.to(args.DEVICE),
            z=z_context.to(args.DEVICE),
            L=L,
            option='categorical'
    ).cpu()
    
    return X_diversify_samples


def randomly_diversify_CM_gene(
    args: any,
    model: nn.Module,
    seq_list: list,
    max_seq_len: int,
    L: int
    ) -> torch.FloatTensor:
    
    # eval mode: 
    model.eval()
    
    # gap region
    num_gaps = max_seq_len - len(seq_list[0])
    
    # number of candidates 
    n = 5000
    
    # create torch tensor
    X = convert_list_to_tensor(
                seq_list=seq_list,
                max_seq_len=max_seq_len
    )
    
    if len(X.shape) == 2:
        X = X.unsqueeze(0).repeat(n,1,1)
    elif len(X.shape) != 3:
        quit
    else:
        pass
    
    X_temp = model.create_uniform_tensor(args=args,X=X)
    # diversify sequence of interest
    X_rand_diversify_samples = model.randomly_diversify(
                                            args=args,
                                            X_context=X.to(args.DEVICE),
                                            L=L,
                                            option='categorical'
    ).cpu()
    
    # include deletion gaps
    X_rand_diversify_samples[:, -1, -num_gaps:,:] = X[:,-num_gaps:, :]
    
    return X_rand_diversify_samples, 


#########################################
# Specific functions CM diversification #
#########################################

@torch.no_grad()
def sample_z(
        args: any,
        model: nn.Module,
        protein_len: int,
        n: int=100
    ) -> torch.FloatTensor:

    # eval mode
    model.eval()

    # setup sampling dist
    normal_dist = generate_tools.create_normal_dist(args=args)

    # latent-conditioning info.
    Z_context = normal_dist.sample((n,)).to(args.DEVICE)

    return Z_context



@torch.no_grad()
def diversify_CM_seq(
        args: any,
        model: nn.Module,
        seq_list: list,
        z_context: torch.FloatTensor,
        max_seq_len: int,
        L: int,
        n: int=100
        ) -> (
                torch.FloatTensor,
                torch.FloatTensor,
                torch.FloatTensor,
        ):

            # eval mode
            model.eval()

            # latent conditioning
            X_latent_diversify = diversify_CM_gene(
                    args=args,
                    model=model,
                    seq_list=seq_list,
                    max_seq_len=max_seq_len,
                    z_context=z_context,
                    L=L
            )


            # NO latent conditioning
            X_NOlatent_diversify = diversify_CM_gene(
                    args=args,
                    model=model,
                    seq_list=seq_list,
                    max_seq_len=max_seq_len,
                    z_context=torch.zeros_like(z_context),
                    L=L
            )

            # random diversification
            X_random_diversify = randomly_diversify_CM_gene(
                    args=args,
                    model=model,
                    seq_list=seq_list,
                    max_seq_len=max_seq_len,
                    L=L
            )


            return (
                    X_latent_diversify,
                    X_NOlatent_diversify,
                    X_random_diversify
            )


@torch.no_grad()
def sample_CM_seq(
        args: any,
        model: nn.Module,
        X_context: torch.FloatTensor,
        z_context: torch.FloatTensor,
        max_seq_len: int,
        ) -> (
                torch.FloatTensor,
                torch.FloatTensor,
                torch.FloatTensor,
        ):

            # eval mode
            model.eval()

            # latent conditioning
            X_latent_diversify = diversify_CM_gene(
                    args=args,
                    model=model,
                    seq_list=seq_list,
                    max_seq_len=max_seq_len,
                    z_context=z_context,
                    L=L
            )


            # NO latent conditioning
            X_NOlatent_diversify = diversify_CM_gene(
                    args=args,
                    model=model,
                    seq_list=seq_list,
                    max_seq_len=max_seq_len,
                    z_context=torch.zeros_like(z_context),
                    L=L
            )

            # random diversification
            X_random_diversify = randomly_diversify_CM_gene(
                    args=args,
                    model=model,
                    seq_list=seq_list,
                    max_seq_len=max_seq_len,
                    L=L
            )


            return (
                    X_latent_diversify,
                    X_NOlatent_diversify,
                    X_random_diversify
            )





@torch.no_grad()
def Cterminus_diversify(
        args: any,
        model: nn.Module,
        z_context: torch.FloatTensor,
        max_seq_len: int
    ) -> (
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor
    ):

    # condition on first alpha helix
    Ecoli_WT_1st_helix = ['TSENPLLALREKISALDEKLLALLAERRELAVEVGKAKLL']

    # starting index for C-temrinus diversification
    L = len(Ecoli_WT_1st_helix[0])
    
    delta_gaps = max_seq_len - L # number of gaps needed to add at the end 
    # create seq list
    Ecoli_WT_1st_helix_list = [Ecoli_WT_1st_helix[0] for _ in range(z.shape[0])]
    
    # get sequence tensors
    X_Cterm_diversify, X_Cterm_NOlatent_diversify, X_Cterm_random_diversify = diversify_CM_seq(
            args=args,
            model=model,
            seq_list=Ecoli_WT_1st_helix_list,
            z_context=z_context,
            max_seq_len=max_seq_len,
            L=L,
            n=args.N
    )
    
    # create empty list
    X_NOCterm_diversify = model.sample(
            args=args,
            X_context=torch.zeros_like(X_Cterm_diversify[:,-1,:,:]),
            z=z_context
    )

    return (
            X_Cterm_diversify,
            X_Cterm_NOlatent_diversify,
            X_Cterm_random_diversify,
            X_NOCterm_diversify,
    )



def create_df(
        args: any,
        X: torch.FloatTensor,
        z: torch.FloatTensor,
        output_path: str
    ) -> None:

    # create empty list
    sample_dict = {}
    
    # create sequences
    aa_seqs = util_tools.create_seqs(X=X)
    ids = [f'id_{ii}' for ii in range(len(aa_seqs))]
    
    sample_dict['id'] = ids
    sample_dict['sequence'] = aa_seqs

    for z_axis in range(args.z_dim):

        sample_dict[f'z_{z_axis}'] = list(z[:, z_axis].cpu().numpy())


    sample_df = pd.DataFrame(sample_dict)
    colabfold_df = sample_df[['id', 'sequence']]

    print(sample_df)
    # save dataframes
    sample_df.to_csv(output_path, index=False)
    colabfold_df.to_csv(output_path.replace('.csv','_colabfold.csv'), index=False)
  
    return 

if __name__ == '__main__':

    # get variable arguments
    parser = argparse.ArgumentParser()
    train_sess.get_args(parser)
    generate_tools.sample_get_args(parser)
    CM_train_sess.get_SS_args(parser)
    diversify_args(parser)
    args = parser.parse_args()
    args.alignment = False # only unalignments
    os.makedirs(args.folder_path, exist_ok = True)

    # reprod.
    train_sess.set_SEED(args = args)

    # set GPU (cuda)
    args.DEVICE = train_sess.set_GPU(args=args)

    # load data (get sequence length)
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

    # load weight
    model = generate_tools.load_weights(
            args=args,
            model=PL_model.model
    )
    
    z = sample_z(
            args=args,
            model=model,
            protein_len=protein_len,
            n=args.N
    )

    X_Cterm_div, X_Cterm_NOlatent_div, X_Cterm_random_div, X_NOCterm_div = Cterminus_diversify(
            args=args,
            model=model,
            z_context=z,
            max_seq_len=protein_len
    )

    # Cterminus diversification of CM alpha-helix 
    create_df(
            args=args,
            X=X_Cterm_div,
            z=z,
            output_path=args.Cterminus_output_path
    )

    # Design CM from scratch with the same latent codes
    create_df(
            args=args,
            X=X_NOCterm_div,
            z=z,
            output_path=args.DeNovo_output_path
    )
