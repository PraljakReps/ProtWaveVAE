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
import train_ProtWaveVAE as ProtWaveVAE
import compute_design_pool_novelty as compute_pool_novelty
import generate_proteins as gen_proteins
import utils.tools as util_tools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import sys
import argparse
from tqdm import tqdm
import random
import os



def get_args() -> any:

    # write output path name
    parser = argparse.ArgumentParser()
    
    # path varibles
    parser.add_argument('--dataset_path', default='./data/ACS_SynBio_SH3_dataset.csv')
    parser.add_argument('--output_results_path', default='./outputs/SH3_task/ProtWaveVAE_SSTrainingHist.csv')
    parser.add_argument('--output_model_path', default='./outputs/SH3_task/ProtWaveVAE_SSTrainingHist.pth')
    parser.add_argument('--save_dir', default='./outputs/SH3_design_pool')
       
    # paths for loading design dataframes
    parser.add_argument('--file_path', default='./outputs/')
    parser.add_argument('--L', default='16|33|50')
    parser.add_argument('--filenames', default='L=16|L=33|L=50')
    parser.add_argument('--leven_column', default='')
    parser.add_argument('--output_path', default='')
    parser.add_argument('--reference', default='WT')

    # model training variables
    parser.add_argument('--SEED', default=42, type=int, help='Random seed')
    parser.add_argument('--batch_size', default=512, type=int, help='Size of the batch.')
    parser.add_argument('--epochs', default=1000, type=int, help='Number of epochs')
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--DEVICE', default='cuda', help='Learning rate')
    parser.add_argument('--dataset_split', default=1, type=int, help='Choose whether to split into train/valid sets')
    parser.add_argument('--N', default=100, type=int, help='Batch size')

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


def ref_seq(args:any) -> str:

    if args.reference == 'WT':
        WT_seq = 'DNFIYKAKALYPYDADDDDAYEISFEQNEILQVSDIEGRWWKARRANGETGIIPSNYVQLIDGPEE'
        seq = WT_seq

    elif args.reference == 'PARTIAL':
        partial_seq = 'NKILFYVEAMYDYTATIEEEFNFQAGDIIAVTDIPDDGWWSGELLDEARREEGRHVFPSNFVRLF'
        seq = partial_seq
    
    elif args.reference == 'PARALOG':
        paralog_seq = 'PKENPWATAEYDYDAAEDNELTFVENDKIINIEFVDDDWWLGELEKDGSKGLFPSNYVSLGN'
        seq = paralog_seq

    elif args.reference == 'ORTHOLOG':
        ortholog_seq = 'GVYMHRVKAVYSYKANPEDPTELTFEKGDTLEVVDIQGKWWQARQVKADGQTNIGIVPSNYMQVI'
        seq = ortholog_seq
    
    return seq

def guided_random_diver_gene(
    args: any,
    model: nn.Module,
    seq_list: list,
    max_seq_len: int,
    L: int,
    min_leven_dists: list,
    ) -> (
    torch.FloatTensor,
    torch.FloatTensor,
    list,
    list):
   
    # eval mode: 
    model.eval()
    
    # useful parameters
    seq_ref = ref_seq(args=args)
    print(seq_ref)
    seq_len = len(seq_ref) # sequence length with no gaps
    num_gaps = max_seq_len - seq_len # gap region
    design_seq_lens = [len(seq) for seq in seq_list] # length of each design sequence
    ref_seq_list = [seq_ref for _ in range(len(seq_list))] # copy reference sequence until size of the design pool is reached
    # create torch tensor
    X_ref = gen_proteins.convert_list_to_tensor(
            seq_list = ref_seq_list,
            max_seq_len=max_seq_len
    )

    X_design = gen_proteins.convert_list_to_tensor(
                seq_list=seq_list,
                max_seq_len=max_seq_len
    )
    

   # X_temp = model.create_uniform_tensor(args=args,X=X)
   
    print(f'Seq length: {seq_len} | num gaps: {num_gaps}')
    # diversify sequence of interest
    X_rand_diversify_samples = model.guided_randomly_diversify(
                                            args=args,
                                            X_context=X_ref.to(args.DEVICE),
                                            X_design=X_design.to(args.DEVICE),
                                            L=L,
                                            min_leven_dists=min_leven_dists,
                                            option='guided',
                                            design_seq_lens=design_seq_lens,
                                            ref_seq_len=seq_len,
                                            num_gaps=num_gaps
    ).cpu()
    
    # include deletion gaps
    #X_rand_diversify_samples[:, -1, -num_gaps:,:] = X[:,-num_gaps:, :]
    
    

    # amino acid sequences
    seq_rand_diversify = gen_proteins.create_seqs(X=X_rand_diversify_samples)
    # get lengths for each mutant
    seq_lengths = [len(seq) <= 66 for seq in seq_rand_diversify]
    
    # add padded tokens to allow for same size length sequences
    pad_ref_seq_list = [seq + (max_seq_len-len(seq)) * '-' for seq in ref_seq_list]
    pad_rand_seq_list = [seq + (max_seq_len-len(seq)) * '-' for seq in seq_rand_diversify]
    
    #print(pad_ref_seq_list[0], pad_rand_seq_list[0])
    
    

    # hamming distance and similarity
    hamming_dists, similarity = util_tools.compute_hamming_dist(
        seq1_list=pad_ref_seq_list,
        seq2_list=pad_rand_seq_list
    )
    

    return (
        X_rand_diversify_samples,
        X_design,
        seq_rand_diversify,
        hamming_dists,
        similarity
    )




def create_df(
        args: any,
        seq: list,
        hamming: list,
        similarity: list
    ) -> pd.Series:


    random_seq_dict = {}
    
    # create dataframe
    random_seq_dict['header'] = [f'seq_{ii}' for ii in range(len(seq))]
    random_seq_dict['unaligned_sequence'] = seq
    random_seq_dict['hamming'] = hamming
    random_seq_dict['similarity'] = similarity

    df = pd.DataFrame(random_seq_dict)

    return df

if __name__ == '__main__':

    args = get_args()
    os.makedirs(args.output_path, exist_ok=True)

    ProtWaveVAE.set_GPU() # set GPU
    ProtWaveVAE.set_SEED(args=args) # set SEED (reproducibility)
    
    # get data
    train_dataloader, valid_dataloader, protein_len = ProtWaveVAE.get_data(args=args)
    train_dataset = train_dataloader.dataset
    valid_dataset = valid_dataloader.dataset
    Xtrain, _, _, _ = train_dataset[0:1]
    max_seq_len = Xtrain.shape[1] # (B, L, 21)
    args.max_seq_len = max_seq_len 
    
    # get model
    PL_model = ProtWaveVAE.get_model(
                            args=args,
                            protein_len=protein_len
    ).to(args.DEVICE)
    model = PL_model.model
    model.load_state_dict(torch.load(args.output_model_path))
    
        
    filenames = args.filenames.split('|')
    L = [int(l) for l in args.L.split('|')]
    

    for (l, filename) in zip(L, filenames):


        design_df = compute_pool_novelty.load_design_data(
                            args=args,
                            filename=filename
        )

        seq_list = [seq for seq in design_df.unaligned_sequence.values]
        min_leven_dists = list(design_df[args.leven_column].values)
        print('Starting position:', l)

        X_rand, X, seq_rand, hamming, similarity = guided_random_diver_gene(
                args=args,
                model=model,
                seq_list=seq_list,
                max_seq_len=max_seq_len,
                L=l,
                min_leven_dists=min_leven_dists
        )
        
        df = create_df(
                args=args,
                seq=seq_rand,
                hamming=hamming,
                similarity=similarity
        )
       
        df.to_csv(args.output_path + '/'+ filename.replace('[novelty]', '[random]'), index=False)
