import numpy as np
import pandas as pd
import sys
import argparse
from numba import jit
from tqdm import tqdm

import source.pfam_preprocess as pfam_prep
import train_on_CM as CM_train_sess
import train_on_pfam as train_sess
import utils.levenshtein_tools as leven_tools

import torch
import torch.nn as nn

def get_args(parser):


    parser.add_argument('--SEED', dest='SEED', default=42, type=int, help='Flag:random seed')
    parser.add_argument('--design_path', dest='design_path', default=None, type=str, help='Flag: path for the design sequences')
    parser.add_argument('--data_path', dest='data_path', default=None, type=str, help='Flag: path for the dataset sequences')
    parser.add_argument('--synthetic_path', dest='synthetic_path', default=None, type=str, help='Flag: path for synthetic dataset sequences')
    parser.add_argument('--des_seq_column', dest='des_seq_column', default='sequence', type=str, help='Flag: column for the design sequence')
    parser.add_argument('--dataset_seq_column', dest='dataset_seq_column', default='sequence', type=str, help='Flag: column for the natural sequence')
    parser.add_argument('--output_path', dest='output_path', default=None, type=str, help='Flag: save results to output path')
    parser.add_argument('--option', dest='option', default=0, type=int, help='Flag: option 0: compare between train samples | 1: compare design between training samples')

    # load data
    parser.add_argument('--learning_option', dest='learning_option', default='semi-supervised', type=str, help='Flag: learning option')
    parser.add_argument('--alignment', dest='alignment', default=False, type=bool, help='Flag: data preprocess option')
    parser.add_argument('--homolog_option', dest='homolog_option', default=0, type=int, help='Flag: pfam dataset')
    
    # design paths
    parser.add_argument('--design_Cterm_path', dest='design_Cterm_path', default='../outputs/prediction/.', type=str, help='Flag: path to C-terminus designs')
    parser.add_argument('--design_Denovo_path', dest='design_Denovo_path', default='../outputs/prediction/.', type=str, help='Flag: path to De Novo designs')

    # save output path for similarity
    parser.add_argument('--save_Cterm_output_path', dest='save_Cterm_output_path', default='../outputs/prediction/.', type=str, help='Flag: save path for sequence similarity C-terminus designs')
    parser.add_argument('--save_DeNovo_output_path', dest='save_DeNovo_output_path', default='../outputs/prediction/.', type=str, help='Flag: save path for sequence similarity De Novo designs')


    

def compute_hamming_dist(
        seq_A: str,
        seq_B: str
    ) -> int:
    
    hamming_dist = 0 
    for aa_A, aa_B in zip(seq_A, seq_B):
        if aa_A != aa_B:
            hamming_dist += 1

        else: pass

    return hamming_dist

def compute_similarity(
        args: any,
        design_seqs: list,
        train_nat_df: pd.Series,
        max_seq_len: int
    ):
    
    # condition on first alpha helix
    Ecoli_WT_1st_helix = ['TSENPLLALREKISALDEKLLALLAERRELAVEVGKAKLL']

    # starting index for C-temrinus diversification
    Lhelix = len(Ecoli_WT_1st_helix[0])
    
    Ecoli_seq = train_nat_df[train_nat_df['ID to EcCM'] == 1].Unaligned_sequence.values

    Lwt = len(Ecoli_seq[0])
    
    Ecoli_seq_full = Ecoli_seq[0] + '-' * (max_seq_len - Lwt)

    # dict containing seq sim. 
    seq_sim_dict = {}
    
    full_ham_dist_list, Nterm_ham_dist_list, Cterm_ham_dist_list = [], [], []

    for design_seq in design_seqs:
        
        Ldesign = len(design_seq)
        design_seq = design_seq + '-' * (max_seq_len - Ldesign)

        full_ham_dist_list.append(
                compute_hamming_dist(
                    seq_A=Ecoli_seq_full,
                    seq_B=design_seq
            )
        )
        
        Nterm_ham_dist_list.append(
                compute_hamming_dist(
                    seq_A=Ecoli_seq_full[:Lhelix],
                    seq_B=design_seq[:Lhelix]
                )
        )

        
        Cterm_ham_dist_list.append(
                compute_hamming_dist(
                    seq_A=Ecoli_seq_full[Lhelix:],
                    seq_B=design_seq[Lhelix:]
                )
        )

    
    # id headers
    seq_sim_dict['id'] = [f'id_{ii}' for ii in range(len(design_seqs))]

    # full length sequence
    seq_sim_dict['hamming_dist[full]'] = full_ham_dist_list
    seq_sim_dict['seq_sim[full]'] = [1 - dist/max_seq_len for dist in full_ham_dist_list]
    
    # Nterm condition sequence
    seq_sim_dict['hamming_dist[Nterm]'] = Nterm_ham_dist_list
    seq_sim_dict['seq_sim[Nterm]'] = [1 - dist/Lhelix for dist in Nterm_ham_dist_list]
    
    # Cterm design sqeuence
    seq_sim_dict['hamming_dist[Cterm]'] = Cterm_ham_dist_list
    seq_sim_dict['seq_sim[Cterm]'] = [1 - dist/(max_seq_len - Lhelix) for dist in Cterm_ham_dist_list]
        
    seq_sim_df = pd.DataFrame(seq_sim_dict)

    return seq_sim_df


def get_CM_data(args: any) -> (
        pd.Series,
        pd.Series,
        int
    ):

    design_df = pd.read_csv(args.design_path) # novel sampled sequences
    train_nat_df = pd.read_csv(args.data_path) # dataset used for training
    train_nat_df['Unaligned_sequence'] = [seq.replace('-','') for seq in train_nat_df.Sequence]
    test_synthetic_df = pd.read_csv(args.synthetic_path) # datased use for testing
    test_synthetic_df['Unaligned_sequence'] = [seq.replace('-','') for seq in test_synthetic_df.Sequence]

    all_df = pd.concat((train_nat_df, test_synthetic_df))
   
    # prepare natural homologs as training data
    train_num_X, train_OH_X, train_y = pfam_prep.prepare_CM_dataset(
            data_path=args.data_path,
            alignment=args.alignment
    )
    
    # max training sequence length
    max_seq_len = train_OH_X.shape[1]


    return (
            train_nat_df,
            test_synthetic_df,
            max_seq_len
    )

def get_design_dfs(args: any) -> (
        pd.Series,
        pd.Series
    ):

    Cterm_design_df = pd.read_csv(args.design_Cterm_path)
    DeNovo_design_df = pd.read_csv(args.design_Denovo_path)

    return (
            Cterm_design_df,
            DeNovo_design_df
    )


if __name__ == '__main__':
    
    # write input argument variables
    parser = argparse.ArgumentParser()
    get_args(parser=parser)
    args = parser.parse_args()
    args.alignment=False
    
    # reprod
    train_sess.set_SEED(args=args)

    # compute min levenshteins and save results ...
    train_nat_df, _, max_seq_len = get_CM_data(args=args)

    Cterm_CM_df, DeNovo_CM_df = get_design_dfs(
            args=args
    )

    Cterm_seq_dim_df = compute_similarity(
            args=args,
            design_seqs=list(Cterm_CM_df.sequence),
            train_nat_df=train_nat_df,
            max_seq_len=max_seq_len
    )

    DeNovo_seq_dim_df = compute_similarity(
            args=args,
            design_seqs=list(DeNovo_CM_df.sequence),
            train_nat_df=train_nat_df,
            max_seq_len=max_seq_len
    )

    Cterm_seq_dim_df.to_csv(args.save_Cterm_output_path, index=False)
    DeNovo_seq_dim_df.to_csv(args.save_DeNovo_output_path, index=False)

