import numpy as np
import pandas as pd
import sys
import argparse
from numba import jit
from tqdm import tqdm

import train_on_pfam as train_sess
import utils.levenshtein_tools as leven_tools

def get_args(parser):


    parser.add_argument('--SEED', dest='SEED', default=42, type=int, help='Flag:random seed')
    parser.add_argument('--design_path', dest='design_path', default=None, type=str, help='Flag: path for the design sequences')
    parser.add_argument('--data_path', dest='data_path', default=None, type=str, help='Flag: path for the dataset sequences')
    parser.add_argument('--synthetic_path', dest='synthetic_path', default=None, type=str, help='Flag: path for synthetic dataset sequences')
    parser.add_argument('--des_seq_column', dest='des_seq_column', default='sequence', type=str, help='Flag: column for the design sequence')
    parser.add_argument('--dataset_seq_column', dest='dataset_seq_column', default='sequence', type=str, help='Flag: column for the natural sequence')
    parser.add_argument('--output_path', dest='output_path', default=None, type=str, help='Flag: save results to output path')
    parser.add_argument('--option', dest='option', default=0, type=int, help='Flag: option 0: compare between train samples | 1: compare design between training samples')


def compute_leven(args: any):

    design_df = pd.read_csv(args.design_path) # novel sampled sequences
    train_nat_df = pd.read_csv(args.data_path) # dataset used for training
    train_nat_df['Unaligned_sequence'] = [seq.replace('-','') for seq in train_nat_df.Sequence]
    test_synthetic_df = pd.read_csv(args.synthetic_path) # datased use for testing
    test_synthetic_df['Unaligned_sequence'] = [seq.replace('-','') for seq in test_synthetic_df.Sequence]

    all_df = pd.concat((train_nat_df, test_synthetic_df))

    print(f'Size of the training set = {train_nat_df.shape[0]} | testing set = {test_synthetic_df.shape[0]}| all set = {all_df.shape[0]} ')
    # compute min levenshtein distance between samples
    if args.option == 0:

        print('Entered option 0')
        train_df['min_leven'], train_df['perc_min_leven'] = leven_tools.compute_train_leven(
                                    pool1=list(train_df[arg.dataset_seq_column].values[:])
        )

        train_df.to_csv(args.output_path, index = False)

    elif args.option == 1:

        print('Entered option 1')

        design_df['min_leven[from=nat]'], design_df['perc_min_leven[from=nat]'] = leven_tools.compute_design_leven(
                            design_pool=list(design_df[args.des_seq_column].values[:]),
                            train_pool=list(train_nat_df[args.dataset_seq_column].values[:])
        )


        design_df['min_leven[from=all]'], design_df['perc_min_leven[from=all]'] = leven_tools.compute_design_leven(
                            design_pool=list(design_df[args.des_seq_column].values[:]),
                            train_pool=list(all_df[args.dataset_seq_column].values[:])
        )


        design_df.to_csv(args.output_path, index = False)


    print('Finished...')

    return



if __name__ == '__main__':
    
    # write input argument variables
    parser = argparse.ArgumentParser()
    get_args(parser=parser)
    args = parser.parse_args()

    # reprod
    train_sess.set_SEED(args=args)

    # compute min levenshteins and save results ...
    compute_leven(args=args)

