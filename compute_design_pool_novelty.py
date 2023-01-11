
import numpy as np
import pandas as pd
import sys
import argparse
from numba import jit
from tqdm import tqdm
import utils.compute_min_levenshtein as utils_min_leven
import random
import os




def get_arguments() -> any:

    parser = argparse.ArgumentParser()

    parser.add_argument('--SEED', default=42, type=int, help='flag: random seed')
    parser.add_argument('--dataset_path', default='./data/ACS_SynBio_SH3_dataset.csv', type=str, help='flag: path for the dataset')
    parser.add_argument('--output_results_path', default='outputs/min_leven/',  type=str, help='flag: path for the results')
    parser.add_argument('--option', default='design', type=str, help='flag: choose whether to measure leven. distances for design or natural.')
    parser.add_argument('--file_path', default='./outputs/SH3_design_pool/', type=str, help='flag: choose dataset path.')
    parser.add_argument('--output_df_path', default='./outputs/SH3_design_pool/min_leven', type=str, help='flag: choose output path for containing min leven measurements.')


    args = parser.parse_args()

    return args


def set_SEED(args:any) -> None:


    random.seed(args.SEED)
    np.random.seed(args.SEED)

    return


def load_data(args:any) -> (
        pd.Series,
        pd.Series
    ):

    df = pd.read_csv(args.dataset_path)

    nat_df = df[df.header.str.contains('nat_')] # natural homologs
    design_df = df[~df.header.str.contains('nat_')] # VAE synthetic homologs

    return (
            nat_df,
            design_df,
    )

def load_design_data(
        args: any, 
        filename: str,
    ) -> pd.Series:

    csv_path = args.file_path + filename
    
    df = pd.read_csv(csv_path)

    return df


def compute_dists(
        target_list: list,
        ref_list: list=[]
    ) -> (
            list,
            list
    ):

        # measure novelty of the synthetics
        min_leven, perc_min_leven = utils_min_leven.compute_design_leven(
                                                                design_pool=target_list,
                                                                train_pool=ref_list
        )

        return (
                min_leven,
                perc_min_leven
        )




def collect_levenshteins(
        args: any,
        target_seqs: list,
        nat_df: pd.Series,
        design_df: pd.Series
    ) -> pd.Series:

    # get sequences (training dataset)
    nat_seqs = list(nat_df.Sequences_unaligned)
    design_seqs = list(design_df.Sequences_unaligned)

    # min levenshtein distances referenced from the natural dataset
    min_leven_from_nat, perc_min_leven_from_nat = compute_dists(
                                            target_list=target_seqs,
                                            ref_list=nat_seqs
    )
    
    # min levenshtein distances referenced from the whole training dataset
    min_leven_from_all, perc_min_leven_from_all = compute_dists(
                                            target_list=target_seqs,
                                            ref_list=nat_seqs+design_seqs
    )

       
    # create dict. for containing novelty measurements
    novel_dict = {}
    
    novel_dict['header'] = [f'seq_{ii}' for ii, _ in enumerate(target_seqs)]
    novel_dict['min_leven[from=nat]'] = min_leven_from_nat
    novel_dict['perc_min_leven[from=nat]'] = perc_min_leven_from_nat
    novel_dict['min_leven[from=all]'] = min_leven_from_all
    novel_dict['perc_min_leven[from=all]'] = perc_min_leven_from_all

    novel_df = pd.DataFrame(novel_dict)

    return novel_df

def save_results(
        args: any,
        df: pd.Series,
        output_path: str
    ) -> None:


    df.to_csv(output_path)

    return

def get_filenames(args:any) -> list:

    filename_list = []
    for filename in os.listdir(args.file_path):

        if filename.endswith(".csv"):
            
            filename_list.append(filename)
        
    for ref_filename in [
                        'PartialRescueParalog.csv',
                        'Sho1Ortholog_diversify.csv',
                        ]:

        filename_list.remove(ref_filename)

    return filename_list

if __name__ == '__main__':

    # prepare script
    args = get_arguments()
    set_SEED(args=args)

    os.makedirs(args.output_df_path, exist_ok=True)

    # load data
    train_nat_df, train_design_df = load_data(args=args)

    # get the list of filenames that contain design spreadsheets
    filename_list = get_filenames(args=args)
    print('list:', filename_list)

    for filename in filename_list:

        # load design sequences
        design_df = load_design_data(
                            args=args,
                            filename=filename
        )
        design_list = list(design_df.unaligned_sequence)

        # get min. leven distances
        min_leven_design_df = collect_levenshteins(
                args=args,
                target_seqs=design_list,
                nat_df=train_nat_df,
                design_df=train_design_df
        )
        

        # save dataframe
        output_df_path = args.output_df_path + '/[novelty]' + filename
        save_results(
                args=args,
                df=min_leven_design_df,
                output_path=output_df_path
        )
        



