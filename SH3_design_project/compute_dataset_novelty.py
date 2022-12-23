
import numpy as np
import pandas as pd
import sys
import argparse
from numba import jit
from tqdm import tqdm
import utils.compute_min_levenshtein as min_leven
import random


def get_arguments() -> any:
   
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--SEED', default=42, type=int, help='flag: random seed')
    parser.add_argument('--dataset_path', default='./data/ACS_SynBIO_SH3_dataset.csv', type=str, help='flag: path for the dataset')
    parser.add_argument('--output_results_path', default='outputs/min_leven/',  type=str, help='flag: path for the results')
    parser.add_argument('--option', default='design', type=str, help='flag: choose whether to measure leven. distances for design or natural.')

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


def compute_dists(
        design_list: list,
        nat_list: list=[],
        option: str='design'
    ) -> (
            list,
            list
    ):
    
    if option=='design':
        # measure novelty of the synthetics
        design_min_leven, design_perc_min_leven = min_leven.compute_design_leven(
                                                design_pool=design_list,
                                                train_pool=nat_list
        )
        return (
                design_min_leven,
                design_perc_min_leven
        )

    else:
        # collect natural ssequence novelty
        nat_min_leven, nat_perc_min_leven = min_leven.natural_min_leven(seq_pool=nat_list)
        return (
                nat_min_leven,
                nat_perc_min_leven
        )



def collect_levenshteins(
        args: any,
        nat_df: pd.Series,
        design_df: pd.Series,
        ) -> pd.Series:
    
    # get sequences
    nat_seqs = list(nat_df.Sequences_unaligned)
    design_seqs = list(design_df.Sequences_unaligned)
    
    min_leven, perc_min_leven = compute_dists(
                                    design_list=design_seqs,
                                    nat_list=nat_seqs,
                                    option=args.option
    )
                                    
    result_dict = {}
    # collect levenshtein estimates

    if args.option == 'design':
        result_dict['header'] = list(design_df.header)
    else:
        result_dict['header'] = list(nat_df.header)

    result_dict['min_leven'] = min_leven
    result_dict['perc_min_leven[dissimilarity]'] = perc_min_leven
    
    result_df = pd.DataFrame(result_dict)

    return result_df


def save_results(
        args: any,
        df: pd.Series
    ) -> None:
    
    df[['header', 'min_leven', 'perc_min_leven[dissimilarity]']].to_csv(args.output_results_path)

    return


if __name__ == '__main__':

    # prepare script
    args = get_arguments()
    set_SEED(args=args)

    # load data
    nat_df, design_df = load_data(args=args) 
    
    # get dataframe containing min levenhstein distances
    final_df = collect_levenshteins(
                args=args,
                nat_df=nat_df,
                design_df=design_df
    )
    
    # save results
    save_results(
            args=args,
            df=final_df
    )

