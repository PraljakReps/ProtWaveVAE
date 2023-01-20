
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



def return_ref_seqs(
        args: any
    ) -> (
            list,
            list,
            list,
            list
    ):
        WT_seq = ['DNFIYKAKALYPYDADDDDAYEISFEQNEILQVSDIEGRWWKARRANGETGIIPSNYVQLIDGPEE']
        Partial_paralog_seq = ['NKILFYVEAMYDYTATIEEEFNFQAGDIIAVTDIPDDGWWSGELLDEARREEGRHVFPSNFVRLF']
        Nonfunc_paralog_seq = ['PKENPWATAEYDYDAAEDNELTFVENDKIINIEFVDDDWWLGELEKDGSKGLFPSNYVSLGN']
        Ortholog_seq = ['GVYMHRVKAVYSYKANPEDPTELTFEKGDTLEVVDIQGKWWQARQVKADGQTNIGIVPSNYMQVI']

        return (
                WT_seq,
                Partial_paralog_seq,
                Nonfunc_paralog_seq,
                Ortholog_seq
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
    
    # get reference sequeences
    WT_seq, Partial_paralog_seq, Nonfunc_paralog_seq, Ortholog_seq = return_ref_seqs(args=args)

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

    # min leven. distances referenced to sequence of interest
    min_leven_from_WT, perc_min_leven_from_WT = compute_dists(
                                            target_list=target_seqs,
                                            ref_list=WT_seq
    )
    min_leven_from_partial, perc_min_leven_from_partial = compute_dists(
                                            target_list=target_seqs,
                                            ref_list=Partial_paralog_seq
    )
    min_leven_from_paralog, perc_min_leven_from_paralog = compute_dists(
                                            target_list=target_seqs,
                                            ref_list=Nonfunc_paralog_seq
    )
    min_leven_from_ortholog, perc_min_leven_from_ortholog = compute_dists(
                                            target_list=target_seqs,
                                            ref_list=Ortholog_seq
    )
       
    # create dict. for containing novelty measurements
    novel_dict = {}
    
    novel_dict['header_2'] = [f'seq_{ii}' for ii, _ in enumerate(target_seqs)]
    novel_dict['min_leven[from=nat]'] = min_leven_from_nat
    novel_dict['perc_min_leven[from=nat]'] = perc_min_leven_from_nat
    novel_dict['min_leven[from=all]'] = min_leven_from_all
    novel_dict['perc_min_leven[from=all]'] = perc_min_leven_from_all
    novel_dict['min_leven[from=WT]'] = min_leven_from_WT
    novel_dict['perc_min_leven[from=WT]'] = perc_min_leven_from_WT
    novel_dict['min_leven[from=Ortholog]'] = min_leven_from_ortholog
    novel_dict['perc_min_leven[from=Ortholog]'] = perc_min_leven_from_ortholog
    novel_dict['min_leven[from=Partial]'] = min_leven_from_partial
    novel_dict['perc_min_leven[from=Partial]'] = perc_min_leven_from_partial
    novel_dict['min_leven[from=Paralog]'] = min_leven_from_paralog
    novel_dict['perc_min_leven[from=Paralog]'] = perc_min_leven_from_paralog

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
                        'NonfuncParalog.csv'
                        ]:
        
        try:
            filename_list.remove(ref_filename)

        except ValueError:
            pass

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
        
        min_leven_design_df = pd.concat((design_df, min_leven_design_df), axis = 1)

        # save dataframe
        output_df_path = args.output_df_path + '/[novelty]' + filename
        save_results(
                args=args,
                df=min_leven_design_df,
                output_path=output_df_path
        )
        



