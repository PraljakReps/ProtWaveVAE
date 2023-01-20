"""
code description: compute the minimum levenshtein distance between a BO design sequence and the corresponding training dataset.

"""

import numpy as np
import pandas as pd
import sys
import argparse
from numba import jit
from tqdm import tqdm

@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit 
def levenshteinDistanceDP(token1, token2):

    """
    By Ahmed Fawzy Gas on PaperspaceBlog, titled blog: 
    Implementing the Levenshtein Distance for Word Autocomplementation and Autocorrection.
    """

    distances = np.zeros((len(token1) + 1, len(token2) + 1))

    for t1 in range(len(token1) + 1):
        distances[t1][0] = t1

    for t2 in range(len(token2) + 1):
        distances[0][t2] = t2

    a = 0
    b = 0
    c = 0

    for t1 in range(1, len(token1) + 1):
        for t2 in range(1, len(token2) + 1):
            
            if (token1[t1-1] == token2[t2-1]):
                distances[t1][t2] = distances[t1 - 1][t2 - 1]

            else:
            
                a = distances[t1][t2 - 1]
                b = distances[t1 - 1][t2]
                c = distances[t1 - 1][t2 - 1]

                if (a <= b and a <= c):
                    distances[t1][t2] = a + 1
                elif (b <= a and b <= c):
                    distances[t1][t2] = b + 1
                else:
                    distances[t1][t2] = c + 1

    return distances[len(token1)][len(token2)]



def finding_nat_indices(
                seq_list: list,
                index: int
    ) -> list:
    """
    To compare against everything else, we need to remove the sequence of interest within the list. 
    """
    compare_seq_list = seq_list[0:index] + seq_list[index+1:]
    return compare_seq_list


def natural_min_leven(seq_pool: list) -> (
            list,
            list
    ):
    
    leven_dist_pool, perc_dissim_pool = [], []
    
    for idx, seq in tqdm(enumerate(seq_pool)):
        
        temp_leven_dists = []
        
        ref_pool = finding_nat_indices(seq_list=seq_pool, index=idx)

        for ref_seq in ref_pool:
            
            # compute levenshtein distance
            distance = levenshteinDistanceDP(token1=seq,token2=ref_seq)
            # allocate distances
            temp_leven_dists.append(distance)
            

        target_seq_len = len(seq_pool) # length of the sequence of interest
        ref_seq_len = len(ref_pool[np.argmin(temp_leven_dists)].replace('-','')) # length of the remaining sequences
        max_seq_len = max(ref_seq_len,target_seq_len) # take the maximum length sequence
        
        # allocate all of the sequences
        leven_dist_pool.append( min(temp_leven_dists) )
        perc_dissim_pool.append( min(temp_leven_dists) / max_seq_len )
        
    return leven_dist_pool, perc_dissim_pool


def compute_design_leven(
    design_pool: list,
    train_pool: list
) -> list:
    
    leven_distance_pool = []
    perc_dissim_pool = []

    for design_seq in tqdm(design_pool):

        leven_distances = []
        
        for train_seq in train_pool:
            
            distance = levenshteinDistanceDP(
                token1 = design_seq.replace('-',''),
                token2 = train_seq.replace('-','')
            )


            leven_distances.append(distance)

        # compute the maximum protein length between the minimum levenshtein distance pair
        ref_seq_len = len(train_pool[np.argmin(leven_distances)].replace('-',''))
        target_seq_len = len(design_seq)
        max_seq_len = max(ref_seq_len, target_seq_len)

        leven_distance_pool.append( min(leven_distances) )
        perc_dissim_pool.append(  min(leven_distances) / max_seq_len )

    return leven_distance_pool, perc_dissim_pool




