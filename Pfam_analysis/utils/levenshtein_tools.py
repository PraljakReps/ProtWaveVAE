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

def compute_train_leven(
    pool1: list
) -> (list, list):

    # function description: compute min. levenshtein dsitance amongst the training samples

    leven_distance_pool = []
    perc_dissim_pool = []
    for seq in tqdm(pool1):
        
        # copy training seqs and remove the train sample of interest
        temp_pool = pool1.copy()
        temp_pool.remove(seq)

        leven_distances = []

        for other_train_seq in tqdm(temp_pool):
            
            distance = levenshteinDistanceDP(
                token1 = seq.replace('-',''),
                token2 = other_train_seq.replace('-','')
            )

            leven_distances.append(distance)
    
       
        # compute the maximum protein length between the minimum levenshtein distance pair
        target_seq_len = len(temp_pool[np.argmin(leven_distances)].replace('-',''))
        ref_seq_len = len(seq)
        max_seq_len = max(ref_seq_len, target_seq_len)

        leven_distance_pool.append( min(leven_distances) )
        perc_dissim_pool.append( max_seq_len)

    return leven_distance_pool, perc_dissim_pool


