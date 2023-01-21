"""

author: Niksa Praljak
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import sys
import argparse
from tqdm import tqdm

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.autograd import Variable
from torch.nn import functional as F

import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import CSVLogger


import source.pfam_preprocess as pfam_prep

from sklearn.model_selection import train_test_split



def load_data(
        homolog_option,
        batch_size,
        alignment = False
    ):

    # load hyperparameters

    if homolog_option == 0:
       # CM homologs
    
       # prepare natural homologs as training data
       train_num_X, train_OH_X, train_y = pfam_prep.prepare_CM_dataset(
                 data_path = './data/protein_families/CM/CM_natural_homologs.csv',
                 alignment = alignment
       )

       # prepare design homologs as testing data
       test_num_X, test_OH_X, test_y = pfam_prep.prepare_CM_dataset(
                 data_path = './data/protein_families/CM/CM_synthetic_homologs.csv',
                 alignment = alignment
       )
      
           
       # train dataset:
       train_dataset = pfam_prep.CM_dataset(
                                    num_inputs = train_num_X,
                                    onehot_inputs = train_OH_X,
                                    pheno_outputs = train_y
       )

       train_dataloader = DataLoader(
                                dataset = train_dataset,
                                batch_size = batch_size,
                                num_workers = 4,
                                shuffle = True
       )


       # test dataset (for now):
       valid_dataset = pfam_prep.CM_dataset(
                                    num_inputs = test_num_X,
                                    onehot_inputs = test_OH_X,
                                    pheno_outputs = test_y
       )

       valid_dataloader = DataLoader(
                                dataset = valid_dataset,
                                batch_size = batch_size,
                                num_workers = 4,
                                shuffle = False
       )

       test_dataset = None
       test_dataloader = None


    elif homolog_option == 1:
        # S1A serine protease family:
     
       # prepare natural homologs as training data
       train_num_X, train_OH_X, _, _, _, _, _, _, _ = pfam_prep.prepare_S1A_dataset(
                 data_path = './data/protein_families/S1A/pfam_S1A.csv',
                 alignment = alignment
       )
     
       # split dataset into train/valid:
       train_num_X, valid_num_X = train_test_split(
                                            train_num_X,
                                            test_size = 0.2,
                                            random_state = 42
       )       

       # split dataset into train/valid:
       train_OH_X, valid_OH_X = train_test_split(
                                            train_OH_X,
                                            test_size = 0.2,
                                            random_state = 42
       )       


       # train dataset:
       train_dataset = pfam_prep.S1A_dataset(
                                    num_inputs = train_num_X,
                                    onehot_inputs = train_OH_X,
       )

       train_dataloader = DataLoader(
                                dataset = train_dataset,
                                batch_size = batch_size,
                                num_workers = 4,
                                shuffle = True
       )


       # valid dataset:
       valid_dataset = pfam_prep.S1A_dataset(
                                    num_inputs = valid_num_X,
                                    onehot_inputs = valid_OH_X,
       )

       valid_dataloader = DataLoader(
                                dataset = valid_dataset,
                                batch_size = batch_size,
                                num_workers = 4,
                                shuffle = False
       )

       test_dataloader = None

    elif homolog_option == 2:
        # lactamase protein family:
     
       # prepare natural homologs as training data
       train_num_X, train_OH_X = pfam_prep.prepare_lactamase_dataset(
                 data_path = './data/protein_families/lactamase/pfam_lactamase.csv',
                 alignment = alignment
       )
     
       # split dataset into train/valid:
       train_num_X, valid_num_X = train_test_split(
                                            train_num_X,
                                            test_size = 0.2,
                                            random_state = 42
       )       

       # split dataset into train/valid:
       train_OH_X, valid_OH_X = train_test_split(
                                            train_OH_X,
                                            test_size = 0.2,
                                            random_state = 42
       )       


       # train dataset:
       train_dataset = pfam_prep.lactamase_dataset(
                                    num_inputs = train_num_X,
                                    onehot_inputs = train_OH_X,
       )

       train_dataloader = DataLoader(
                                dataset = train_dataset,
                                batch_size = batch_size,
                                num_workers = 4,
                                shuffle = True
       )


       # valid dataset:
       valid_dataset = pfam_prep.lactamase_dataset(
                                    num_inputs = valid_num_X,
                                    onehot_inputs = valid_OH_X,
       )

       valid_dataloader = DataLoader(
                                dataset = valid_dataset,
                                batch_size = batch_size,
                                num_workers = 4,
                                shuffle = False
       )

       test_dataloader = None

    elif homolog_option == 3:
        # G-protein family:
      
       # prepare natural homologs as training data
       train_num_X, train_OH_X = pfam_prep.prepare_lactamase_dataset(
                 data_path = './data/protein_families/G_protein/pfam_G_protein.csv',
                 alignment = alignment
       )
     
       # split dataset into train/valid:
       train_num_X, valid_num_X = train_test_split(
                                            train_num_X,
                                            test_size = 0.2,
                                            random_state = 42
       )       

       # split dataset into train/valid:
       train_OH_X, valid_OH_X = train_test_split(
                                            train_OH_X,
                                            test_size = 0.2,
                                            random_state = 42
       )       


       # train dataset:
       train_dataset = pfam_prep.Gprotein_dataset(
                                    num_inputs = train_num_X,
                                    onehot_inputs = train_OH_X,
       )

       train_dataloader = DataLoader(
                                dataset = train_dataset,
                                batch_size = batch_size,
                                num_workers = 4,
                                shuffle = True
       )


       # valid dataset:
       valid_dataset = pfam_prep.Gprotein_dataset(
                                    num_inputs = valid_num_X,
                                    onehot_inputs = valid_OH_X,
       )

       valid_dataloader = DataLoader(
                                dataset = valid_dataset,
                                batch_size = batch_size,
                                num_workers = 4,
                                shuffle = False
       )

       test_dataloader = None

    elif homolog_option == 4:
        # DHFR protein family:
       
       # prepare natural homologs as training data
       train_num_X, train_OH_X = pfam_prep.prepare_DHFR_dataset(
                 data_path = './data/protein_families/DHFR/pfam_DHFR.csv',
                 alignment = alignment
       )
     
       # split dataset into train/valid:
       train_num_X, valid_num_X = train_test_split(
                                            train_num_X,
                                            test_size = 0.2,
                                            random_state = 42
       )       

       # split dataset into train/valid:
       train_OH_X, valid_OH_X = train_test_split(
                                            train_OH_X,
                                            test_size = 0.2,
                                            random_state = 42
       )       


       # train dataset:
       train_dataset = pfam_prep.Gprotein_dataset(
                                    num_inputs = train_num_X,
                                    onehot_inputs = train_OH_X,
       )

       train_dataloader = DataLoader(
                                dataset = train_dataset,
                                batch_size = batch_size,
                                num_workers = 4,
                                shuffle = True
       )


       # valid dataset:
       valid_dataset = pfam_prep.Gprotein_dataset(
                                    num_inputs = valid_num_X,
                                    onehot_inputs = valid_OH_X,
       )

       valid_dataloader = DataLoader(
                                dataset = valid_dataset,
                                batch_size = batch_size,
                                num_workers = 4,
                                shuffle = False
       )

       test_dataloader = None
    else:
        pass

    _, protein_len, _ = train_OH_X.shape

    return train_dataloader, valid_dataloader, test_dataloader, protein_len


