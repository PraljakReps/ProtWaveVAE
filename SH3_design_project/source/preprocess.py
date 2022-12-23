"""
@author: Niksa Praljak
"""



import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.autograd import Variable


import numpy as np
import pandas as pd
from numba import jit


' ___________ Preprocess for Wavenet-based autoregressive decoder ___________'




@jit(nopython=True)
def pad_ends(
        seqs:list,
        max_seq_length:int
    ) -> list:

    padded_seqs = [] # end padded gaps at the end of each sequence
    for seq in seqs:
        
        seq_length = len(seq)
        # number of pads needed
        pad_need = max_seq_length - seq_length
        # add number of padded tokens to the end
        seq += '-'*pad_need

        padded_seqs.append(seq)

    return padded_seqs


# creatae numerical represented sequences
def create_num_seqs(seq_list:list) -> list:

    """
        function description: convert amino acid tokens into numerical values -- preprocess sequences into their corresponding num. reps.
    """


    # tokenizer
    token2int = {x:ii for ii, x in enumerate('ACDEFGHIKLMNPQRSTVWY-')}

    # empty list to hold num rep. seqs.
    num_seq_list = []

    # convert aa labels into num labels.
    for seq in seq_list:
        num_seq_list.append([token2int[aa] for aa in seq])


    return num_seq_list


def prepare_SH3_data(
        df: pd.Series,
        max_seq_len: int,
        unaligned: int = 0
    ) -> (
            torch.FloatTensor,
            torch.FloatTensor,
            torch.FloatTensor,
            torch.FloatTensor
    ):
    """
    function description: preparing sh3 data.
    """

    annotated_df = df.iloc[df.RE_norm.dropna().index] # drop indexes without any norm R.E.

    # prepare sequences
    seq_list = [seq.replace('-','') for seq in list(df.Sequences_unaligned)]
    seq_lens = [len(seq) for seq in seq_list]
    padded_seq_list = pad_ends(seqs=seq_list, max_seq_length=max_seq_len)
    num_seq_list = create_num_seqs(padded_seq_list) # numerical representations

    '_______ Create dataset for training ______'

    onehot_transformer = torch.eye(21) # converter for creating  one-hot encodings
    X_onehot = onehot_transformer[num_seq_list] # create one-hot tensors

    # create torch datasets
    # --------------------
    # (1) create numerical represented sequences
    # num_seq_inputs = torch.tensor(num_seq_list) # do not need this for now...
    # (2) create one-hot encoded sequences
    X_onehot = torch.FloatTensor(X_onehot)
    # (3) create r.e. scores
    y_RE_reg = torch.FloatTensor(df.RE_norm.values).unsqueeze(1)
    # (4) create allowed loss terms and predictions
    accept_loss_samples = (~df.RE_norm.isnull()*1.).values # since some samples do not have exp. assay values, we want to ignore these values for disc loss
    accept_loss_samples = torch.FloatTensor(accept_loss_samples).unsqueeze(1)
    # (5) create binary classes
    y_RE_class = (y_RE_reg > 0.5)*1.0
    
    return (
            X_onehot,
            y_RE_reg,
            y_RE_class,
            accept_loss_samples
    )



class SH3_dataset(Dataset):
    """ 
    Sequence dataloader
    """

    def __init__(
            self,
            onehot_inputs: torch.FloatTensor,
            re_inputs: torch.FloatTensor,
            C_inputs: torch.FloatTensor,
            accept_inputs: torch.FloatTensor,
            transform: bool=True
        ):


        if not torch.is_tensor(re_inputs):
            self.re_inputs = torch.tensor(re_inputs).float()
        else:
            self.re_inputs = re_inputs

        if not torch.is_tensor(onehot_inputs):
            self.onehot_inputs = torch.tensor(onehot_inputs).float()
        else:
            self.onehot_inputs = onehot_inputs

        if not torch.is_tensor(accept_inputs):
            self.accept_inputs = torch.tensor(accept_inputs).float()
        else:
            self.accept_inputs = accept_inputs

        if not torch.is_tensor(C_inputs):
            self.C_inputs = torch.tensor(C_inputs).float()
        else:
            self.C_inputs = C_inputs


        self.transform = transform




    def __len__(self,):
        """
        number of samples total
        """
        return len(self.onehot_inputs)


    def __getitem__(self, idx:any) -> (
            torch.FloatTensor,
            torch.FloatTensor,
            torch.FloatTensor,
            torch.FloatTensor
        ):
        """
        extract and return the data batch samples
        """
    
        # convert the tensor to a simple list
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # r.e. score:
        y_re_reg = self.re_inputs[idx]
        # classification r.e.
        y_re_class = self.C_inputs[idx]
        # accept loss predictions
        accept_loss_preds = self.accept_inputs[idx]
        # onehot encoded sequences
        X_onehot = self.onehot_inputs[idx]
        
        return (
                X_onehot,
                y_re_reg,
                y_re_class,
                accept_loss_preds
        )

