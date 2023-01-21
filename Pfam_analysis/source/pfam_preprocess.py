"""
@author: Niksa Praljak

@Summay:

Preprocess function for the tasks that correspond to protein family designs.

"""


import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.autograd import Variable

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


import numpy as np
import pandas as pd
import numba
from numba import jit

from source.preprocess import pad_ends, create_num_seqs


def prepare_CM_dataset(
               data_path = './data/protein_families/CM/CM_natural_homologs.csv',
               alignment = True
    ):


    # create dataframe
    df = pd.read_csv(data_path)
 
    # protein sequences
    X = df.Sequence.values    
    
    # relative enrichment score:
    y = np.array( df['norm r.e.'].values ).reshape(-1, 1)
     
    if alignment:
         pass
    else:
         # remove deletion gaps
         X = [seq.replace('-','') for seq in X]
         
         # max length
         max_seq_length = max([len(seq) for seq in X])
   
         # pad ends
         seq_pad_ends = pad_ends(
                           seqs = X,
                           max_seq_length = max_seq_length
         )

    # one-hot encoded transformations:
    onehot_trans = np.eye(21)
 
    # convert sequences with padded ends into numerical and one-hot represented sequences
    seq_num = np.array( create_num_seqs( seq_pad_ends ) )
    seq_OH = onehot_trans[seq_num]


    return seq_num, seq_OH, y



class CM_dataset(Dataset):
    """
    CM sequence dataloader  
    """

    def __init__(
              self,
              num_inputs,
              onehot_inputs,
              pheno_outputs,
              unsupervised_option = True
    ):

        # numerical represented sequences
        if not torch.is_tensor(num_inputs):
            self.num_inputs = torch.tensor(num_inputs).float()

        else:
            self.num_inputs = num_inputs

        # protein sequences
        if not torch.is_tensor(onehot_inputs):
            self.onehot_inputs = torch.tensor(onehot_inputs).float()
        else:
            self.onehot_inputs = onehot_inputs
 
        # phenotype predictions
        if not torch.is_tensor(pheno_outputs):
            self.pheno_outputs = torch.tensor(pheno_outputs).float()
        else:
            self.pheno_outputs = pheno_outputs

        self.unsupervised = unsupervised_option

    def __len__(self):
        """
        function description: NUmber of sampled total
        """
        return len(self.onehot_inputs)
 
    def __getitem__(
                  self,
                  idx
        ):
        """ 
        function description: extract and return the data sample (i.e. workhorse)
        """         

        # convert the tensor to a simple list
        if torch.is_tensor(idx):
           idx = idx.tolist()

        # protein sequences
        x_inputs = self.num_inputs[idx] # numerical inputs
        x_OH_inputs = self.onehot_inputs[idx] # onehot-encoded inputs
        y_pheno = self.pheno_outputs[idx]

        if self.unsupervised:
           return x_inputs, x_OH_inputs

        else:
           return x_inputs, x_OH_inputs, y_pheno
        

'______________ S1A proteases _______________________'


def prepare_S1A_dataset(
               data_path = './data/protein_families/S1A/pfam_S1A.csv',
               alignment = False
    ):


    # create dataframe
    df = pd.read_csv(data_path)
 
    # extract additional sequence info:
    C_organism = df.organism.values
    C_env = df.environment.values
    C_vert = df.vert.values
    C_phylo_level0 = df.phylo_level0.values
    C_phylo_level1 = df.phylo_level1.values
    C_phylo_level2 = df.phylo_level2.values
    y_specificity = df.specificity.values
     
    # protein sequences
    X = df.Sequence.values    
   
    # one-hot encoded transformations:
    onehot_trans = np.eye(21)
 
    if alignment:
         X = list( df.Sequence.values )
         
         # convert sequences into numerical and one-hot represented sequences
         seq_num = np.array( create_num_seqs( X ) )
     
    else:

         X = list( df.Unaligned_Sequence.values )

         # max length
         max_seq_length = max([len(seq) for seq in X])
         
         # pad ends
         seq_pad_ends = pad_ends(
                           seqs = X,
                           max_seq_length = max_seq_length
         )

         # convert sequences with padded ends into numerical and one-hot represented sequences
         seq_num = np.array( create_num_seqs( seq_pad_ends ) )
    

    seq_OH = onehot_trans[seq_num]


    return seq_num, seq_OH, y_specificity, C_vert, C_env, C_organism, C_phylo_level0, C_phylo_level1, C_phylo_level2



class S1A_dataset(Dataset):
    """
    S1A sequence dataloader  
    """

    def __init__(
              self,
              num_inputs,
              onehot_inputs
    ):

        # numerical represented sequences
        if not torch.is_tensor(num_inputs):
            self.num_inputs = torch.tensor(num_inputs).float()

        else:
            self.num_inputs = num_inputs

        # protein sequences
        if not torch.is_tensor(onehot_inputs):
            self.onehot_inputs = torch.tensor(onehot_inputs).float()
        else:
            self.onehot_inputs = onehot_inputs

    def __len__(self):
        """
        function description: NUmber of sampled total
        """
        return len(self.onehot_inputs)
 
    def __getitem__(
                  self,
                  idx
        ):
        """ 
        function description: extract and return the data sample (i.e. workhorse)
        """         

        # convert the tensor to a simple list
        if torch.is_tensor(idx):
           idx = idx.tolist()

        # protein sequences
        x_inputs = self.num_inputs[idx] # numerical inputs
        x_OH_inputs = self.onehot_inputs[idx] # onehot-encoded inputs
 
        return x_inputs, x_OH_inputs

' ________________ Lactamase: ___________________'


def prepare_lactamase_dataset(
                     data_path = './data/protein_families/lactamase/lactamase_protein_family.csv',
                     alignment = False
    ):

    # create dataframe
    df = pd.read_csv(data_path)
 
    # extract additional sequence info:
    # fill this in later..

   
    # one-hot encoded transformations:
    onehot_trans = np.eye(21)
    

    if alignment:
         
         # sequence list
         X = list(df.Sequence.values)
        
          # convert sequences with padded ends into numerical and one-hot represented sequences
         seq_num = np.array( create_num_seqs( X ) )
    
    else:
         # sequence list
         X = list(df.Unaligned_Sequence.values)
         # max length
         max_seq_length = max([len(seq) for seq in X])

         # pad ends
         seq_pad_ends = pad_ends(
		           seqs = X,
		           max_seq_length = max_seq_length
         )

         # convert sequences with padded ends into numerical and one-hot represented sequences
         seq_num = np.array( create_num_seqs( seq_pad_ends ) )
    
    seq_OH = onehot_trans[seq_num]


    return seq_num, seq_OH



class lactamase_dataset(Dataset):
    """
    Lactamase sequence dataloader  
    """

    def __init__(
              self,
              num_inputs,
              onehot_inputs
    ):

        # numerical represented sequences
        if not torch.is_tensor(num_inputs):
            self.num_inputs = torch.tensor(num_inputs).float()

        else:
            self.num_inputs = num_inputs

        # protein sequences
        if not torch.is_tensor(onehot_inputs):
            self.onehot_inputs = torch.tensor(onehot_inputs).float()
        else:
            self.onehot_inputs = onehot_inputs

    def __len__(self):
        """
        function description: NUmber of sampled total
        """
        return len(self.onehot_inputs)
 
    def __getitem__(
                  self,
                  idx
        ):
        """ 
        function description: extract and return the data sample (i.e. workhorse)
        """         

        # convert the tensor to a simple list
        if torch.is_tensor(idx):
           idx = idx.tolist()

        # protein sequences
        x_inputs = self.num_inputs[idx] # numerical inputs
        x_OH_inputs = self.onehot_inputs[idx] # onehot-encoded inputs
 
        return x_inputs, x_OH_inputs
    

'__________________ GProtein family: _________________'

def prepare_Gprotein_dataset(
                    data_path = './data/protein_families/G_protein/pfam_G_protein.csv',
                    alignment = False
    ):

    # create dataframe
    df = pd.read_csv(data_path)
   
    # extract additional sequence info:
    # fill this in later ...

    # one-hot encoded transformations:
    onehot_trans = np.eye(21)


    if alignment:
         X = df.Sequence.values    
         # convert sequences with padded ends into numerical and one-hot represented sequences
         seq_num = np.array( create_num_seqs( X ) )
    else:
         X = df.Unaligned_Sequence.values.tolist()
         # max length
         max_seq_length = max([len(seq) for seq in X])

         # pad ends
         seq_pad_ends = pad_ends(
		           seqs = X,
		           max_seq_length = max_seq_length
         )

         # convert sequences with padded ends into numerical and one-hot represented sequences
         seq_num = np.array( create_num_seqs( seq_pad_ends ) )
   

    seq_OH = onehot_trans[seq_num]


    return seq_num, seq_OH




class Gprotein_dataset(Dataset):
    """
    G-protein sequence dataloader  
    """

    def __init__(
              self,
              num_inputs,
              onehot_inputs
    ):

        # numerical represented sequences
        if not torch.is_tensor(num_inputs):
            self.num_inputs = torch.tensor(num_inputs).float()

        else:
            self.num_inputs = num_inputs

        # protein sequences
        if not torch.is_tensor(onehot_inputs):
            self.onehot_inputs = torch.tensor(onehot_inputs).float()
        else:
            self.onehot_inputs = onehot_inputs

    def __len__(self):
        """
        function description: NUmber of sampled total
        """
        return len(self.onehot_inputs)
 
    def __getitem__(
                  self,
                  idx
        ):
        """ 
        function description: extract and return the data sample (i.e. workhorse)
        """         

        # convert the tensor to a simple list
        if torch.is_tensor(idx):
           idx = idx.tolist()

        # protein sequences
        x_inputs = self.num_inputs[idx] # numerical inputs
        x_OH_inputs = self.onehot_inputs[idx] # onehot-encoded inputs
 
        return x_inputs, x_OH_inputs
    


' _________________ DHFR family: _________________'



def prepare_DHFR_dataset(
                data_path = './data/protein_families/DHFR/pfam_DHFR.csv',
                alignment = False
    ):

    # create dataframe
    df = pd.read_csv(data_path)
   
    # extract additional sequence info:
    # fill this in later ...

    # one-hot encoded transformations:
    onehot_trans = np.eye(21)


    if alignment:
         X = list( df.Sequence.values )
         # convert sequences with padded ends into numerical and one-hot represented sequences
         seq_num = np.array( create_num_seqs( X ) )

    else:
         X = list( df.Unaligned_Sequence.values )
         # max length
         max_seq_length = max([len(seq) for seq in X])

         # pad ends
         seq_pad_ends = pad_ends(
		           seqs = X,
		           max_seq_length = max_seq_length
         )
    
         # convert sequences with padded ends into numerical and one-hot represented sequences
         seq_num = np.array( create_num_seqs( seq_pad_ends ) )
    

    seq_OH = onehot_trans[seq_num]


    return seq_num, seq_OH




class DHFR_dataset(Dataset):
    """
    DHFR sequence dataloader  
    """

    def __init__(
              self,
              num_inputs,
              onehot_inputs
    ):

        # numerical represented sequences
        if not torch.is_tensor(num_inputs):
            self.num_inputs = torch.tensor(num_inputs).float()

        else:
            self.num_inputs = num_inputs

        # protein sequences
        if not torch.is_tensor(onehot_inputs):
            self.onehot_inputs = torch.tensor(onehot_inputs).float()
        else:
            self.onehot_inputs = onehot_inputs

    def __len__(self):
        """
        function description: NUmber of sampled total
        """
        return len(self.onehot_inputs)
 
    def __getitem__(
                  self,
                  idx
        ):
        """ 
        function description: extract and return the data sample (i.e. workhorse)
        """         

        # convert the tensor to a simple list
        if torch.is_tensor(idx):
           idx = idx.tolist()

        # protein sequences
        x_inputs = self.num_inputs[idx] # numerical inputs
        x_OH_inputs = self.onehot_inputs[idx] # onehot-encoded inputs
 
        return x_inputs, x_OH_inputs
    



