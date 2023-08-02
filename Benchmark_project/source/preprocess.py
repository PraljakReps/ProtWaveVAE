"""
@author: Niksa Praljak


@summary: 
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
import matplotlib.pyplot
from numba import jit 
import json

'__________________ General functions: __________________________ '

## Create numerical represented sequences
def create_num_seqs(
        seq_list: list
    ) -> list:
    """
    function description: this convert amino acid tokens into numerical values, preprocessing sequences into their
    corresponding numerical representations.


    args: 
        seq_list --> list containing sequences with aa labels

    returns:
        num_seq_list --> list containing sequences with numerical represented aa labels

    """

    
    # We will use this dictionary to map each character to an integer so that it can be used as an input to our ML models:
    dict_int2aa = {
                   0:"A",1:"C",2:"D",3:"E",4:"F",
                   5:"G",6:"H",7:"I",8:"K",9:"L",
                   10:"M",11:"N",12:"P",13:"Q",14:"R",
                   15:"S",16:"T",17:"V",18:"W",19:"Y",20:"-"
                  }
    
    # tokenizer
    token2int = {x:ii for ii, x in enumerate('ACDEFGHIKLMNPQRSTVWY-')}
    
    # empty list to hold numerical represented seqs
    num_seq_list = []
    
    # convert aa labels into numerical labels
    for seq in seq_list:
        num_seq_list.append([token2int[aa] for aa in seq])
        
    return num_seq_list



'_________________ GFP prep. ______________________________' 


@jit(nopython = True)
def pad_ends(
        seqs: list,
        max_seq_length: int=237
    ) -> list:
	"""
	function description: padded the sequence ends
	
	"""

	padded_seqs = [] # seqs with padded ends
	for seq in seqs:
		# current protein seq. length
		seq_length = len(seq)
		# number of pads needed
		pad_need = max_seq_length - seq_length
		# add number of padded tokens
		seq+= '-'*pad_need
		
		padded_seqs.append(seq)

	return padded_seqs

def prepare_GFP_datasets(
        train_path: str='./data/GFP_fluorescence_train.csv',
	valid_path: str='./data/GFP_fluorescence_valid.csv',
	test_path: str='./data/GFP_fluorescence_test.csv'
        ) -> (
                torch.FloatTensor,
                torch.FloatTensor,
                torch.FloatTensor,
                torch.FloatTensor,
                torch.FloatTensor,
                torch.FloatTensor,
                torch.FloatTensor,
                torch.FloatTensor,
                torch.FloatTensor,
                int
        ):
                """
	        function description: prepare the onehot encoded tensors, mutated vectors, phenotypic predictions

    	        args:
	
	        returns:
	
	        """


                # create dataframes
                train_df = pd.read_csv(train_path)
                valid_df = pd.read_csv(valid_path)
                test_df = pd.read_csv(test_path)
                WT_df = train_df[ train_df.num_mutations == 0 ]	
                
                # compute max protein length for each train, valid, and test set
                train_max_len, valid_max_len, test_max_len = max(train_df.protein_length.values),max(valid_df.protein_length.values), max(test_df.protein_length.values)
                # compute max protein length in general
                max_seq_length = max( train_max_len, valid_max_len, test_max_len )
                
                # padded ends
                train_seq_pad_ends = pad_ends(train_df.primary.tolist(), max_seq_length) # train set
                valid_seq_pad_ends = pad_ends(valid_df.primary.tolist(), max_seq_length) # valid set
                test_seq_pad_ends = pad_ends(test_df.primary.tolist(), max_seq_length) # test set
                WT_seq_pad_ends = pad_ends(WT_df.primary.tolist(), max_seq_length) # WT ref. seq

                
                # transformer for onehot encodings
                onehot_trans = torch.eye(21)

                # convert sequences with padded ends into as numerical and one-hot represented sequences	
                # -----------------
                # training set
                train_num = torch.tensor( create_num_seqs( train_seq_pad_ends ) ) 
                train_OH = onehot_trans[train_num]

                # validation set
                valid_num = torch.tensor( create_num_seqs( valid_seq_pad_ends ) )
                valid_OH = onehot_trans[valid_num]

                # test set
                test_num = torch.tensor( create_num_seqs( test_seq_pad_ends ) )
                test_OH = onehot_trans[test_num]

                # WT reference sequence
                WT_num = torch.tensor( create_num_seqs( WT_seq_pad_ends ))
                WT_OH = onehot_trans[WT_num]

                # create gfp regression predictions
                train_pheno = torch.tensor(train_df.log_fluorescence.values).reshape(-1, 1)
                valid_pheno = torch.tensor(valid_df.log_fluorescence.values).reshape(-1, 1)
                test_pheno = torch.tensor(test_df.log_fluorescence.values).reshape(-1, 1)
                
                max_seq_len = train_num.shape[-1]
                print('Max sequence length:', max_seq_len)

                return (
                        train_num,
                        train_OH,
                        train_pheno.float(),
                        valid_num,
                        valid_OH,
                        valid_pheno.float(),
                        test_num,
                        test_OH,
                        test_pheno.float(),
                        max_seq_len
                )




class GFP_dataset(Dataset):
    """
    GFP sequence dataloader
    
    """
    
    def __init__(
            self,
            num_inputs: any,
            onehot_inputs: any,
            pheno_outputs: any
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
        
        # phenotypic predictions
        if not torch.is_tensor(pheno_outputs):
            self.pheno_outputs = torch.tensor(pheno_outputs).float()
        else:
            self.pheno_outputs = pheno_outputs
        
        
    def __len__(self,):
        """
        function description: Number of sampled total
        """
        return len(self.onehot_inputs)
    
    def __getitem__(self, idx: any) -> (
            torch.FloatTensor,
            torch.FloatTensor,
            torch.FloatTensor
        ):
        """
        function description: extract and return the data sample (i.e. workhorse)
        """
        
        # cover the tensor to a simple list
        if torch.is_tensor(idx):
            idx = idx.tolist()
            

        # protein sequences
        x_inputs = self.num_inputs[idx] # numerical inputs
        x_outputs = self.onehot_inputs[idx] # one hot encoded outputs
        
        
        # pheno outupts
        pheno_outputs = self.pheno_outputs[idx]
        
        
        return (
                x_inputs,
                x_outputs,
                pheno_outputs
        )


' ___________________ FLIP task: GB1 __________________________'


# decide which data split to implement ...
def extract_GB1_task_df(
        df: pd.Series,
        split_option: int
    ) -> (
            pd.Series,
            pd.Series,
            pd.Series
    ):
    """
    function description: decided which dataset split to conduct
    """

    def split(
            df: pd.Series,
            TASK: any
        ) -> (
                pd.Series,
                pd.Series,
                pd.Series
        ):
        # dataframe splits ...
        train_df = df[df[TASK] == 'train']
        test_df = df[df[TASK] == 'test']
        valid_df = df.loc[df[TASK + '_validation'].dropna().index.values]
        
        return train_df, valid_df, test_df

   
    if split_option == 0 or split_option == 'sampled':
       TASK = 'sampled'
     
       # dataframe splits ...
       train_df,valid_df, test_df = split(df, TASK)
   
    elif split_option == 1 or split_option == 'one_vs_rest':
         TASK = 'one_vs_rest'
      
         # dataframe splits ..
         train_df, valid_df, test_df = split(df, TASK)
  
    elif split_option == 2 or split_option == 'two_vs_rest':
         TASK = 'two_vs_rest'
      
         # dataframe splits ..
         train_df, valid_df, test_df = split(df, TASK)
  

    elif split_option == 3 or split_option == 'three_vs_rest':
         TASK = 'three_vs_rest'
        
         # dataframe splits ..
         train_df, valid_df, test_df = split(df, TASK)
   
    elif split_option == 4 or split_option == 'low_vs_high':
         TASK = 'low_vs_high'
       
         # dataframe splits ..
         train_df, valid_df, test_df = split(df, TASK)
   
    else:
         return print("No correct option choosen .. Choice (0-6)")
   
    print(f'Train size: {train_df.shape} | Valid size: {valid_df.shape} | Test size: {test_df.shape}')
    return (
            train_df,
            valid_df,
            test_df
    )


def prepare_GB1_datasets(
        GB1_path: str='./data/GB1/four_mutations_full_data.csv',
        split_option: int=0
    ) -> (
            torch.FloatTensor,
            torch.FloatTensor,
            torch.FloatTensor,
            torch.FloatTensor,
            torch.FloatTensor,
            torch.FloatTensor,
            torch.FloatTensor,
            torch.FloatTensor,
            torch.FloatTensor,
            int
    ):
    """
    function description: prepare the onehot encoded tensors, muatted vectors, phenotypic predictions
    """
         
    # load csv file ...
    GB1_df = pd.read_csv(GB1_path)

    # calculate sequence lengths and cnocatenate it to the original dataframe
    GB1_seq_len_df = pd.DataFrame([len(gb1_seq) for gb1_seq in GB1_df.sequence.values])
    GB1_seq_len_df.columns = ['seq_length']
    GB1_df = pd.concat([GB1_df, GB1_seq_len_df], axis = 1) #add a column on sequence lengths ...
    
    # create training, validation, and testing dataframes based on a given task (e.g. split_option = 0)
    train_df, valid_df, test_df = extract_GB1_task_df(GB1_df, split_option)
    
    # compute max protein length for each train, valid, and test set
    max_lengths = [max(train_df.seq_length.values), max(valid_df.seq_length.values), max(test_df.seq_length.values)]
    max_seq_len = max(max_lengths)
   
    # padded ends
    train_seq_pad_ends = pad_ends(train_df.sequence.tolist(), max_seq_len) # train set
    valid_seq_pad_ends = pad_ends(valid_df.sequence.tolist(), max_seq_len) # valid set
    test_seq_pad_ends = pad_ends(test_df.sequence.tolist(), max_seq_len) # test set
     
    # transformation to onehot encodings
    onehot_trans = torch.eye(21)
    

    # convert sequences with padded ends into as numerical and one-hot represented sequences
    # ------------------
    # training set
    train_num = torch.tensor( create_num_seqs( train_seq_pad_ends ) )
    train_OH = onehot_trans[train_num]
    
    # validation set
    valid_num = torch.tensor( create_num_seqs( valid_seq_pad_ends ) )
    valid_OH = onehot_trans[valid_num]

    # testing set
    test_num = torch.tensor( create_num_seqs( test_seq_pad_ends ) )
    test_OH = onehot_trans[test_num]
    
    # -create gfp regression predictions=
    # clean stability scores because they tabulated as lists instead of floating points ...
    train_pheno = torch.tensor([score for score in train_df.Fitness.values]).reshape(-1, 1)
    valid_pheno = torch.tensor([score for score in valid_df.Fitness.values]).reshape(-1, 1)
    test_pheno = torch.tensor([score for score in test_df.Fitness.values]).reshape(-1, 1)

    max_seq_len = train_num.shape[-1]
    print('Max sequence length:', max_seq_len)

    return (
            train_num,
            train_OH,
            train_pheno.float(),
            valid_num,
            valid_OH,
            valid_pheno.float(),
            test_num,
            test_OH,
            test_pheno.float(),
            max_seq_len
    )


class GB1_dataset(Dataset):
    """
    FLIP stability task: learning pheno landscape while being epistatic
    """

    def __init__(
            self,
            num_inputs: any,
            onehot_inputs: any,
            pheno_outputs: any
        ):
       
        # numerical represented sequences
        if not torch.is_tensor(num_inputs):
            self.num_inputs = torch.FloatTensor(num_inputs)
        else:
            self.num_inputs = num_inputs

        # protein sequence
        if not torch.is_tensor(onehot_inputs):
            self.onehot_inputs = torch.FloatTensor(onehot_inputs)
        else:
            self.onehot_inputs = onehot_inputs
        
        # phenotypic predictions
        if not torch.is_tensor(pheno_outputs):
            self.pheno_outputs = torch.FloatTensor(pheno_outputs)
        else:
            self.pheno_outputs = pheno_outputs

    def __len__(self,):
        """
        function description: Number of sampled total
        """
        return len(self.onehot_inputs)
    

    def __getitem__(self, idx: any) -> (
            torch.FloatTensor,
            torch.FloatTensor,
            torch.FloatTensor
        ):
        """
        function description: extract and return the data sample (i.e. workhorse)
        """
      
        # cover the tensor to a simple list
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # protein sequences
        x_num = self.num_inputs[idx] # numerical inputs
        x_onehot = self.onehot_inputs[idx] # one hot encoded outputs
    
        # pheno outputs
        pheno_outputs = self.pheno_outputs[idx]

        return (
                x_num,
                x_onehot,
                pheno_outputs
        )


 
' ___________________ FLIP task: AAV capsids __________________________'


# decide which data split to implement ...
def extract_AAV_task_df(
        df: pd.Series,
        split_option: int=0
    ) -> (
            pd.Series,
            pd.Series,
            pd.Series
    ):
    """
    function description: decided which dataset split to conduct
    """

    def split(
            df: pd.Series,
            TASK: str
        ) -> (
                pd.Series,
                pd.Series,
                pd.Series
        ):

        # dataframe splits ...
        train_df = df[df[TASK] == 'train']
        test_df = df[df[TASK] == 'test']
        valid_df = df.loc[df[TASK + '_validation'].dropna().index.values]
        
        return (
                train_df,
                valid_df,
                test_df
        )

   
    if split_option == 0 or split_option == 'sampled_split':
       TASK = 'sampled_split'
     
       # dataframe splits ...
       train_df,valid_df, test_df = split(df, TASK)
   
    elif split_option == 1 or split_option == 'one_vs_many_split':
         TASK = 'one_vs_many_split'
     
         # dataframe splits ..
         train_df, valid_df, test_df = split(df, TASK)
  
    elif split_option == 2 or split_option == 'two_vs_many_split':
         TASK = 'two_vs_many_split'
      
         # dataframe splits ..
         train_df, valid_df, test_df = split(df, TASK)
  

    elif split_option == 3 or split_option == 'seven_vs_many_split':
         TASK = 'seven_vs_many_split'

         # dataframe splits ..
         train_df, valid_df, test_df = split(df, TASK)
   
    elif split_option == 4 or split_option == 'low_vs_many_split':
         TASK = 'low_vs_high_split'
       
         # dataframe splits ..
         train_df, valid_df, test_df = split(df, TASK)
   
    elif split_option == 5 or split_option == 'mut_des_split':
         TASK = 'mut_des_split'
       
         # dataframe splits ..
         train_df, valid_df, test_df = split(df, TASK)
   
    elif split_option == 6 or split_option == 'des_mut_split':
         TASK = 'des_mut_split'
       
         # dataframe splits ..
         train_df, valid_df, test_df = split(df, TASK)

    else:
         return print("No correct option choosen .. Choice (0-6)")

    return (
            train_df,
            valid_df,
            test_df
    )


def prepare_AAV_datasets(
        df_path: str='./data/AAV/full_data.csv',
        split_option: int=0
    ) -> (
            torch.FloatTensor,
            torch.FloatTensor,
            torch.FloatTensor,
            torch.FloatTensor,
            torch.FloatTensor,
            torch.FloatTensor,
            torch.FloatTensor,
            torch.FloatTensor,
            torch.FloatTensor,
            int
    ):
    """
    function description: prepare the onehot encoded tensors, mutated vectors, phenotypic predictions
    """
         
    # read and create dataframe...
    aav_df = pd.read_csv(df_path)
    # calculate sequence lengths and cnocatenate it to the original dataframe
    aav_seq_len_df = pd.DataFrame([len(aav_seq) for aav_seq in aav_df.full_aa_sequence.values])
    aav_seq_len_df.columns = ['seq_length']
    aav_df = pd.concat([aav_df, aav_seq_len_df], axis = 1) #add a column on sequence lengths ...
    
    # create training, validation, and testing dataframes based on a given task (e.g. split_option = 0)
    train_df, valid_df, test_df = extract_AAV_task_df(aav_df, split_option)
     
    # compute max protein length for each train, valid, and test set
    max_lengths = [max(train_df.seq_length.values), max(valid_df.seq_length.values), max(test_df.seq_length.values)]
    max_seq_len = max(max_lengths)
   
    # padded ends
    train_seq_pad_ends = pad_ends(train_df.full_aa_sequence.tolist(), max_seq_len) # train set
    valid_seq_pad_ends = pad_ends(valid_df.full_aa_sequence.tolist(), max_seq_len) # valid set
    test_seq_pad_ends = pad_ends(test_df.full_aa_sequence.tolist(), max_seq_len) # test set
     
    # transformation to onehot encodings
    onehot_trans = torch.eye(21)
    

    # convert sequences with padded ends into as numerica`l and one-hot represented sequences
    # ------------------
    # training set
    train_num = torch.tensor( create_num_seqs( train_seq_pad_ends ) )
    train_OH = onehot_trans[train_num]
    
    # validation set
    valid_num = torch.tensor( create_num_seqs( valid_seq_pad_ends ) )
    valid_OH = onehot_trans[valid_num]

    # testing set
    test_num =  torch.tensor( create_num_seqs( test_seq_pad_ends ) )
    test_OH = onehot_trans[test_num]
    
    # -create gfp regression predictions=
    # clean stability scores because they tabulated as lists instead of floating points ...
    train_pheno = torch.tensor([score for score in train_df.score.values]).reshape(-1, 1)
    valid_pheno = torch.tensor([score for score in valid_df.score.values]).reshape(-1, 1)
    test_pheno = torch.tensor([score for score in test_df.score.values]).reshape(-1, 1)
    
    max_seq_len = train_num.shape[-1]
    print('Max sequence length:', max_seq_len)

    return (
            train_num,
            train_OH,
            train_pheno.float(),
            valid_num,
            valid_OH,
            valid_pheno.float(),
            test_num,
            test_OH,
            test_pheno.float(), 
            max_seq_len
    )


class AAV_dataset(Dataset):
    """
    FLIP function prediction task:  predict fitness
    """

    def __init__(
            self,
            num_inputs: any,
            onehot_inputs: any,
            pheno_outputs: any
        ):
        
       # numerical represented sequences
        if not torch.is_tensor(num_inputs):
             self.num_inputs = torch.FloatTensor(num_inputs)
        else:
            self.num_inputs = num_inputs

        # protein sequence
        if not torch.is_tensor(onehot_inputs):
            self.onehot_inputs = torch.FloatTensor(onehot_inputs)
        else:
            self.onehot_inputs = onehot_inputs
        
        # phenotypic predictions
        if not torch.is_tensor(pheno_outputs):
#            pass
            self.pheno_outputs = torch.FloatTensor(pheno_outputs)
        else:
            self.pheno_outputs = pheno_outputs

    def __len__(self,):
        """
        function description: Number of sampled total
        """
        return len(self.onehot_inputs)
    

    def __getitem__(self, idx: any) -> (
            torch.FloatTensor,
            torch.FloatTensor,
            torch.FloatTensor
        ):
        """
        function description: extract and return the data sample (i.e. workhorse)
        """
      
        # cover the tensor to a simple list
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # protein sequences
        x_num = self.num_inputs[idx] # numerical inputs
        x_onehot = self.onehot_inputs[idx] # one hot encoded outputs
    
        # pheno outputs
        pheno_outputs = self.pheno_outputs[idx]
        

        return (
                x_num, 
                x_onehot,
                pheno_outputs
        )

' ___________________ TAPE task: stability  __________________________'

def prepare_stability_datasets(
        train_path: str='./data/stability/stability_train.json',
        valid_path: str='./data/stability/stability_valid.json',
        test_path: str='./data/stability/stability_test.json'
    ) -> (
            torch.FloatTensor,
            torch.FloatTensor,
            torch.FloatTensor,
            torch.FloatTensor,
            torch.FloatTensor,
            torch.FloatTensor,
            torch.FloatTensor,
            torch.FloatTensor,
            torch.FloatTensor,
            int
    ):
    """
    function description: prepare the onehot encoded tensors, mutated vectors, phenotypic predictions
    """
         
    # load json files ...
    train_stability_path = open(train_path)
    valid_stability_path = open(valid_path)
    test_stability_path = open(test_path)
 
    # json objet as dicts ..
    train_stability_data = json.load(train_stability_path)
    valid_stability_data = json.load(valid_stability_path)
    test_stability_data = json.load(test_stability_path)
    
    # read and create dataframes
    train_df = pd.DataFrame(train_stability_data)
    valid_df = pd.DataFrame(valid_stability_data)
    test_df = pd.DataFrame(test_stability_data)
    
    # compute max protein length for each train, valid, and test set
    max_lengths = [max(train_df.protein_length.values), max(valid_df.protein_length.values), max(test_df.protein_length.values)]
    max_seq_len = max(max_lengths)
   
    # padded ends
    train_seq_pad_ends = pad_ends(train_df.primary.tolist(), max_seq_len) # train set
    valid_seq_pad_ends = pad_ends(valid_df.primary.tolist(), max_seq_len) # valid set
    test_seq_pad_ends = pad_ends(test_df.primary.tolist(), max_seq_len) # test set
     
    # transformation to onehot encodings
    onehot_trans = torch.eye(21)
    

    # convert sequences with padded ends into as numerical and one-hot represented sequences
    # ------------------
    # training set
    train_num = torch.tensor( create_num_seqs( train_seq_pad_ends ) )
    train_OH = onehot_trans[train_num]
    
    # validation set
    valid_num = torch.tensor( create_num_seqs( valid_seq_pad_ends ) )
    valid_OH = onehot_trans[valid_num]

    # testing set
    test_num = torch.tensor( create_num_seqs( test_seq_pad_ends ) )
    test_OH = onehot_trans[test_num]
    
    # -create gfp regression predictions=
    # clean stability scores because they tabulated as lists instead of floating points ...
    train_pheno = torch.tensor([score for score in train_df.stability_score.values]).reshape(-1, 1)
    valid_pheno = torch.tensor([score for score in valid_df.stability_score.values]).reshape(-1, 1)
    test_pheno = torch.tensor([score for score in test_df.stability_score.values]).reshape(-1, 1)

    max_seq_len = train_num.shape[-1]
    print('Max sequence length:', max_seq_len)

    return (
            train_num,
            train_OH, 
            train_pheno,
            valid_num,
            valid_OH,
            valid_pheno,
            test_num,
            test_OH,
            test_pheno,
            max_seq_len
    )



class stability_dataset(Dataset):
    """
    TAPE stability task: learning pheno landscape while being epistatic
    """

    def __init__(
            self,
            num_inputs: any,
            onehot_inputs: any,
            pheno_outputs: any
        ):
       
        # numerical represented sequences
        if not torch.is_tensor(num_inputs):
            self.num_inputs = torch.FloatTensor(num_inputs)
        else:
            self.num_inputs = num_inputs

        # protein sequence
        if not torch.is_tensor(onehot_inputs):
            self.onehot_inputs = torch.FloatTensor(onehot_inputs)
        else:
            self.onehot_inputs = onehot_inputs
        
        # phenotypic predictions
        if not torch.is_tensor(pheno_outputs):
            self.pheno_outputs = torch.FloatTensor(pheno_outputs)
        else:
            self.pheno_outputs = pheno_outputs

    def __len__(self,):
        """
        function description: Number of sampled total
        """
        return len(self.onehot_inputs)
    

    def __getitem__(self, idx: any) -> (
            torch.FloatTensor,
            torch.FloatTensor,
            torch.FloatTensor
        ):
        """
        function description: extract and return the data sample (i.e. workhorse)
        """
      
        # cover the tensor to a simple list
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # protein sequences
        x_num = self.num_inputs[idx] # numerical inputs
        x_onehot = self.onehot_inputs[idx] # one hot encoded outputs
    
        # pheno outputs
        pheno_outputs = self.pheno_outputs[idx]

        return (
                x_num, 
                x_onehot,
                pheno_outputs
        )



