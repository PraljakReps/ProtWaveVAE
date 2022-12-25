
"""
WaveNet-based generator for protein sequences:

@author: Niksa Praljak
@summary: WaveNet model used as a decoder component for a InfoVAE or MMD-VAE model architecture
"""


import torch
from torch import optim
from torch.nn import functional as F
import torch.nn as nn
from torch.autograd import Variable

import numpy as np
import pandas as pd



# latent conditional net: maps the latent vectors to high-dimensional tensor, matching the input sequences for training WaveNet

class CondNet(nn.Module):
    """
    Linear layer to map from latent space z to sequence with protein length 59

    """

    def __init__(
            self,
            z_dim: int=6,
            output_shape: any=(1,59)
        ):

        super(CondNet, self).__init__()

        self.input_shape = z_dim
        self.output_shape = output_shape
        self.linear = nn.Linear(self.input_shape, np.prod(self.output_shape), bias = False)
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, z: torch.FloatTensor) -> torch.FloatTensor:
        return self.linear(z).view(-1, 1, self.output_shape[-1])



class Causal_conv1d(nn.Conv1d):

    def __init__(
            self,
            C_in: int,
            C_out: int,
            mask_type: any,
            **kwargs
        ):
        
        # default
        kwargs["kernel_size"] = 3 if "kernel_size" not in kwargs else kwargs["kernel_size"]
        kwargs["dilation"] = 1 if "dilation" not in kwargs else kwargs["dilation"]
        kwargs["stride"] = 1 if "stride" not in kwargs else kwargs["strides"]
        kwargs["padding"] = 0 if "padding" not in kwargs else kwargs["padding"]
        kwargs["bias"] = False if "bias" not in kwargs else kwargs["bias"]
      

        # define mask types
        self.mask_type = mask_type


        if self.mask_type == 'A':
            self.__left_pad = kwargs['kernel_size'] # does not peek at the present amino acid
        else: # mask B
            self.__left_pad = kwargs['dilation']*(kwargs['kernel_size'] - 1)+1
    
        kwargs["padding"] = self.__left_pad
      
        super(Causal_conv1d, self).__init__(C_in, C_out, **kwargs)

        self.reset_parameters()
 

    def reset_parameters(self,):
        nn.init.xavier_uniform_(self.weight)


    def forward(
            self,
            x: torch.FloatTensor
        ) -> torch.FloatTensor:
        results = super(Causal_conv1d, self).forward(x)
        if self.__left_pad != 0:
           return results[:, :, :-(self.__left_pad+1)]
        else:
           return results


# wave head: multiple layers of dilated-causal convolutions

class Wave_head(nn.Module):

    def __init__(
            self,
            device: str,
            C_in: int,
            C_out: int,
            num_dilation_rates: int,
            **kwargs
        ):

        super(Wave_head, self).__init__()
        self.num_rates = num_dilation_rates
        # causal (no dilation) block
        self.causal_blocks = nn.ModuleList()
        # signal and gate for the input sequences
        self.signal_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        # conditional module (latent codes)
        self.cond_signal_convs = nn.ModuleList()
        self.cond_gate_convs = nn.ModuleList()
        # skip and residual blocks for the input sequences
        self.residual_blocks = nn.ModuleList()
        self.skip_blocks = nn.ModuleList()

        # default
        kwargs["kernel_size"] = 2 if "kernel_size" not in kwargs else kwargs["kernel_size"]
        # dilation rates: 1, 2, 4, 8, 16, ... , 2^(num_rates - 1)
        dilation_rates = [2**ii for ii in range(self.num_rates)] # grow by power of 2
        # set GPU (or CPU)
        self.device = device

        # first conv layer. 
        self.causal_blocks.append(nn.Conv1d(C_in, C_out, kernel_size = 1, padding = 0, bias = True))
        nn.init.xavier_uniform_(self.causal_blocks[0].weight)

        # loop over the number of rates (depth of the decoder)
        for ii, dilation_rate in enumerate(dilation_rates):


            # method for padding the input sequences
            if ii == 0:
                mask_type = 'A'

            else:
                mask_type = 'B'

            # signal and gate of the input sequences
            self.signal_convs.append(Causal_conv1d(C_out, C_out, mask_type,
                                                kernel_size = kwargs["kernel_size"], dilation = dilation_rate))
            nn.init.xavier_uniform_(self.signal_convs[ii].weight)
            self.gate_convs.append(Causal_conv1d(C_out, C_out, mask_type,
                                                      kernel_size = kwargs["kernel_size"], dilation = dilation_rate))
            nn.init.xavier_uniform_(self.gate_convs[ii].weight)


            # signal and gate for the conditional latent codes
            self.cond_signal_convs.append(nn.Conv1d(1, C_out, kernel_size = 1, bias = False))
            nn.init.xavier_uniform_(self.cond_signal_convs[ii].weight)

            self.cond_gate_convs.append(nn.Conv1d(1, C_out, kernel_size = 1, bias = False))
            nn.init.xavier_uniform_(self.cond_gate_convs[ii].weight)


            # residual and skip blocks
            self.residual_blocks.append(nn.Conv1d(C_out, C_out, kernel_size = 1, stride = 1,
                                             padding = 0, bias = False))
            nn.init.xavier_uniform_(self.residual_blocks[ii].weight)
            self.skip_blocks.append(nn.Conv1d(C_out, C_out, kernel_size = 1, stride = 1,
                                             padding = 0, bias = False))
            nn.init.xavier_uniform_(self.skip_blocks[ii].weight)

        # create nonlinear activation functions
        self.tanh = nn.Tanh()
        self.sigm = nn.Sigmoid()
        self.softmax = nn.Softmax(dim = 1)
    
    def forward(
            self,
            x: torch.FloatTensor,
            z: torch.FloatTensor
        ) -> (
                torch.FloatTensor,
                torch.FloatTensor
        ):

        x = self.causal_blocks[0](x) # 1x1 conv operation

        # create residual connection
        orig_x = x.clone()
        cum_skip = 0
        # loop over the number of dilation rates (increases the model's receptive field)
        for ii in range(self.num_rates):


            # conditional operation for the signal: tanh( W*X + V*Z)
            signal = self.signal_convs[ii](x) + self.cond_signal_convs[ii](z)

            # conditional operation for the gate (memory): sigmoid( W*X + V*Z)
            gate = self.gate_convs[ii](x) + self.cond_gate_convs[ii](z)

            # gate operation
            x =  signal *self.sigm( gate )

            skip = self.skip_blocks[ii](x) # skip operation
            res = self.residual_blocks[ii](x) # residual operation

            if ii == 0:
                x = orig_x + res # residual enter the next layer (most likely dilated)
            else:
                x = res + x
            # cummulate all skip connectino for the final predictions
            cum_skip = cum_skip + skip


        return (
                x,
                cum_skip
        )



# final amino acid prediction head

class Top_head(nn.Module):
    """
    The last head of the network which maps the Wavenet tokens into amino acids distributions


    Inputs:
        - cumulative skip connections are inputted to this top neural network component

    Outputs:
        - probability distribution predictions over individual residual sites

    """
    def __init__(
            self,
            C_in: int,
            C_out: int,
            hidden_state: int=256
        ):
        super(Top_head, self).__init__()

        self.conv1 = nn.Conv1d(C_in, hidden_state, 1, bias = True)
        nn.init.kaiming_uniform_(self.conv1.weight, nonlinearity = 'relu')

        self.conv2 = nn.Conv1d(hidden_state, C_out, 1, bias = True)
        nn.init.kaiming_uniform_(self.conv2.weight, nonlinearity = 'relu')

        self.relu = torch.nn.ReLU()
        self.softmax = nn.Softmax(dim = 1)



    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """
        Same architecture as the diagram shown on Figure 4 in van der Oord et al., ArXiv 2016

        """
        h = self.relu(x)
        h = self.conv1(h)
        h = self.relu(h)
        logits = self.conv2(h)

        # note: outputs only energies/logits and not probs. Therefore, wrap a softmax with this neural network's outputs.
        return logits



# overall model architecture.

class Wave_generator(nn.Module):

    def __init__(
            self,
            protein_len: int,
            class_labels: int,
            DEVICE: str,
            wave_hidden_state: int=32,
            head_hidden_state: int=256,
            num_dil_rates: int=5,
            kernel_size: int=3
        ):
        super(Wave_generator, self).__init__()

        # model parameters
        self.protein_len = protein_len
        self.class_labels = class_labels
        self.wave_hidden_state = wave_hidden_state
        self.head_hidden_state = head_hidden_state
        self.num_dil_rates = num_dil_rates
        self.kernel_size = kernel_size

        # Wave decoder
        self.wave_head = Wave_head(DEVICE, self.class_labels, self.wave_hidden_state, num_dilation_rates = self.num_dil_rates, kernel_size = self.kernel_size)
        # Convolution mapping to classes
        self.output_head = Top_head(self.wave_hidden_state, self.class_labels, self.head_hidden_state)

        # define softmax operator
        self.softmax = nn.Softmax(dim = 1)

        self.device = DEVICE


    def forward(
            self,
            x: torch.FloatTensor,
            z: torch.FloatTensor
        ) -> torch.FloatTensor:

        cum_skip = 0
        x, skip = self.wave_head(x, z)
        cum_skip += skip

        logits = self.output_head(cum_skip) 
        # note: only outputs energies/logits and not probs
        return logits

