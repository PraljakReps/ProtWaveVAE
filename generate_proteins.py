import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.autograd import Variable
from torch.nn import functional as F

# super Pytorch Lightning
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping

import source.preprocess as prep
import source.wavenet_decoder as wavenet
import source.model_components as model_comps
import source.PL_wrapper as PL_wrapper
import train_ProtWaveVAE as ProtWaveVAE

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import sys
import argparse
from tqdm import tqdm
import random
import os



def get_args() -> any:

    # write output path name
    parser = argparse.ArgumentParser()
    
    # path varibles
    parser.add_argument('--dataset_path', default='./data/ACS_SynBio_SH3_dataset.csv')
    parser.add_argument('--output_results_path', default='./outputs/SH3_task/ProtWaveVAE_SSTrainingHist.csv')
    parser.add_argument('--output_model_path', default='./outputs/SH3_task/ProtWaveVAE_SSTrainingHist.pth')
    parser.add_argument('--save_dir', default='./outputs/SH3_design_pool')
    
    # model training variables
    parser.add_argument('--SEED', default=42, type=int, help='Random seed')
    parser.add_argument('--batch_size', default=512, type=int, help='Size of the batch.')
    parser.add_argument('--epochs', default=1000, type=int, help='Number of epochs')
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--DEVICE', default='cuda', help='Learning rate')
    parser.add_argument('--dataset_split', default=1, type=int, help='Choose whether to split into train/valid sets')
    parser.add_argument('--N', default=100, type=int, help='Batch size')

    # general architecture variables
    parser.add_argument('--z_dim', default=6, type=int, help='Latent space size')
    parser.add_argument('--num_classes', default=2, type=int, help='functional/nonfunctional labels')
    parser.add_argument('--aa_labels', default=21, type=int, help='AA plus pad gap (20+1) labels')
    
    # encoder hyperparameters
    parser.add_argument('--encoder_rates', default=5, type=int, help='dilation convolution depth')
    parser.add_argument('--C_in', default=21, type=int, help='input feature depth')
    parser.add_argument('--C_out', default=256, type=int, help='output feature depth')
    parser.add_argument('--alpha', default=0.1, type=float, help='leaky Relu hyperparameter (optional)')
    parser.add_argument('--enc_kernel', default=3, type=int, help='kernel filter size')
    parser.add_argument('--num_fc', default=1, type=int, help='number of fully connect layers')
      
    # top model (discriminative decoder) hyperparameters
    parser.add_argument('--disc_num_layers', default=2, type=int, help='depth of the discrim. top model')
    parser.add_argument('--hidden_width', default=10, type=int, help='width of top model')
    parser.add_argument('--p', default=0.3, type=float, help='top model dropout')

    # decoder wavenet hyperparameters
    parser.add_argument('--wave_hidden_state', default=256, type=int, help='no. filters for the dilated convolutions')
    parser.add_argument('--head_hidden_state', default=128, type=int, help='no. filters for the WaveNets top model')
    parser.add_argument('--num_dil_rates', default=8, type=int, help='depth of the WaveNet')
    parser.add_argument('--dec_kernel_size', default=3, type=int, help='WaveNet kernel size')

    # loss prefactor weights
    parser.add_argument('--nll_weight', default=1., type=float, help='NLL prefactor weight')
    parser.add_argument('--MI_weight', default=0.95, type=float, help='MI prefactor weight')
    parser.add_argument('--lambda_weight', default=2., type=float, help='MMD prefactor weight')
    parser.add_argument('--gamma_weight', default=1., type=float, help='discriminative prefactor weight')

    args = parser.parse_args()
    
    return args


@torch.no_grad()
def make_latent_preds(
        args: any,
        model: nn.Module,
        dataloader: any,
        Y_reg_true: torch.FloatTensor
    ) -> (
            torch.FloatTensor,
            torch.FloatTensor
    ):

    Y_reg_dl = torch.zeros_like(Y_reg_true)

    # placeholder for the model predictions
    Z_pred = torch.zeros(Y_reg_dl.shape[0], args.z_dim)
    
    # eval mode
    model.eval()

    left_idx, right_idx = 0, 0
    for ii, batch in tqdm(enumerate(dataloader)):

        X_temp, Y_reg_temp, Y_C_temp, Y_acc_temp = batch
        batch_size = X_temp.shape[0]

        # update right index
        right_idx += batch_size

        Z_pred_mu, Z_pred_var = model.inference(X_temp.permute(0,2,1).to(args.DEVICE))
        Z_pred_temp = model.reparam_trick(Z_pred_mu, Z_pred_var)
        
        # load empty tensors
        Z_pred[left_idx:right_idx,:] = Z_pred_temp
        Y_reg_dl[left_idx:right_idx,:] = Y_reg_temp

        left_idx = right_idx

    return (
            Z_pred.cpu(),
            Y_reg_dl.cpu()
    )


def infer_latents(
        args: any,
        model: nn.Module,
        train_dataloader: any,
        valid_dataloader: any
    ) -> (
            torch.FloatTensor,
            torch.FloatTensor,
            torch.FloatTensor,
            torch.FloatTensor
    ):

    train_dataset = train_dataloader.dataset
    valid_dataset = valid_dataloader.dataset

    # unwrap datasets
    X_train, Ytrain_reg_true, Ytrain_C_true, Ytrain_accept_True = train_dataset[0:] # train set
    X_valid, Yvalid_reg_true, Yvalid_C_true, Yvalid_accept_True = valid_dataset[0:] # valid set
 


    # training latent inference       
    Zpred_train, Ytrain_true_dl = make_latent_preds(
                        args=args,
                        model=model,
                        dataloader=train_dataloader,
                        Y_reg_true=Ytrain_reg_true
    )
    
    # validation latent inference
    Zpred_valid, Yvalid_true_dl = make_latent_preds(
                        args=args,
                        model=model,
                        dataloader=valid_dataloader,
                        Y_reg_true=Yvalid_reg_true
    )


    return (
            Zpred_train,
            Ytrain_true_dl,
            Zpred_valid,
            Yvalid_true_dl
    )




def aniso_func_dist(
    mean: torch.FloatTensor,
    std: torch.FloatTensor
    ) -> torch.distributions.multivariate_normal.MultivariateNormal:

    # identity matrix
    I = torch.eye(len(mean))
    # covariance matrix
    covar_matrix = torch.mul(I, std**2)

    # functional anisotropic dist
    aniso_dist = torch.distributions.multivariate_normal.MultivariateNormal(mean, covar_matrix)

    return aniso_dist

@torch.no_grad()
def sample_func_SH3(
    args: any,
    model: nn.Module,
    aniso_dist: torch.distributions.multivariate_normal.MultivariateNormal,
    n: int=100
    ) -> (
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor
    ):
    
    # eval mode
    model.eval()

    # set up the sequence context.
    X_context = torch.zeros((n,82,21)).to(args.DEVICE)
    X_context_cat = X_context.clone()
    X_context_greedy = X_context.clone()

    # latent-conditional info.
    Z_context = aniso_dist.sample((n,)).to(args.DEVICE)
    Z_NOcontext = torch.zeros_like(Z_context)

    # generate sequences
    X_samples_cat = model.sample(
        args=args,
        X_context=X_context_cat,
        z=Z_context,
        option='categorical'
    ).cpu()

    X_samples_argmax = model.sample(
        args=args,
        X_context=X_context_greedy,
        z=Z_context,
        option='greedy'
    ).cpu()

    X_samples_NOlatent = model.sample(
        args=args,
        X_context=X_context_cat,
        z=Z_NOcontext,
        option='categorical'
    ).cpu()

    return (
        X_samples_cat,
        X_samples_argmax,
        X_samples_NOlatent,
        Z_context.cpu(),
        Z_NOcontext.cpu()
    )

def create_seqs(X: torch.FloatTensor) -> list:

    # int to aa label
    int2token = {ii:label for ii, label in enumerate('ACDEFGHIKLMNPQRSTVWY-')}

    # convert one hot encoded sequences into amino acids
    num_seq = [
        list(seq) for seq in torch.argmax(X, dim = -1).cpu().numpy()[:,-1,:]
    ]

    # temp aa seq list
    temp_aa_seq = []
    # convert num to str
    for seq in num_seq:
        temp_aa_seq.append([int2token[num_label] for num_label in seq])


    # get list for aa string list
    aa_seq = []
    for seq in temp_aa_seq:
        aa_seq.append(''.join(seq).replace('-',''))


    return aa_seq

def create_df(
        args: any,
        aa_seqs: list,
        z: torch.FloatTensor,
    ) -> pd.Series:



    # create dictionary
    seq_dict = {}

    seq_dict['header'] = []

    # header column
    for ii, seq in enumerate(aa_seqs):
        seq_dict['header'].append(f'seq_{ii}')

    seq_dict['unaligned_sequence'] = aa_seqs

    # column names corresponding to latent coordinates
    column_name = [f'z_{ii}' for ii in range(z.shape[-1])]

    for ii, name in enumerate(column_name):
        seq_dict[name] = list(
            z.numpy()[:,ii]
        )


    return pd.DataFrame(seq_dict)

def convert_list_to_tensor(
        seq_list: list,
        max_seq_len: int,

    ) -> torch.FloatTensor:

    padded_seq_list = prep.pad_ends(seqs=seq_list,max_seq_length=max_seq_len)
    num_seq_list = prep.create_num_seqs(padded_seq_list)

    onehot_transformer = torch.eye(21)
    x_onehot = onehot_transformer[num_seq_list]

    return x_onehot

@torch.no_grad()
def diversify_gene(
    args: any,
    model: nn.Module,
    seq_list: list,
    max_seq_len: int,
    z_context: torch.FloatTensor,
    L: int
    ) -> torch.FloatTensor:
    
    # eval mode: 
    model.eval()
    
    # number of candidates 
    n = z_context.shape[0]
    
    # create torch tensor
    X = convert_list_to_tensor(
                seq_list=seq_list,
                max_seq_len=max_seq_len
    )
    
    if len(X.shape) == 2:
        X = X.unsqueeze(0).repeat(n,1,1)
    elif len(X.shape) != 3:
        quit
    else:
        pass
    
    # diversify sequence of interest
    X_diversify_samples = model.diversify(
            args=args,
            X_context=X.to(args.DEVICE),
            z=z_context.to(args.DEVICE),
            L=L,
            option='categorical'
    ).cpu()
    
    return X_diversify_samples
           

    
def randomly_diversify_gene(
    args: any,
    model: nn.Module,
    seq_list: list,
    max_seq_len: int,
    L: int
    ) -> torch.FloatTensor:
    
    # eval mode: 
    model.eval()
    
    # gap region
    num_gaps = max_seq_len - len(seq_list[0])
    
    # number of candidates 
    n = 5000
    
    # create torch tensor
    X = convert_list_to_tensor(
                seq_list=seq_list,
                max_seq_len=max_seq_len
    )
    
    if len(X.shape) == 2:
        X = X.unsqueeze(0).repeat(n,1,1)
    elif len(X.shape) != 3:
        quit
    else:
        pass
    
    X_temp = model.create_uniform_tensor(args=args,X=X)
    print(X_temp.shape)
    # diversify sequence of interest
    X_rand_diversify_samples = model.randomly_diversify(
                                            args=args,
                                            X_context=X.to(args.DEVICE),
                                            L=L,
                                            option='categorical'
    ).cpu()
    
    # include deletion gaps
    X_rand_diversify_samples[:, -1, -num_gaps:,:] = X[:,-num_gaps:, :]
    
    return X_rand_diversify_samples, X

def diversify_seq(
    args: any,
    model: nn.Module,
    aniso_dist: torch.distributions.multivariate_normal.MultivariateNormal,
    seq_list: list,
    max_seq_len: int,
    L: int,
    n: int=50,
    ) -> (
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor
    ):
    
    # eval mode
    model.eval()
    
    # latent-conditional info. 
    Z_context = aniso_dist.sample((n,)).to(args.DEVICE)
    Z_NOcontext = torch.zeros_like(Z_context).to(args.DEVICE)
    
    # latent conditioning
    X_latent_diversify = diversify_gene(
        args=args,
        model=model,
        seq_list=seq_list,
        max_seq_len=max_seq_len,
        z_context=Z_context,
        L=L
    ) 
    
    # no latent conditioning 
    X_NOlatent_diversify = diversify_gene(
        args=args,
        model=model,
        seq_list=seq_list,
        max_seq_len=max_seq_len,
        z_context=Z_NOcontext,
        L=L
    )


    # random diversification
    X_random_diversify, _ = randomly_diversify_gene(
            args=args,
            model=model,
            seq_list=seq_list,
            max_seq_len=max_seq_len,
            L=L
    )

    return (
        X_latent_diversify,
        X_NOlatent_diversify,
        Z_context,
        Z_NOcontext,
        X_random_diversify
    )


def create_func_aniso(
                Z: torch.FloatTensor,
                Y: torch.FloatTensor
    ) -> torch.distributions.multivariate_normal.MultivariateNormal: 

    # dataset embeddings that only correspond to functional sequences
    Z_func_pred = Z[(Y[:,0] > 0.5), :]

    # estimate mean and std of an anisotropic Gaussian dist.
    Z_func_mean = torch.mean(Z_func_pred, dim=0)
    Z_func_std = torch.std(Z_func_pred, dim=0)

    latent_func_aniso_dist = aniso_func_dist(
                                    mean=Z_func_mean,
                                    std=Z_func_std
    )

    return latent_func_aniso_dist


def design_SH3_LatentOnly(
            args: any,
            model: nn.Module,
            aniso_dist: torch.distributions.multivariate_normal.MultivariateNormal
    ) -> None:


    X_samples_cat, X_samples_argmax, X_NOlatent_samples, Z_sample_context, Z_sample_NOcontext = sample_func_SH3(
        args=args,
        model=model,
        aniso_dist=aniso_dist,
        n=300
    )      
        
    # generate dataframe for the amino acid sequences of the categorical sampling
    aa_seqs = create_seqs(
        X=X_samples_cat
    )

    cat_sample_df = create_df(
        args=args,
        aa_seqs=aa_seqs,
        z=Z_sample_context.cpu()
    )

    # generate dataframe for the amino acid sequences of the argmax sampling
    aa_seqs = create_seqs(
        X=X_samples_argmax
    )

    argmax_sample_df = create_df(
        args=args,
        aa_seqs=aa_seqs,
        z=Z_sample_context.cpu()
    )

    # generate dataframe for the amino acid sequences of the cat. sampling with no latent codes 
    aa_seqs = create_seqs(
        X=X_NOlatent_samples
    )

    Nocontext_sample_df = create_df(
        args=args,
        aa_seqs=aa_seqs,
        z=Z_sample_NOcontext.cpu()
    )

    # save the tensor predictions
    design_tensors = (X_samples_cat, X_samples_argmax, X_NOlatent_samples, Z_sample_context, Z_sample_NOcontext)
    torch.save(design_tensors, os.path.join(args.save_dir, f"LatentOnly_Sho1Designs.tensors"))

    # save dataframes
    cat_sample_df.to_csv(os.path.join(args.save_dir, f"LatentOnly_Sho1Designs[categorical].csv"), index=False)
    Nocontext_sample_df.to_csv(os.path.join(args.save_dir, f"NoLatent_Sho1Designs[categorical].csv"), index=False)
    argmax_sample_df.to_csv(os.path.join(args.save_dir, f"Latent_Sho1Designs[argmax].csv"), index=False)


    return 
    
def diversify_SH3_WT(
                args: any,
                model: nn.Module,
                aniso_dist: torch.distributions.multivariate_normal.MultivariateNormal,

    ) -> None:

    df = pd.read_csv(args.dataset_path) # all of the training smaples (nat, synthetic)
    WT_df = df[df.RE_norm == 1] # wild-type S.Cerevisiae spreadsheet
    WT_seq = list(WT_df.Sequences_unaligned)

    X_wt = convert_list_to_tensor(
                        max_seq_len=max_seq_len,
                        seq_list=WT_seq
    )

    for ii, perc in enumerate([0.25, 0.5, 0.75]):
        
        L = round( len(WT_seq[0].replace('-','')) * perc)
        
        # get data tensors
        Xwt_latent_diversify, Xwt_NOlatent_diversify, Zwt_context, Zwt_NOcontext, Xwt_random_diversify = diversify_seq(
                        args=args,
                        model=model,
                        aniso_dist=aniso_dist,
                        seq_list=WT_seq,
                        max_seq_len=args.max_seq_len,
                        L=L,
                        n=args.N
        )

        # generate dataframes for the amino acid sequences that used latent conditioning
        Xwt_diversify_seqs = create_seqs(
                X=Xwt_latent_diversify
        )
        Xwt_latent_df = create_df(
                args=args,
                aa_seqs=Xwt_diversify_seqs,
                z=Zwt_context.cpu()
        )
        
        # generate dataframes for the amino acid sequences that used latent conditoning
        Xwt_NOlatent_seqs = create_seqs(
                X=Xwt_NOlatent_diversify
        )

        Xwt_NOlatent_df = create_df(
                args=args,
                aa_seqs=Xwt_NOlatent_seqs,
                z=torch.zeros((len(Xwt_NOlatent_seqs), Zwt_context.cpu().shape[-1]))
        )
        

        # generate dataframes for the amino acids that randomly diversified 
        Xwt_random_seqs = create_seqs(
                X=Xwt_random_diversify
        )
        

        Xwt_random_df = create_df(
                args=args,
                aa_seqs=Xwt_random_seqs,
                z=torch.zeros((len(Xwt_random_seqs), Zwt_context.cpu().shape[-1]))
        )

        # save the tensor predictions 
        design_tensors = (Xwt_latent_diversify, Xwt_NOlatent_diversify, Zwt_context, Zwt_NOcontext, Xwt_random_diversify)
        torch.save(design_tensors, os.path.join(args.save_dir, f"WT_diversify[L={L}].tensors.{ii}"))

        # save datafranes
        Xwt_latent_df.to_csv(os.path.join(args.save_dir, f"WT_diversify[L={L}].csv"), index=False)
        Xwt_NOlatent_df.to_csv(os.path.join(args.save_dir, f"WT_diversify_NOlatent[L={L}].csv"), index=False)
        Xwt_random_df.to_csv(os.path.join(args.save_dir, f"WT_diversify_random[L={L}].csv"), index=False)


    return 


def compute_partial_bounds(
            nat_RE_norm: np.single
    ) -> (
        float,
        float
    ):
    
    from sklearn.mixture import GaussianMixture
    
    GM_params = [0.11614647287176139, 0.8232292945743522]

    # nonfunctional mode: 
    nonfunc_RE_norm = nat_RE_norm[ nat_RE_norm < GM_params[0] ]

    # functional mode:
    func_RE_norm = nat_RE_norm[ nat_RE_norm > GM_params[-1] ]
    
    # nonfunctional gaussian mode ...
    gm_nonfunc = GaussianMixture(n_components=1, random_state=42).fit(nonfunc_RE_norm.reshape(-1,1))

    # functional gaussian mode ...
    gm_func = GaussianMixture(n_components=1, random_state=42).fit(func_RE_norm.reshape(-1,1))
    
    # nonfunctional guassian mean and std 
    nonfunc_std = gm_nonfunc.covariances_[0][0] ** (1/2)
    nonfunc_mean = gm_nonfunc.means_[0]

    # functional guassian mean and std 
    func_std = gm_func.covariances_[0][0] ** (1/2)
    func_mean = gm_func.means_[0]
    
    # partial rescue region:
    upper_bound = func_mean - 2*func_std
    lower_bound = nonfunc_mean + 2*nonfunc_std
    
    return (
        upper_bound[0],
        lower_bound[0]
    )
    
def diversify_partial_rescue_paralog(
                            args: nn.Module,
                            model: nn.Module,
                            aniso_dist: torch.distributions.multivariate_normal.MultivariateNormal,
    ) -> None:

    df = pd.read_csv(args.dataset_path)
    nat_df = df[df.header.str.contains('nat')]
    partial_upper_bound, partial_lower_bound = compute_partial_bounds(
                                                            nat_RE_norm=np.array(nat_df.RE_norm.values)
    )

    # define partial rescue paralog dataframe
    paralog_nat_df = nat_df[~(nat_df.orthologous_group == 'NOG09120')]
    partial_para_nat_df = paralog_nat_df[
            (paralog_nat_df.RE_norm > partial_lower_bound) & (paralog_nat_df.RE_norm < partial_upper_bound)
    ]

    # drop miss paralog annotations...
    partial_para_nat_df = partial_para_nat_df.dropna(subset=['orthologous_group'])
    paralog_of_interest_df = partial_para_nat_df.loc[
            partial_para_nat_df.Sequences_unaligned == 'NKILFYVEAMYDYTATIEEEFNFQAGDIIAVTDIPDDGWWSGELLDEARREEGRHVFPSNFVRLF'
    ]
    
    # save the spreadsheet for the paralog of interest
    paralog_of_interest_df.to_csv(os.path.join(args.save_dir, f"PartialRescueParalog.csv"), index = False)

    partial_paralog_seq = list(paralog_of_interest_df.Sequences_unaligned)
    
    for ii, perc in enumerate([0.25, 0.5, 0.75]):

        L = round( len(partial_paralog_seq[0].replace('-','')) * perc)

        # get data tensors
        Xpartial_latent_diversify, Xpartial_NOlatent_diversify, Zpartial_context, Zpartial_NOcontext, Xpartial_random_diversify = diversify_seq(
                args=args,
                model=model,
                aniso_dist=aniso_dist,
                seq_list=partial_paralog_seq,
                max_seq_len=args.max_seq_len,
                L=L,
                n=args.N
        )

        # generate dataframes for the amino acid sequences that used latent conditioning + known amino acids 
        Xpartial_diversify_seqs = create_seqs(
                X=Xpartial_latent_diversify
        )
        Xpartial_latent_df = create_df(
                args=args,
                aa_seqs=Xpartial_diversify_seqs,
                z=Zpartial_context.cpu()
        )
        
        # generate dataframes for the amino acid sequences that used latent conditioning + known amino acids
        Xpartial_NOlatent_seqs = create_seqs(
                X=Xpartial_NOlatent_diversify
        )

        Xpartial_NOlatent_df = create_df(
                args=args,
                aa_seqs=Xpartial_NOlatent_seqs,
                z=torch.zeros((len(Xpartial_NOlatent_seqs), Zpartial_context.shape[-1]))
        )

        # generate dataframes for the amino acid sequences that used latent conditioning + known amino acids
        Xpartial_random_seqs = create_seqs(
                X=Xpartial_random_diversify
        )

        Xpartial_random_df = create_df(
                args=args,
                aa_seqs=Xpartial_random_seqs,
                z=torch.zeros((len(Xpartial_random_seqs), Zpartial_context.shape[-1]))
        )

        # save the tensor predictions 
        design_tensors = (Xpartial_latent_diversify, Xpartial_NOlatent_diversify, Zpartial_context, Zpartial_NOcontext, Xpartial_random_diversify)
        torch.save(design_tensors, os.path.join(args.save_dir, f"PartialParalog_diversify[L={L}].tensors.{ii}"))

        # save datafranes
        Xpartial_latent_df.to_csv(os.path.join(args.save_dir, f"PartialParalog_diversify[L={L}].csv"), index=False)
        Xpartial_NOlatent_df.to_csv(os.path.join(args.save_dir, f"PartialParalog_diversify_NOlatent[L={L}].csv"), index=False)
        Xpartial_random_df.to_csv(os.path.join(args.save_dir, f"PartialParalog_diversify_random[L={L}].csv"), index=False)


    return 

     
def diversify_paralog(
        args: nn.Module,
        model: nn.Module,
        aniso_dist: torch.distributions.multivariate_normal.MultivariateNormal,
    ) -> None:

    df = pd.read_csv(args.dataset_path)
    nat_df = df[df.header.str.contains('nat')]
    partial_upper_bound, partial_lower_bound = compute_partial_bounds(
                                                            nat_RE_norm=np.array(nat_df.RE_norm.values)
    )

    # define partial rescue paralog dataframe
    paralog_nat_df = nat_df[~(nat_df.orthologous_group == 'NOG09120')]
    nonfunc_para_nat_df = paralog_nat_df[
            paralog_nat_df.RE_norm < partial_lower_bound
    ]

    # drop miss paralog annotations...
    nonfunc_para_nat_df = nonfunc_para_nat_df.dropna(subset=['orthologous_group'])
    paralog_of_interest_df = nonfunc_para_nat_df.loc[
            nonfunc_para_nat_df.Sequences_unaligned == 'PKENPWATAEYDYDAAEDNELTFVENDKIINIEFVDDDWWLGELEKDGSKGLFPSNYVSLGN' 
    ]
    
    # save the spreadsheet for the paralog of interest
    paralog_of_interest_df.to_csv(os.path.join(args.save_dir, f"NonfuncParalog.csv"), index = False)

    nonfunc_paralog_seq = list(paralog_of_interest_df.Sequences_unaligned)
    
    for ii, perc in enumerate([0.25, 0.5, 0.75]):

        L = round( len(nonfunc_paralog_seq[0].replace('-','')) * perc)

        # get data tensors
        Xparalog_latent_diversify, Xparalog_NOlatent_diversify, Zparalog_context, Zparalog_NOcontext, Xparalog_random_diversify = diversify_seq(
                args=args,
                model=model,
                aniso_dist=aniso_dist,
                seq_list=nonfunc_paralog_seq,
                max_seq_len=args.max_seq_len,
                L=L,
                n=args.N
        )

        # generate dataframes for the amino acid sequences that used latent conditioning + known amino acids 
        Xparalog_diversify_seqs = create_seqs(
                X=Xparalog_latent_diversify
        )
        Xparalog_latent_df = create_df(
                args=args,
                aa_seqs=Xparalog_diversify_seqs,
                z=Zparalog_context.cpu()
        )
        
        # generate dataframes for the amino acid sequences that used latent conditioning + known amino acids
        Xparalog_NOlatent_seqs = create_seqs(
                X=Xparalog_NOlatent_diversify
        )

        Xparalog_NOlatent_df = create_df(
                args=args,
                aa_seqs=Xparalog_NOlatent_seqs,
                z=torch.zeros((len(Xparalog_NOlatent_seqs), Zparalog_context.shape[-1]))
        )

        # generate dataframes for the amino acid sequences that used latent conditioning + known amino acids
        Xparalog_random_seqs = create_seqs(
                X=Xparalog_random_diversify
        )

        Xparalog_random_df = create_df(
                args=args,
                aa_seqs=Xparalog_random_seqs,
                z=torch.zeros((len(Xparalog_random_seqs), Zparalog_context.shape[-1]))
        )

        # save the tensor predictions 
        design_tensors = (Xparalog_latent_diversify, Xparalog_NOlatent_diversify, Zparalog_context, Zparalog_NOcontext, Xparalog_random_diversify)
        torch.save(design_tensors, os.path.join(args.save_dir, f"NonfuncParalog_diversify[L={L}].tensors.{ii}"))

        # save datafranes
        Xparalog_latent_df.to_csv(os.path.join(args.save_dir, f"NonfuncParalog_diversify[L={L}].csv"), index=False)
        Xparalog_NOlatent_df.to_csv(os.path.join(args.save_dir, f"NonfuncParalog_diversify_NOlatent[L={L}].csv"), index=False)
        Xparalog_random_df.to_csv(os.path.join(args.save_dir, f"NonfuncParalog_diversify_random[L={L}].csv"), index=False)


    return 


def diversify_Sho1_orthology(
                args: any,
                model: nn.Module,
                aniso_dist: torch.distributions.multivariate_normal.MultivariateNormal,

    ) -> None:

    
    df = pd.read_csv(args.dataset_path) # all of the training smaples (nat, synthetic)
    sho1_df = df[df.orthologous_group == 'NOG09120'] # containing only sho1 orthologs
    sho1_df = sho1_df.dropna(subset=['RE_norm']) # remove sequences with no fitness measurements
    func_sho1_df = sho1_df.iloc[(sho1_df.RE_norm > 0.5).values] # only functional sho1 orthologs
    ortholog_df = func_sho1_df.iloc[np.argmax(func_sho1_df['perc_min_leven[dissimilarity]']),:].to_frame().T

    # save dataframe
    ortholog_df.to_csv(os.path.join(args.save_dir, f"Sho1Ortholog_diversify.csv"), index=False)

    # get sequence
    ortholog_seq = list(ortholog_df.Sequences_unaligned)

    for ii, perc in enumerate([0.25, 0.5, 0.75]):
        
        # number of amino acids to conditioned (region does not get diversified)
        L = round( len(ortholog_seq[0].replace('-','')) * perc)
        
        # get data tensors
        Xortho_latent_diversify, Xortho_NOlatent_diversify, Zortho_context, Zortho_NOcontext, Xortho_random_diversify = diversify_seq(
                        args=args,
                        model=model,
                        aniso_dist=aniso_dist,
                        seq_list=ortholog_seq,
                        max_seq_len=args.max_seq_len,
                        L=L,
                        n=args.N
        )

        # generate dataframes for the amino acid sequences that used latent conditioning
        Xortho_diversify_seqs = create_seqs(
                X=Xortho_latent_diversify
        )
        Xortho_latent_df = create_df(
                args=args,
                aa_seqs=Xortho_diversify_seqs,
                z=Zortho_context.cpu()
        )
        
        # generate dataframes for the amino acid sequences that used latent conditoning
        Xortho_NOlatent_seqs = create_seqs(
                X=Xortho_NOlatent_diversify
        )

        Xortho_NOlatent_df = create_df(
                args=args,
                aa_seqs=Xortho_NOlatent_seqs,
                z=torch.zeros((len(Xortho_NOlatent_seqs), Zortho_context.cpu().shape[-1]))
        )

        # generate dataframes for the amino acid sequences that used random diversification
        Xortho_random_seqs = create_seqs(
                X=Xortho_random_diversify
        )

        Xortho_random_df = create_df(
                args=args,
                aa_seqs=Xortho_random_seqs,
                z=torch.zeros((len(Xortho_random_seqs), Zortho_context.cpu().shape[-1]))
        )

        # save the tensor predictions 
        design_tensors = (Xortho_latent_diversify, Xortho_NOlatent_diversify, Zortho_context, Zortho_NOcontext, Xortho_random_diversify)
        torch.save(design_tensors, os.path.join(args.save_dir, f"Ortholog_diversify[L={L}].tensors.{ii}"))

        # save datafranes
        Xortho_latent_df.to_csv(os.path.join(args.save_dir, f"Ortholog_diversify[L={L}].csv"), index=False)
        Xortho_NOlatent_df.to_csv(os.path.join(args.save_dir, f"Ortholog_diversify_NOlatent[L={L}].csv"), index=False)
        Xortho_random_df.to_csv(os.path.join(args.save_dir, f"Ortholog_diversify_random[L={L}].csv"), index=False)

    return 


if __name__ == '__main__':

    args = get_args()
    ProtWaveVAE.set_GPU() # set GPU
    ProtWaveVAE.set_SEED(args=args) # set SEED (reproducibility)
   
    # get data
    train_dataloader, valid_dataloader, protein_len = ProtWaveVAE.get_data(args=args)
    train_dataset = train_dataloader.dataset
    valid_dataset = valid_dataloader.dataset
    Xtrain, _, _, _ = train_dataset[0:1]
    max_seq_len = Xtrain.shape[1] # (B, L, 21)
    args.max_seq_len = max_seq_len 
    
    # get model
    PL_model = ProtWaveVAE.get_model(
                            args=args,
                            protein_len=protein_len
    ).to(args.DEVICE)
    model = PL_model.model
    model.load_state_dict(torch.load(args.output_model_path))
    
    # infer latent embeddings using pretrained model
    Zpred_train, Ytrain_true_dl, _, _ = infer_latents(
                          args=args,
                          model=model,
                          train_dataloader=train_dataloader,
                          valid_dataloader=valid_dataloader
    )
    
    # create an anisotropic Gaussian distribution defined by the functional latent embeddings
    latent_func_aniso_dist = create_func_aniso(
                                        Z=Zpred_train,
                                        Y=Ytrain_true_dl
    )

    # collect and save design sequence spreadsheets: latent-only conditioning 
    design_SH3_LatentOnly(
        args=args,
        model=model,
        aniso_dist=latent_func_aniso_dist
    )

    # collect and save WT latent-diversification
    diversify_SH3_WT(
            args=args,
            model=model,
            aniso_dist=latent_func_aniso_dist
    )

    # collect and save partial paralog latent diversification
    diversify_partial_rescue_paralog(
                            args=args,
                            model=model,
                            aniso_dist=latent_func_aniso_dist
    )

    # collect and save Sho1 ortholog latent-diversification
    diversify_Sho1_orthology(
             args=args,
             model=model,
             aniso_dist=latent_func_aniso_dist
    )

    # collect and save paralog latent diversification
    diversify_paralog(
            args=args,
            model=model,
            aniso_dist=latent_func_aniso_dist
    )
