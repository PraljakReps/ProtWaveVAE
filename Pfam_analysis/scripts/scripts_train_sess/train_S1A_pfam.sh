#!/usr/bin/env sh

python -V
export DIR="$(dirname "$(pwd)")"
#source activate torch_GPU
export PYTHONPATH=${PYTHONPATH}:${DIR}


# path variables
export data_path='../.././data/protein_families/S1A/pfam_S1A.csv'
export alignment=False
export output_results_path='../.././outputs/train_sess/pfam/S1A/training_results.csv'
export model_output_path='../.././outputs/train_sess/pfam/S1A/S1A_model.pth'
export dataset_split=0 # 1: train/valid | 0: train

# model training variables
export SEED=42
export homolog_option=1 # 0: CM | 1: S1A | 2: lactamase | 3: Gprotein | 4: DHFR
export epochs=500 #500
export DEVICE='cuda'

# model configuration
# model hps
export z_dim=4
export class_labels=21

# encoder
export encoder_rates=0
export C_in=21
export C_out=256
export alpha=0.1
export enc_kernel=3
export num_fc=3

# generator
export wave_hidden_state=128
export head_hidden_state=512
export num_dil_rates=5
export dec_kernel_size=3

# loss weights
export xi_weight=10
export alpha_weight=0.99
export lambda_weight=1
export lr=1e-4




python ../../train_on_pfam.py \
		--data_path ${data_path} \
		--alignment ${alignment} \
		--output_results_path ${output_results_path} \
		--model_output_path ${model_output_path} \
		--dataset_split ${dataset_split} \
		--SEED ${SEED} \
		--homolog_option ${homolog_option} \
                --epochs ${epochs} \
		--DEVICE ${DEVICE} \
		--z_dim ${z_dim} \
		--class_label ${class_labels} \
		--encoder_rates ${encoder_rates} \
		--C_in ${C_in} \
		--C_out ${C_out} \
		--alpha ${alpha} \
		--enc_kernel ${enc_kernel} \
		--num_fc ${num_fc} \
		--wave_hidden_state ${wave_hidden_state} \
		--head_hidden_state ${head_hidden_state} \
		--num_dil_rates ${num_dil_rates} \
		--dec_kernel_size ${dec_kernel_size} \
		--xi_weight ${xi_weight} \
		--alpha_weight ${alpha_weight} \
		--lambda_weight ${lambda_weight} \
		--lr ${lr} \









	









