#!/usr/bin/env sh

python -V
export DIR="$(dirname "$(pwd)")"
#source activate torch_GPU
export PYTHONPATH=${PYTHONPATH}:${DIR}

# ==================================
# = Varibles from training session =
# ==================================

# path variables
export data_path='../.././data/protein_families/CM/CM_natural_homologs.csv'
export train_path='../.././data/protein_families/CM/CM_natural_homologs.csv'
export test_path='../.././data/protein_families/CM/CM_synthetic_homologs.csv'
export alignment=False
export output_results_path='../.././outputs/train_sess/pfam/CM/training_SS_results.csv'
export model_output_path='../.././outputs/train_sess/pfam/CM/CM_model.pth'
export dataset_split=0 # 1: train/valid | 0: train

# model training variables
export SEED=42
export homolog_option=0 # 0: CM | 1: S1A | 2: lactamase | 3: Gprotein | 4: DHFR
export epochs=300 #500
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
export num_fc=0

# generator
export wave_hidden_state=64
export head_hidden_state=512
export num_dil_rates=8
export dec_kernel_size=3

# loss weights
export xi_weight=1
export alpha_weight=0.99
export lambda_weight=10
export lr=1e-4



# ==========================
# = For sampling sequences =
# ==========================

export folder_path='../.././outputs/prediction/pfam/CM'
export samples_output_path='../.././outputs/prediction/pfam/CM/CM_sample_sequences_SS.csv'
export weights_path=${model_output_path}

# ========================
# = Only relevant for SS =
# ========================

# training variable for CM
export learning_option='semi-supervised'
export disc_num_layers=2
export hidden_width=10
export num_classes=1
export p=0.3
export gamma_weight=1




python ../../infer_CM_latents.py \
		--folder_path ${folder_path} \
		--samples_output_path ${samples_output_path} \
		--weights_path ${weights_path} \
		--data_path ${data_path} \
		--train_path ${train_path} \
		--test_path ${test_path} \
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
		--learning_option ${learning_option} \
		--disc_num_layers ${disc_num_layers} \
		--hidden_width ${hidden_width} \
		--num_classes ${num_classes} \
		--p ${p} \
		--gamma_weight ${gamma_weight} \








	









