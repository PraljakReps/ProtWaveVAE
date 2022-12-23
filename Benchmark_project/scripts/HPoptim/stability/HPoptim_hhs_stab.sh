#!/usr/bin/env sh

python -V
export DIR="$(dirname "$(pwd)")"
#source activate torch_GPU
export PYTHONPATH=${PYTHONPATH}:${DIR}


# path variables
export data_path='stability.csv' # do not need it
export train_path='../../.././data/stability/stability_train.json'
export valid_path='../../.././data/stability/stability_valid.json'
export test_path='../../.././data/stability/stability_test.json'
export output_results_path='../../.././outputs/HPoptim/stability/stability_hhs'
export output_model_path='../../.././outputs/HPoptim/stability/final_model/stability_hhs.pth'
export output_folder_path='../../.././outputs/HPoptim/stability'
export protein='stability'


# model training variables
export SEED=42
export batch_size=256
export epochs=300 # 300 
export lr=1e-4
export DEVICE='cuda'
export split_option=0

# general architecture variables
export z_dim='11'
export num_classes=1

# encoder hyperparameters
export encoder_rates='0'
export C_in=21
export C_out='512'
export alpha=0.1 # might not be necessary (Only for leaky relu)
export enc_kernel='3'
export num_fc='0'

# top model (discriminative decoder) hyperparameters
export disc_num_layers='2'
export hidden_width='10'
export p='0.1'

# decoder wavenet hyperparameters
export wave_hidden_state='64'
export head_hidden_state='32,64,128,256,512'
export num_dil_rates='8'
export dec_kernel_size='3'
export aa_labels=21

# loss prefactor
export nll_weight='1.0'
export MI_weight='0.95'
export lambda_weight='2.0'
export gamma_weight='1.0'

export search_variable='head_hidden_state'
export n_trials=5
export K=1


python ../../../HPoptim_benchmark_ProtWaveVAE.py \
		--data_path ${data_path} \
		--train_path ${train_path} \
		--valid_path ${valid_path} \
		--test_path ${test_path} \
		--output_results_path ${output_results_path} \
		--output_model_path ${output_model_path} \
		--output_folder_path ${output_folder_path} \
		--protein ${protein} \
		--SEED ${SEED} \
		--batch_size ${batch_size} \
                --epochs ${epochs} \
		--lr ${lr} \
		--DEVICE ${DEVICE} \
		--z_dim ${z_dim} \
		--num_classes ${num_classes} \
                --aa_labels ${aa_labels} \
		--encoder_rates ${encoder_rates} \
		--C_in ${C_in} \
		--C_out ${C_out} \
		--enc_kernel ${enc_kernel} \
		--num_fc ${num_fc} \
		--disc_num_layers ${disc_num_layers} \
		--hidden_width ${hidden_width} \
		--p ${p} \
		--wave_hidden_state ${wave_hidden_state} \
                --head_hidden_state ${head_hidden_state} \
                --num_dil_rates ${num_dil_rates} \
                --dec_kernel_size ${dec_kernel_size} \
                --aa_labels ${aa_labels} \
		--nll_weight ${nll_weight} \
                --MI_weight ${MI_weight} \
                --lambda_weight ${lambda_weight} \
                --gamma_weight ${gamma_weight} \
		--search_variable ${search_variable} \
		--n_trials ${n_trials} \
		--K ${K} \












	









