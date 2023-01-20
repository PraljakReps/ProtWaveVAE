#!/usr/bin/env sh

python -V
export DIR="$(dirname "$(pwd)")"
#source activate torch_GPU
export PYTHONPATH=${PYTHONPATH}:${DIR}


# path variables
export dataset_path='../.././data/ACS_SynBio_SH3_dataset.csv'
export output_results_path='../.././outputs/novelty/SH3_DesignTraingDataset_MinLeven.csv'
export SEED=42
export option='design' # not important variable here...
export file_path='../.././outputs/SH3_design_pool/NonfuncParalog/'
export output_df_path='../.././outputs/SH3_design_pool/NonfuncParalog/min_leven'

python ../../compute_design_pool_novelty.py \
		--dataset_path ${dataset_path} \
		--output_results_path ${output_results_path} \
		--SEED ${SEED} \
		--option ${option} \
		--file_path ${file_path} \
		--output_df_path ${output_df_path} \









	









