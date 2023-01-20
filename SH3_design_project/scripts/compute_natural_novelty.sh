#!/usr/bin/env sh

python -V
export DIR="$(dirname "$(pwd)")"
#source activate torch_GPU
export PYTHONPATH=${PYTHONPATH}:${DIR}


# path variables
export dataset_path='.././data/ACS_SynBio_SH3_dataset.csv'
export output_results_path='.././outputs/novelty/SH3_NaturalTrainDataset_MinLeven.csv'
export SEED=42
export option='natural'

python ../compute_dataset_novelty.py \
		--dataset_path ${dataset_path} \
		--output_results_path ${output_results_path} \
		--SEED ${SEED} \
		--option ${option} \









	









