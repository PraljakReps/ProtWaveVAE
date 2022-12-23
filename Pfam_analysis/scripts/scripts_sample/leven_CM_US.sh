#!/usr/bin/env sh

python -V
export DIR="$(dirname "$(pwd)")"
#source activate torch_GPU
export PYTHONPATH=${PYTHONPATH}:${DIR}




# ===========================
# = For compute levenshtein =
# ===========================

export SEED=42
export design_path='../.././outputs/prediction/pfam/CM/CM_sample_sequences_US.csv'
export data_path='../.././data/protein_families/CM/CM_natural_homologs.csv'
export synthetic_path='../.././data/protein_families/CM/CM_synthetic_homologs.csv'
export des_seq_column='sequence'
export dataset_seq_column='Unaligned_sequence'
export output_path=${design_path}
export option=1


python ../../compute_min_levenshtein_CM.py \
	--SEED ${SEED} \
	--design_path ${design_path} \
	--data_path ${data_path} \
	--synthetic_path ${synthetic_path} \
	--des_seq_column ${des_seq_column} \
	--dataset_seq_column ${dataset_seq_column} \
	--output_path ${output_path} \
	--option ${option} \



	









