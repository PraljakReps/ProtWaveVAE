#!/usr/bin/env sh

python -V
export DIR="$(dirname "$(pwd)")"
#source activate torch_GPU
export PYTHONPATH=${PYTHONPATH}:${DIR}




# ===========================
# = For compute levenshtein =
# ===========================

export SEED=42
export design_path='../.././outputs/prediction/Cterminus_design/CM/Cterminus_CM_sample_sequences_SS.csv'
export data_path='../.././data/protein_families/CM/CM_natural_homologs.csv'
export synthetic_path='../.././data/protein_families/CM/CM_synthetic_homologs.csv'
export des_seq_column='sequence'
export dataset_seq_column='Unaligned_sequence'
export output_path=${design_path}
export option=1
export homolog_option=0 # 0: CM


# data path
export design_Cterm_path='../../outputs/prediction/Cterminus_design/CM/Cterminus_CM_sample_sequences_SS.csv'
export design_Denovo_path='../../outputs/prediction/Cterminus_design/CM/DeNovo_CM_sample_sequences_SS.csv'

# save data path
export save_Cterm_output_path='../../outputs/prediction/Cterminus_design/CM/Cterm_CM_similarity_sequences_SS.csv'
export save_DeNovo_output_path='../../outputs/prediction/Cterminus_design/CM/DeNovo_CM_similarity_sequences_SS.csv'



python ../../compute_min_sim_CtermCM.py \
	--SEED ${SEED} \
	--design_path ${design_path} \
	--data_path ${data_path} \
	--synthetic_path ${synthetic_path} \
	--des_seq_column ${des_seq_column} \
	--dataset_seq_column ${dataset_seq_column} \
	--output_path ${output_path} \
	--option ${option} \
	--homolog_option ${homolog_option} \
	--design_Cterm_path ${design_Cterm_path} \
	--design_Denovo_path ${design_Denovo_path} \
	--save_Cterm_output_path ${save_Cterm_output_path} \
	--save_DeNovo_output_path ${save_DeNovo_output_path} 
	









