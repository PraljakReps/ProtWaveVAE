#!/usr/bin/env sh

python -V
export DIR="$(dirname "$(pwd)")"
#source activate torch_GPU
export PYTHONPATH=${PYTHONPATH}:${DIR}

export protein_name='S1A'
export input_path='../.././TMalign/sup_files'
export output_path='S1A_TMscore_temp.csv'
export output_folder_path='../.././TMalign/TMscore_results'

# path variables
python ../.././TMalign/extract_results.py \
	-pn ${protein_name} \
	-op ${output_path} \
	-ip ${input_path} \
	--output_folder_path ${output_folder_path} \
	


