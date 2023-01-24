#!/usr/bin/env sh

python -V
export DIR="$(dirname "$(pwd)")"
#source activate torch_GPU
export PYTHONPATH=${PYTHONPATH}:${DIR}

export ref_pdb='../.././TMalign/1ecm.pdb' # DHFR pdb file
export data_path='../.././TMalign/CM_SS_pdbs'
export protein_name='CM_SS'
export TMalign_path='../.././TMalign/TMalign'
export output_path='../.././TMalign/sup_files'

# path variables
python ../.././TMalign/run_TMalign.py \
	-rp ${ref_pdb} \
	-dp ${data_path} \
	-pn ${protein_name} \
	--TMalign_path ${TMalign_path} \
	


