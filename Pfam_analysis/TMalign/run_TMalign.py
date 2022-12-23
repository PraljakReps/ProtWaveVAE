import subprocess
import os
import argparse



def create_output_folders(
        args: any,
        foldername: str,
        protein_name: str
    ):
    
    fname = foldername.replace('.pdb','')
    run_make_out_folders = [
                'mkdir',
                f'{args.output_path}/{protein_name}/{fname}'
    ]
    
    subprocess.run(run_make_out_folders)
    

def run_TMalign_func(
    args: any,
    ref_pdb: str,
    target_pdb: str,
    data_path: str,
    foldername: str,
    protein_name: str
    ):
    
    # remove pdb substring
    output_foldername = foldername.replace('.pdb', '')
 
    #   './TMalign', 
    run_TMalign_list = [
        args.TMalign_path,
        ref_pdb,
        f'{data_path}/{foldername}',
        '-o',
        f'{args.output_path}/{protein_name}/{output_foldername}/{output_foldername}'
   ]


    # outputfolder: #  f'./sup_files/{protein_name}/{output_foldername}/{output_foldername}'
 
    subprocess.run(run_TMalign_list)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-rp', dest = 'ref_pdb', default = '1rx2.pdb', type = str, help = 'Flag: reference pdb name')
    parser.add_argument('-dp', dest = 'data_path', default = './DHFR_pdbs', type = str, help = 'Flag: path to the AF predicted pdbs name')
    parser.add_argument('-pn', dest = 'protein_name', default = './DHFR', type = str, help = 'Flag: protein name')
    parser.add_argument('--TMalign_path', dest = 'TMalign_path', default = '../.././TMalign/TMalign', type = str, help = 'Flag: TMalign algorithm path')
    parser.add_argument('--output_path', dest = 'output_path', default = '../.././TMalign/sup_files', type = str, help = 'Flag: TMalign output folder path')
    
    args = parser.parse_args()


    ref_pdb_filename = args.ref_pdb
    data_path = args.data_path
    protein_name = args.protein_name
    
    os.makedirs(f'{args.output_path}/{protein_name}',exist_ok = True)
    AFpred_pdb_filepaths = os.listdir(data_path)
    
    for AF_filepath in AFpred_pdb_filepaths:
        
        create_output_folders(
                    args=args,
                    foldername = AF_filepath,
                    protein_name = protein_name
        )
 
        run_TMalign_func(
            args=args,
            ref_pdb=ref_pdb_filename,
            target_pdb=AF_filepath,
            data_path=data_path,
            foldername=AF_filepath,
            protein_name=protein_name
        )
 

