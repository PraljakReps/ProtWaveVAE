import subprocess
import os
import pandas as pd
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-pn', dest = 'protein_name', default = 'DHFR', type = str, help = 'Flag: Name of the protein')
    parser.add_argument('-op', dest = 'output_path', default = 'DHFR.csv', type = str, help = 'Flag: output path')
    parser.add_argument('-ip', dest = 'input_path', default = './sup_files', type = str, help = 'Flag: folder path containing all of the predictions')
    parser.add_argument('--output_folder_path', dest = 'output_folder_path', default = './TMresults', type = str, help = 'Flag: folder output path')
    args = parser.parse_args()

    protein_name = args.protein_name
    output_path = args.output_path
    input_path = args.input_path
   
    # result dictionary:
    result_dict = {
        'name': list(),
        'RMSD': list(),
        'TMscore': list(),
        'ID': list()
    }

    # read all of the computed TMaligned values
    TMalign_pred_list = os.listdir(f'{input_path}/{protein_name}')
    
    for TMalign_pathname in TMalign_pred_list:
        
        TMalign_pred_file = TMalign_pathname + '_atm'
        
        print(TMalign_pred_file)

        if '_unrelaxed_rank_' not in TMalign_pathname:
            pass

        else:
        
            with open(f'{input_path}/{protein_name}/' + str(TMalign_pathname) + '/' + str(TMalign_pred_file)) as f:


                lines = f.readlines()

            for line in lines:
                
                if 'REMARK Aligned length=' in line:
                    print(line)

                    RMSD_idx = line.find('RMSD= ')
                    TMscore_idx = line.find(', TM-score=')
                    ID_idx = line.find(', ID=')
            

                    RMSD = float( line[RMSD_idx+len('RMSD= '): TMscore_idx] )
                    TMscore = float( line[TMscore_idx+len(', TM-score='): ID_idx])
                    ID = float(line[ID_idx + len(', ID='):])
                
                    result_dict['name'].append(TMalign_pathname)
                    result_dict['RMSD'].append(RMSD)
                    result_dict['TMscore'].append(TMscore)
                    result_dict['ID'].append(ID)


                    print('RMSD='+str(RMSD) + ' | TMscore=' + str(TMscore) + ' | ID=' +str(ID))


    TMscore_result_df = pd.DataFrame(result_dict)
    print('Number of TMscore measurements:', TMscore_result_df.shape[0])
    os.makedirs(args.output_folder_path, exist_ok = True)
    TMscore_result_df.to_csv(f"{args.output_folder_path}/{args.output_path}", index = False)
    #TMscore_result_df.to_csv(f"./TMalign_results/{output_path}", index = False)
