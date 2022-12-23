# Project subdirectory task: PFAM analysis

This subdirectory is dedicated to reproducing the protein family and Chorismate mutate inference and design tasks, as described in the paper "ProtWaveVAE: Integrating Autoregressive Sampling with Latent-based Inference for Data-driven Protein Design". It assumes that you have CUDA-compatible GPUs installed and have executed `pip install -r ../requirements.txt`.

## Running scripts

Follow these step-by-step instructions to reproduce the results:

### Chorismate mutase (CM) protein design task

1.) (Optional) Train unsupervised and semisupervised models from scratch. The pretrain model weights can be found in `./outputs/train_sess/CM/CM_US_model.pth` or `./outputs/train_sess/CM/CM_US_model.pth`. Here are the training shell scripts.

```
cd scripts/scripts_train_sess
sh train_CM_US.sh # unsupervised CM training
sh train_CM_SS.sh # semis-supervised CM trainin
```

Note: above steps can be skipped by loading the pretrained model weights

2.) To generate de novo proteins by only sampling latent codes, we can use the following shell scripts.

```
cd scripts/scripts_sample
sh sample_CM_US.sh
sh sample_CM_SS.sh
```

3.) For CM tasks, To diversify the C-terminus while latent conditioning, we can use the following shell scripts.

```
cd scripts/scripts_sample
sh diversify_CM_US.sh # unsupervised model's designs
sh diversify_CM_SS.sh # semi-supervised model's designs
```

4.) To infer latent embeddings, usie the following scripts.

```
cd scripts/scripts_sample
sh infer_z_CM_US.sh # inference with natural homologs using unsupervised model
sh infer_z_CM_SS.sh # inference with natural homologs using semi-supervised model
```

5.) To measure novelty of the CM designs, use the following shell scripts.

```
cd scripts/scripts_sample
sh leven_CM_US.sh # levenshtein distance measurements for unsupervised model designs
sh leven_CM_SS.sh # levenshtein distance measurements for semi-supervised model designs
```

6.) To compute the TMalign scores and RMSDs between CM design structure predictions and natural E.coli PDB structure, use the following shell scripts.

Note: PDB structure predictions need to be inside `./ProtWaveVAE/Pfam_analysis/TMalign/CM_US_pdbs` or `./ProtWaveVAE/Pfam_analysis/TMalign/CM_SS_pdbs`.

```
cd scripts/scripts_TM
# TM scores for de novo designs
sh compute_CM_US_TMscores.sh # unsupervised model
sh compute_CM_US_TMscores.sh # semi-supervised model
sh compute_DeNovo_CM_SS_TMscores.sh # semi-supervised model
# TM score for latent-conditioning + N-terminus conditioning designs
sh compute_Cterm_CM_SS_TMscores.sh# unsupervised model
```

7.) To convert the TMscore values from TMalign algorithm into a readable format, use the following shell scripts.

```
cd scripts/scripts_TM
# latent conditioning
sh extract_CM_US_TMscores.sh # unsupervised model
sh extract_CM_SS_TMscores.sh # semi-supervised model
sh extract_DeNovo_CM_SS_TMscores.sh # semi-supervised model
# N-term + latent conditioning
sh extract_Cterm_CM_SS_TMscores.sh # semi-supervised model
```

### Four protein family analysis task


1.) (Optional) Train unsupervised and semisupervised models from scratch. The pretrain model weights can be found in `./outputs/train_sess/*/*.pth`. Here are the training shell scripts.

```
cd scripts/scripts_train_sess
# training mode scripts
sh train_DHFR_pfam.sh # DHFR family
sh train_G_pfam.sh # Gprotein family
sh train_lacta_pfam.sh # Lactamase family
sh train_S1A_pfam.sh # S1A family
```
Note: above steps can be skipped by loading the pretrained model weights


2.) To generate de novo proteins by only sampling latent codes, we can use the following shell scripts.

```
cd scripts/scripts_sample
# sampling design scripts
sh sample_DHFR.sh # DHFR family
sh sample_G.sh # Gprotein family
sh sample_lacta.sh # lactamase family
sh sample_S1A.sh # S1A family
```

3.) For protein family tasks, to infer latent embeddings, usie the following scripts.

```
cd scripts/scripts_sample
# latent inference of sequences (e.g. natural homologs
sh infer_z_DHFR.sh.sh # DHFR family
sh infer_z_G.sh # Gprotein family
sh infer_z_lacta.sh # DHFR family
sh infer_z_S1A.sh # S1A family
```

4.) To measure novelty of the protein family designs, use the following shell scripts.

```
cd scripts/scripts_sample
sh leven_DHFR.sh # DHFR family
sh leven_G.sh # Gprotei nfamily
sh leven_lacta.sh # Lactamase family
sh leven_S1A.sh # S1A family
```

5.) To compute the TMalign scores and RMSDs between CM design structure predictions and natural E.coli PDB structure, use the following shell scripts.

Note: PDB structure predictions need to be inside `./ProtWaveVAE/Pfam_analysis/TMalign/`.

```
cd scripts/scripts_TM
# TM scores for pfam designs
sh compute_DHFR_TMscores.sh # DHFR family
sh compute_G_TMscores.sh # Gprotein family
sh compute_lactamase_TMscores.sh #  lactamase family
sh compute_S1A_TMscores.sh # S1A family
```

6.) To convert the TMscore values from TMalign algorithm into a readable format, use the following shell scripts

```
cd scripts/scripts_TM
# latent conditioning
sh extract_DHFR_TMscores.sh # DHFR family
sh extract_G_TMscores.sh # Gprotein family
sh extract_lactamase_TMscores.sh # Lactamase family
sh extract_S1A_TMscores.sh # S1A family
```

