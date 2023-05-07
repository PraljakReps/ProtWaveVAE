# Project Subdirectory Task

This subdirectory is dedicated to reproducing the SH3 protein design experiment, as described in the paper "ProtWaveVAE: Integrating Autoregressive Sampling with Latent-based Inference for Data-driven Protein Design". It assumes that you have CUDA-compatible GPUs installed and have executed pip install -r ../requirements.txt.
## Running Scripts

Follow these step-by-step instructions to reproduce the results:

1. (Optional) Run the hyperparameter optimization protocols. This computationally expensive step can be skipped if you want to use the reported hyperparameters instead. Navigate to the scripts/HPoptim directory and follow the README.md instructions for hyperparameter optimization.

```
cd scripts/HPoptim
```

2. (Optional) After finding the optimal hyperparameters for the SH3 design task, you can run a Cross-validation script to verify that the model doesn’t overfit the training set. Execute the following command:  
```
cd scripts # Enter this subdirectory for the remaining shell scripts
sh CV_SH3_ProtWaveVAE.sh
```
Note: If you experience GPU memory allocation issues, lower the batch_size parameter in the shell script. This solution applies to the following scripts as well. 

3. Train the final configuration model for the SH3 protein design task. (Optional) You can skip this step and load the pretrained weights for the next scripts in inference and generation. The pretrained weights are located at `../outputs/SH3_task/final_model/final_ProtWaveVAE_SSTrainingHist.pth`. Run the following command:  
```
sh final_train_SH3_ProtWaveVAE.sh
```

4. Using the pretrained weights from the previous step, run an inference using the encoder to obtain latent vectors for the training dataset. Execute the following command:  

```
sh infer_natural_latents.sh    
```
5. To generate novel sequences by sampling the latent space and reproducing the paper's results, run the following command:  

```
sh generate_SH3_design_pool.sh    
```

6. Perform exploratory data analysis (EDA) to compute the novelty of the sequences, including both natural and synthetic design homologs. Run the following commands: 

```
sh compute_natural_novelty.sh 
sh compute_design_novelty.sh
sh compute_design_pool_novelty.sh  
```

7. To compute the novelty in terms of Levenshtein distances for the design pools, run the following commands:

```
sh compute_leven_latent.sh # de novo sampled designs
sh compute_leven_WTdiversify.sh # wild-type C-terminus inpainted designs
sh compute_leven_Orthologdiversify.sh # weak-binding ortholog C-terminus inpainted designs
sh compute_leven_PartialParadiversify.sh # weak-binding paralog C-terminus inpainted designs
sh compute_leven_NonfuncParadiversify.sh # Non-functional paralog in Sho1-pathway, C-terminus inpainted designs   
```

8. To generate randomly mutated C-terminus regions for control, run the following commands: bash

```
sh randomly_mutate_ORTHOdesigns.sh # random mutated proteins were experimentally tested
sh randomly_mutate_PARTIALdesigns.sh # random mutated proteins were experimentally tested
sh randomly_mutate_PARALOGdesigns.sh # omitted for oligo pool restriction
sh randomly_mutate_WTdesigns.sh # omitted for oligo pool restriction    
```



