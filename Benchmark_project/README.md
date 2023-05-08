# Project Subdirectory Task: Benchmark fitness and function

This subdirectory is dedicated to benchmarking fitness and function of FLIP and TAPE community datsasets. Download FLIP and TAPE datasets here:

TAPE --> ["Fluorescence (GFP)"](http://s3.amazonaws.com/songlabdata/proteindata/data_raw_pytorch/fluorescence.tar.gz) | ["Stability"](http://s3.amazonaws.com/songlabdata/proteindata/data_raw_pytorch/stability.tar.gz)
FLIP --> ["AAV"](https://www.google.com/url?q=https%3A%2F%2Fgithub.com%2FJ-SNACKKB%2FFLIP%2Fraw%2Fmain%2Fsplits%2Faav%2Ftasks.zip&sa=D&sntz=1&usg=AOvVaw1HlW_RQxAAvqAPR43UmSWU) | ["GB1"](https://www.google.com/url?q=https%3A%2F%2Fgithub.com%2FJ-SNACKKB%2FFLIP%2Fraw%2Fmain%2Fsplits%2Fgb1%2Ftasks.zip&sa=D&sntz=1&usg=AOvVaw0erzALqXduWLOKiuKdjhi7)

Make sure the datasets are unzipped and prepared in the `./data/AAV`, `./data/GB1`, `./data/GFP`, and `./data/stability` subfolders. Instead of download raw dataset and preparing the spreadsheets for reproducing the results shown here, you can download the spreadsheet directly from the following hugging face hub: ['ProtWaveVAE_benchmark_datasets'](https://huggingface.co/niksapraljak1/ProtWaveVAE_benchmark_datasets/tree/main).
 
0.) Prepare datasets

```
# make sure we are in the ProtWaveVAE/Benchmark_project directory...
git clone https://huggingface.co/niksapraljak1/ProtWaveVAE_benchmark_datasets
mkdir data
mv ProtWaveVAE_benchmark_datasets/* ./data
```


1.) (Optional) To hyperparameter optimize model on the benchmark tasks, follow the shell scripts in `./scripts/HPoptim`.

2.) To benchmark model on the tasks, follow the shell scripts.

```
cd scripts

# GB1 tasks with corresponding dataset split
sh benchmark_GB1_split0.sh
sh benchmark_GB1_split1.sh
sh benchmark_GB1_split2.sh
sh benchmark_GB1_split3.sh
sh benchmark_GB1_split4.sh

# AAV tasks with corresponding dataset split
sh benchmark_AAV_split0.sh
sh benchmark_AAV_split1.sh
sh benchmark_AAV_split2.sh
sh benchmark_AAV_split3.sh
sh benchmark_AAV_split4.sh
sh benchmark_AAV_split5.sh
sh benchmark_AAV_split6.sh

# GFP task
sh benchmark_GFP.sh

# stability task
sh benchmark_stability.sh
```
