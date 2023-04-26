
# ProtWaveVAE Repository

## Project Description

This repository contains source code and scripts for reproducing the results of the paper titled ["ProtWaveVAE: Integrating Autoregressive Sampling with Latent-based Inference for Data-driven Protein Design"](https://doi.org/10.1101/2023.04.23.537971). The project is divided into three subfolders, each containing scripts and source code for reproducing specific tasks discussed in the paper:

1. `Benchmark_project`: Contains scripts and instructions for reproducing fitness and function benchmarking tasks from TAPE and FLIP using ProtWave-VAE.
2. `Pfam_analysis`: Contains scripts and instructions for reproducing protein family latent inference studies, Chorismate mutase semi-supervised learning tasks, and C-terminus diversification with latent conditioning.
3. `SH3_design_project`: Contains scripts and instructions for designing protein sequences that were experimentally tested.

For detailed instructions and steps, navigate to one of the three folders.

## Installation and Setup

Follow these step-by-step instructions to install and set up the project, including downloading and installing any dependencies:

1. Create a new virtual environment (e.g., use a conda environment):

```
conda create --name ProtWaveVAE_env python=3.8
```

Optionally, upgrade pip:

```
python -m pip install --upgrade pip
```

2. Activate the environment and install library packages:
```
source activate ProtWaveVAE_env
pip install -r requirements.txt
```

3. Enter the directory for reproducing tasks and follow the task-specific instructions in the given directory's README.md:
```
# example for entering the directory for reproducing the benchmark tasks
cd Benchmark_project
```

## Dependencies and Hardware Requirements

This project requires the following dependencies:

- PyTorch
- torchvision
- PyTorch-lightning

Please note that for training PyTorch models, some of the dependencies require an NVIDIA GPU with CUDA support. If your system does not have an NVIDIA GPU, you can still run the code, but the training process will be significantly slower as it will use the CPU for computation.

To check if your system has an NVIDIA GPU and if it supports CUDA, you can visit the [NVIDIA CUDA GPUs page](https://developer.nvidia.com/cuda-gpus).



