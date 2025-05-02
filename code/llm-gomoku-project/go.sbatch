#!/bin/bash
#SBATCH --job-name=gomoku_train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --partition=normal
#SBATCH --account=msccsit2024
#SBATCH --gres=gpu:1
#SBATCH --output=gomoku_train_%j.log

# Load modules
module purge  
module load cuda12.2/toolkit/12.2.2  
module load Anaconda3/2023.09-0

# Verify GPU availability
nvidia-smi

# Set up environment
source $(conda info --base)/etc/profile.d/conda.sh
conda create -n gomoku_env python=3.10 -y
conda activate gomoku_env

# Install compatible versions with CUDA support
pip install torch==2.1.0 torchvision==0.16.0 --extra-index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install datasets pydantic pandas tqdm wandb peft numpy packaging

# Force GPU settings
export CUDA_VISIBLE_DEVICES=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TOKENIZERS_PARALLELISM=false

# Monitor GPU before training
echo "GPU before training:"
nvidia-smi

# Run pipeline with larger batch size and fewer frozen layers
python run_pipeline.py --train --model gpt2-medium --epochs 1 --output_dir models --batch_size 16 --frozen_layers 12

# Monitor GPU after training
echo "GPU after training:"
nvidia-smi