#!/bin/bash
#SBATCH --job-name=werewolf_train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=50:00:00
#SBATCH --partition=normal
#SBATCH --account=msccsit2024
#SBATCH --gres=gpu:1
#SBATCH --output=werewolf_train_%j.log

# Load modules
module purge  
module load cuda12.2/toolkit/12.2.2  
module load Anaconda3/2023.09-0

# Verify GPU availability
nvidia-smi

conda create -n werewolf python=3.10 -y
conda activate werewolf
pip install -r requirements.txt

# 提交训练
cd werewolf_game_reasoning
sbatch run_werewolf.sbatch