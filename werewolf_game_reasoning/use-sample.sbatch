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

#here maybe use conda to manage environemnts, to be added!

#here maybe install some dependencies, to be added!

#this script is to be submitted to computing cluster, mayebe use python xxx.py, just put it here




#to submit the script, just use sbatch use-sample.sbatch and then you will get the jobid.