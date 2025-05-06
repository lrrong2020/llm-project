#!/bin/bash
#SBATCH --job-name=compare_models
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=2:00:00
#SBATCH --account=msccsit2024
#SBATCH --output=logs/compare_models_%j.out
#SBATCH --error=logs/compare_models_%j.err

# 确保日志目录存在
mkdir -p logs

# 加载模块
module purge
module load slurm
module load cuda12.2/toolkit/12.2.2
module load Anaconda3/2023.09-0

# 激活conda环境
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate werewolf

# 查找最新的SFT检查点
LATEST_CHECKPOINT=$(ls -1d output/sft/checkpoint-* | sort -V | tail -n 1)
if [ -z "$LATEST_CHECKPOINT" ]; then
    echo "错误: 未找到SFT检查点目录"
    exit 1
fi

echo "使用SFT检查点: $LATEST_CHECKPOINT"
echo "使用Basic检查点: output/basic"

# 运行比较脚本
echo "===== 开始比较 Basic 与 SFT 模型 $(date) ====="
python compare_models.py \
       --basic_model_dir output/basic \
       --sft_model_dir "$LATEST_CHECKPOINT" \
       --base_model "Qwen/Qwen2.5-1.5B" \
       --data_dir . \
       --num_examples 20

echo "===== 比较完成 $(date) ====="