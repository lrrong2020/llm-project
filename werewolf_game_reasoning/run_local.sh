#!/bin/bash
# 本地模式训练脚本（不需要SwanLab账号，结果保存在本地）

# 确保目录存在
mkdir -p logs output/basic output/sft

# 加载模块
module purge
module load slurm
module load cuda12.2/toolkit/12.2.2
module load Anaconda3/2023.09-0

# 激活环境
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate werewolf

echo "===== 开始基础知识微调 (本地模式) $(date) ====="
python train_werewolf.py --stage basic \
       --model_dir "Qwen/Qwen2.5-1.5B" \
       --data_dir . \
       --output_dir output \
       --per_device_train_batch_size 4 \
       --gradient_accumulation_steps 8 \
       --use_swanlab \
       --use_local_swanlab

# 检查是否成功
if [ $? -ne 0 ]; then
    echo "基础训练失败。退出。"
    exit 1
fi

echo "==== 基础训练完成，本地训练日志位于 swanlab-local/ 目录 =====" 