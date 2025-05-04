#!/bin/bash
# 创建并安装依赖的脚本

# 确保目录结构
mkdir -p logs output/basic output/sft

# 创建conda环境
# 直接使用 conda 命令而不是 conda activate（批处理模式下不生效）
module purge
module load cuda12.2/toolkit/12.2.2
module load Anaconda3/2023.09-0

# 创建环境
conda create -n werewolf python=3.10 -y

# 写入环境激活命令到配置文件（用于随后激活）
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate werewolf

# 安装依赖
pip install -r ../requirements.txt

echo "环境设置完成。现在可以使用 sbatch run_werewolf.sbatch 提交作业了。" 