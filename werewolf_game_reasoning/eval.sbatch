#!/bin/bash
#SBATCH --job-name=werewolf_eval
#SBATCH --partition=normal   # 使用H800大内存GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G          # SFT需要更多内存
#SBATCH --time=2:00:00
#SBATCH --account=msccsit2024
#SBATCH --output=logs/werewolf_eval_%j.out
#SBATCH --error=logs/werewolf_eval_%j.err

# 确保日志目录存在
mkdir -p logs

# 加载模块
module purge
module load slurm
module load cuda12.2/toolkit/12.2.2
module load Anaconda3/2023.09-0

# 验证GPU可用
nvidia-smi

# 正确激活conda环境
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate werewolf

# 检查环境
which python
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, 设备: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"

# 检查SFT模型是否存在
if [ ! -d "output/sft" ]; then
    echo "错误: 未找到SFT模型目录 output/sft，请先完成SFT训练"
    exit 1
fi

# 查找最新检查点
LATEST_CHECKPOINT=$(ls -1d output/sft/checkpoint-* | sort -V | tail -n 1)
if [ -z "$LATEST_CHECKPOINT" ]; then
    echo "错误: 未找到检查点目录，请确认SFT训练已正确完成"
    exit 1
fi

echo "找到最新检查点: $LATEST_CHECKPOINT"

# 运行评估脚本
echo "===== 开始评估SFT模型 $(date) ====="
python eval_sft.py \
       --model_dir "$LATEST_CHECKPOINT" \
       --base_model "Qwen/Qwen2.5-1.5B" \
       --data_dir . \
       --batch_size 4 \
       --num_examples 30

echo "===== 评估完成 $(date) =====" 