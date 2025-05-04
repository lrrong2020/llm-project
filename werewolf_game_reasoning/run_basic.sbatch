#!/bin/bash
#SBATCH --job-name=werewolf_basic
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --account=msccsit2024
#SBATCH --output=logs/werewolf_basic_%j.out
#SBATCH --error=logs/werewolf_basic_%j.err

# 确保日志目录存在
mkdir -p logs output/basic

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
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# 替换为您的SwanLab API密钥
SWANLAB_API_KEY="u8Y02Lpjiad3zWE2l1DpR"

# 仅运行基础知识微调
echo "===== 开始基础知识微调 $(date) ====="
python train_werewolf.py --stage basic \
       --model_dir "Qwen/Qwen2.5-1.5B" \
       --data_dir . \
       --output_dir output \
       --per_device_train_batch_size 4 \
       --gradient_accumulation_steps 8 \
       --use_swanlab \
       --swanlab_api_key "$SWANLAB_API_KEY" \
       --swanlab_project "Werewolf-LoRA"

echo "===== 基础知识微调完成 $(date) =====" 