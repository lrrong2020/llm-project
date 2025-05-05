#!/bin/bash
#SBATCH --job-name=werewolf_sft_7000
#SBATCH --partition=normal   # 使用H800大内存GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G          # SFT需要更多内存
#SBATCH --time=24:00:00
#SBATCH --account=msccsit2024
#SBATCH --output=logs/werewolf_sft_v2_%j.out
#SBATCH --error=logs/werewolf_sft_v2_%j.err

# 确保日志和输出目录存在
mkdir -p logs output/sft

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

# 检查基础模型是否存在
if [ ! -d "output/basic" ]; then
    echo "错误: 未找到基础模型目录 output/basic，请先运行 basic 训练"
    exit 1
fi

# 替换为您的SwanLab API密钥
SWANLAB_API_KEY="u8Y02Lpjiad3zWE2l1DpR"

# 仅运行SFT微调 - 修改参数以解决之前的错误
echo "===== 开始 SFT 微调 (v2 - 禁用FP16) $(date) ====="
python train_werewolf.py --stage sft \
       --model_dir "Qwen/Qwen2.5-1.5B" \
       --data_dir . \
       --output_dir output \
       --per_device_train_batch_size 1 \
       --per_device_eval_batch_size 1 \
       --gradient_accumulation_steps 4 \
       --max_seq_len 8192 \
       --learning_rate 1e-4 \
       --use_bf16 \
       --use_swanlab \
       --swanlab_api_key "$SWANLAB_API_KEY" \
       --swanlab_project "Werewolf-LoRA"


echo "===== SFT 微调完成 $(date) =====" 