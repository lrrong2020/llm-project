#!/bin/bash
#SBATCH --job-name=werewolf
#SBATCH --partition=normal  # 如有 H800 分区请改为 h800
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=50:00:00
#SBATCH --account=msccsit2024
#SBATCH --output=logs/werewolf_%j.out
#SBATCH --error=logs/werewolf_%j.err

# 确保日志目录存在
mkdir -p logs output/basic output/sft

# 输出诊断信息
echo "Job started at $(date)"
echo "Running on host: $(hostname)"
echo "Working directory: $(pwd)"

# 加载模块
module purge
module load slurm
module load cuda12.2/toolkit/12.2.2
module load Anaconda3/2023.09-0

# 验证GPU可用
nvidia-smi || echo "Warning: nvidia-smi failed. Check GPU availability."

# 正确激活conda环境（在SLURM脚本中需要此方式）
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate werewolf

# 打印Python路径和包版本确认环境正确激活
which python
python -c "import torch; print(f'PyTorch version: {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
python -c "import swanlab; print(f'SwanLab version: {swanlab.__version__}')"

# 替换为您的SwanLab API密钥
SWANLAB_API_KEY="u8Y02Lpjiad3zWE2l1DpR"

# 1) 基础知识微调
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

# 检查上一条命令是否成功
if [ $? -ne 0 ]; then
    echo "Basic training failed. Exiting."
    exit 1
fi

# 2) SFT 微调
echo "===== 开始 SFT 微调 $(date) ====="
python train_werewolf.py --stage sft \
       --model_dir "Qwen/Qwen2.5-1.5B" \
       --data_dir . \
       --output_dir output \
       --per_device_train_batch_size 2 \
       --gradient_accumulation_steps 4 \
       --max_seq_len 4096 \
       --use_swanlab \
       --swanlab_api_key "$SWANLAB_API_KEY" \
       --swanlab_project "Werewolf-LoRA"

echo "===== 所有训练完成 $(date) =====" 