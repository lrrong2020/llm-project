#!/bin/bash
#SBATCH --job-name=werewolf_sft
#SBATCH --partition=normal   
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G          # SFT需要更多内存
#SBATCH --time=24:00:00
#SBATCH --account=msccsit2024
#SBATCH --output=logs/werewolf_sft_%j.out
#SBATCH --error=logs/werewolf_sft_%j.err

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
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# 检查基础模型是否存在
if [ ! -d "output/basic" ]; then
    echo "错误: 未找到基础模型目录 output/basic，请先运行 basic 训练"
    exit 1
fi

# 替换为您的SwanLab API密钥
SWANLAB_API_KEY="kY3I7h6Paz1AI3MiF2CUV"

# 仅运行SFT微调
echo "===== 开始 SFT 微调 $(date) ====="
# python train_werewolf.py --stage sft \
#        --model_dir "Qwen/Qwen2.5-1.5B" \
#        --data_dir . \
#        --output_dir output \
#        --per_device_train_batch_size 2 \
#        --gradient_accumulation_steps 4 \
#        --max_seq_len 4096 \
#        --use_swanlab \
#        --swanlab_api_key "$SWANLAB_API_KEY" \
#        --swanlab_project "Werewolf-LoRA"

# 212821(手动取消)
# python train_werewolf.py --stage sft \
#        --model_dir "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" \
#        --data_dir . \
#        --output_dir output \
#        --per_device_train_batch_size 2 \
#        --gradient_accumulation_steps 4 \
#        --max_seq_len 6000 \
#        --num_train_epochs 1 \
#        --eval_steps 1 \
#        --learning_rate 1e-3 \
#        --use_swanlab \
#        --swanlab_api_key "$SWANLAB_API_KEY" \
#        --swanlab_project "Werewolf-LoRA"

# 212824 (learning_rate 5e-6 seems good)
# testset 10
# # lr_scheduler_type="linear",
# # warmup_ratio=0.1,
# # load_best_model_at_end=True  # 自动加载最佳模型
# python train_werewolf.py --stage sft \
#        --model_dir "Qwen/Qwen2.5-1.5B" \
#        --data_dir . \
#        --output_dir output \
#        --per_device_train_batch_size 1 \
#        --gradient_accumulation_steps 8 \
#        --max_seq_len 6000 \
#        --num_train_epochs 3 \
#        --eval_steps 1 \
#        --learning_rate 5e-6 \
#        --use_swanlab \
#        --swanlab_api_key "$SWANLAB_API_KEY" \
#        --swanlab_project "Werewolf-LoRA"

# 212828(挂了) or 212829(挂了)
# testset 100 不行
# lr_scheduler_type="linear",
# warmup_ratio=0.1,
# load_best_model_at_end=True  # 自动加载最佳模型

# 212832 (learning_rate 5e-7)

# 212833 (learning_rate 5e-8)

# speech_zh.csv → 行数: 3114，最长行长度: 5151 ---- 
# vote_zh.csv → 行数: 6401，最长行长度: 7482 ---- 
# action_zh.csv → 行数: 2622，最长行长度: 5144 ---- 
# train_zh.csv -> 行数: 12137，最长行长度: 7482 ---- 212989

# testset 
# eval_strategy="no",
        # lr_scheduler_type="cosine",
        # warmup_ratio=0.1,
# load_best_model_at_end=True  # 自动加载最佳模型
python train_werewolf.py --stage sft \
       --model_dir "Qwen/Qwen2.5-3B" \
       --data_dir . \
       --output_dir output \
       --per_device_train_batch_size 1 \
       --gradient_accumulation_steps 4 \
       --max_seq_len 7500 \
       --num_train_epochs 2 \
       --eval_steps 1 \
       --learning_rate 1e-4 \
       --use_swanlab \
       --swanlab_api_key "$SWANLAB_API_KEY" \
       --swanlab_project "Werewolf-LoRA"

#  (action_zh learning_rate 5e-6 seems good)
#  (vote_zh learning_rate 5e-6 seems good)

# 212929 1e-4
# testset 1
# eval_strategy="no",
# lr_scheduler_type="linear",
# warmup_ratio=0.1,
# load_best_model_at_end=True  # 自动加载最佳模型
# python train_werewolf.py --stage sft \
#        --model_dir "Qwen/Qwen2.5-3B" \
#        --data_dir . \
#        --output_dir output \
#        --per_device_train_batch_size 1 \
#        --gradient_accumulation_steps 8 \
#        --max_seq_len 6000 \
#        --num_train_epochs 3 \
#        --eval_steps 1 \
#        --learning_rate 1e-4 \
#        --use_swanlab \
#        --swanlab_api_key "$SWANLAB_API_KEY" \
#        --swanlab_project "Werewolf-LoRA"

echo "===== SFT 微调完成 $(date) =====" 