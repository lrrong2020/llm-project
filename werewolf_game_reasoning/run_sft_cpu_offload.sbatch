#!/bin/bash
#SBATCH --job-name=werewolf_sft
#SBATCH --partition=h800   # 使用H800大内存GPU
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12  # 使用更多CPU核心用于卸载
#SBATCH --mem=120G         # 更大内存用于CPU卸载
#SBATCH --time=48:00:00    # 更长时间
#SBATCH --account=msccsit2024
#SBATCH --output=logs/werewolf_sft_cpu_%j.out
#SBATCH --error=logs/werewolf_sft_cpu_%j.err

# 确保日志和输出目录存在
mkdir -p logs output/sft

# 加载模块
module purge
module load slurm
module load cuda12.2/toolkit/12.2.2
module load Anaconda3/2023.09-0

# 验证GPU可用和内存
nvidia-smi
free -h

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

# 运行CPU卸载版SFT微调
echo "===== 开始 SFT 微调 (CPU卸载版) $(date) ====="

# 使用accelerate启动器运行训练
# 首先生成配置文件
cat > accelerate_config.yaml << EOF
compute_environment: LOCAL_MACHINE
deepspeed_config: {}
distributed_type: NO
downcast_bf16: 'no'
machine_rank: 0
main_process_ip: null
main_process_port: null
main_training_function: main
mixed_precision: 'no'
num_machines: 1
num_processes: 1
use_cpu: false
gpu_ids: "0"
same_network: true
dynamo_backend: "no"
rdzv_backend: static
use_deepspeed: false
deepspeed_multinode_launcher: standard
optimizer: "auto"
zero_stage: 2
offload_optimizer_device: "cpu"
offload_param_device: "cpu"
offload_optimizer_state: true
zero3_init_flag: false
zero3_save_16bit_model: false
EOF

# 使用accelerate启动训练
accelerate launch --config_file accelerate_config.yaml train_werewolf.py \
  --stage sft \
  --model_dir "Qwen/Qwen2.5-1.5B" \
  --data_dir . \
  --output_dir output \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --max_seq_len 2048 \
  --learning_rate 5e-5 \
  --max_train_samples 2000 \
  --use_swanlab \
  --swanlab_api_key "$SWANLAB_API_KEY" \
  --swanlab_project "Werewolf-LoRA"

echo "===== SFT 微调完成 $(date) =====" 