#!/bin/bash
# 交互式运行训练（适合调试和短时间试验）

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

# 请求交互式资源
echo "正在申请交互式GPU资源，请稍等..."
srun --gres=gpu:1 --time=4:00:00 --account=msccsit2024 --partition=normal --pty bash -c '
    echo "获取GPU成功！"
    nvidia-smi
    
    # 环境检查
    python -c "import torch; print(f\"PyTorch: {torch.__version__}, GPU可用: {torch.cuda.is_available()}, GPU型号: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"无\"}\")"
    
    # 显示帮助
    echo -e "\n可用命令:"
    echo "1. python train_werewolf.py --stage basic --use_swanlab  # 基础微调"
    echo "2. python train_werewolf.py --stage sft --use_swanlab    # SFT微调"
    echo "3. python train_werewolf.py --help                       # 查看所有参数"
    echo -e "\n输入exit退出会话\n"
    
    # 启动交互shell
    bash
' 