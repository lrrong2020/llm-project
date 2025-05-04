# 狼人杀模型训练

本项目基于LoRA技术微调Qwen2.5-1.5B模型，用于狼人杀游戏中的策略理解和决策。训练分为两个阶段：

1. 基础知识微调 (game_strategy_and_term.csv)
2. SFT微调 (train_zh.csv)

## 准备工作

### 1. 环境设置

在计算集群登录节点运行：

```bash
# 克隆到当前目录（如已有代码则跳过）
git clone <repository-url>
cd werewolf_game_reasoning

# 创建并设置环境（仅首次运行）
bash setup.sh
```

或手动设置：

```bash
module load cuda12.2/toolkit/12.2.2
module load Anaconda3/2023.09-0

# 创建环境
conda create -n werewolf python=3.10 -y
conda activate werewolf

# 安装依赖
pip install -r ../requirements.txt
```

### 2. SwanLab 设置（可选）

SwanLab用于可视化训练过程。有三种使用方式：

**A. 使用API密钥（推荐）：**
1. 在 [SwanLab官网](https://swanlab.cn) 注册账号
2. 在个人设置页面获取API密钥
3. 修改 run_basic.sbatch 文件中的 `SWANLAB_API_KEY="your-api-key-here"`

**B. 使用本地模式（无需账号）：**
```bash
# 使用本地模式运行
bash run_local.sh
```

**C. 交互式登录（不适用于Slurm批处理作业）：**
```bash
# 在登录节点或交互式会话中
conda activate werewolf
swanlab login
# 粘贴API密钥并回车
```

### 3. 训练

**基础知识阶段：**

```bash
# 提交基础知识微调作业
sbatch run_basic.sbatch
```

**SFT阶段 (多个选项)：**

根据上一次运行情况，我们有几个不同的SFT训练选项，可以解决不同环境下的问题：

```bash
# 选项1：标准SFT (如果基础训练成功且SFT训练失败)
sbatch run_sft_v2.sbatch  # 禁用FP16，使用更小批次和样本数

# 选项2：CPU卸载版本 (如果GPU显存不足)
sbatch run_sft_cpu_offload.sbatch  # 使用CPU卸载技术减轻GPU负担

# 选项3：完整训练流程 (一次性运行基础+SFT)
sbatch run_werewolf.sbatch
```

**交互式调试：**

```bash
# 交互式运行（适合调试）
bash run_interactive.sh
```

## SFT训练参数说明

### 标准版本 (run_sft_v2.sbatch)
- 禁用了FP16混合精度训练，避免优化器错误
- 批次大小减少到1，梯度累积增加到8
- 限制训练样本数为2000（可扩大）
- 序列长度限制为2048（可安全扩展到4096）
- 支持使用BF16代替FP16（在A100/H100上更稳定）
- 降低学习率为1e-4

### CPU卸载版本 (run_sft_cpu_offload.sbatch)
- 参数卸载到CPU，大幅减少GPU显存需求
- 使用DeepSpeed ZeRO Stage 2优化
- 梯度累积增加到16，批次保持为1
- 需要更多CPU内存(120GB)和核心数(12)
- 学习率进一步降低为5e-5
- 使用accelerate启动器而非直接执行

## 参数说明

主要训练参数：

- `--stage`: 训练阶段，可选 basic/sft/full
- `--model_dir`: 模型路径，默认 "Qwen/Qwen2.5-1.5B"
- `--per_device_train_batch_size`: 每设备批次大小
- `--gradient_accumulation_steps`: 梯度累积步数
- `--max_seq_len`: 最大序列长度
- `--use_swanlab`: 开启SwanLab记录
- `--swanlab_api_key`: SwanLab API密钥（可直接在命令行提供）
- `--use_local_swanlab`: 使用SwanLab本地模式记录
- `--max_train_samples`: 限制训练样本数量
- `--use_bf16`: 使用BF16精度代替FP16

## 常见问题

1. **SFT训练失败 (No inf checks were recorded for this optimizer)**：
   - 问题原因：FP16混合精度训练中梯度缩放器错误
   - 解决方法：使用run_sft_v2.sbatch脚本禁用FP16或使用BF16
   - 备选方案：减少序列长度，限制训练样本数量

2. **显存不足**：
   - 使用run_sft_cpu_offload.sbatch脚本启用CPU卸载
   - 或降低序列长度(max_seq_len)和批次大小

3. **SwanLab登录问题**：
   - 批处理作业中无法交互式登录，请使用 `--swanlab_api_key` 参数传入API密钥
   - 或使用 `--use_local_swanlab` 参数启用本地模式
   - 如果无需可视化训练过程，可移除 `--use_swanlab` 参数

4. **找不到模型**：
   - 若网络受限，可预先下载模型放到指定位置，通过 `--model_dir` 指定

5. **CUDA错误**：
   - 检查是否加载了正确的CUDA模块
   - 验证PyTorch是否可以检测到GPU：`python -c "import torch; print(torch.cuda.is_available())"` 