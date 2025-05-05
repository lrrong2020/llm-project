#!/usr/bin/env python
"""train_werewolf.py
A clean, runnable training entry that supports two-stage LoRA fine-tuning of
Qwen2.5-1.5B on the Werewolf (狼人杀) datasets.

Stages
------
1) basic : domain terms / strategy QA (game_strategy_and_term.csv)
2) sft   : real game behaviour (train_zh.csv)
3) full  : run basic then sft in one go

This script intentionally avoids heavyweight notebook artefacts and runs on
single-GPU or multi-GPU (torchrun) environments. It also integrates SwanLab
logging (only in the main process).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Dict

import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
import swanlab
from swanlab.integration.transformers import SwanLabCallback

# -------------------------------------------------------
# Utility functions
# -------------------------------------------------------

def get_bnb_4bit_config() -> BitsAndBytesConfig:
    """Return 4-bit nf4 quantisation config that works for A100 / H800."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def build_lora_cfg() -> LoraConfig:
    """LoRA hyper-parameters (shared across stages)."""
    return LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=[
            "c_attn",  # Qwen uses c_attn instead of q_proj/k_proj/v_proj
            "mlp.down_proj",
            "mlp.up_proj",
        ],
        lora_dropout=0.2,
        bias="none",
        task_type="CAUSAL_LM",
    )


# -------------------------------------------------------
# Dataset helpers
# -------------------------------------------------------

def format_basic(example: Dict[str, str]) -> Dict[str, str]:
    """<Prompt>prompt</Prompt> <Response>resp</Response>"""
    return {
        "text": f"<Prompt>{example['prompt']}</Prompt>\n<Response>{example['response']}</Response>"
    }


def format_sft(example: Dict[str, str]) -> Dict[str, str]:
    full_prompt = (
        f"<Instruction>{example['instruction']}</Instruction>\n"
        f"<Prompt>{example['prompt']}</Prompt>\n"
        f"<Response>{example['response']}</Response>"
    )
    return {"text": full_prompt}


def build_tokenize_fn(tokenizer, max_len: int, truncate_only_first: bool = False):
    """Return a callable for `datasets.map` that tokenises & creates labels."""

    def _fn(examples):
        toks = tokenizer(
            examples["text"],
            padding="max_length",
            truncation="only_first" if truncate_only_first else True,
            max_length=max_len,
            return_tensors="pt",
        )
        toks["labels"] = toks["input_ids"].clone()
        return toks

    return _fn


# -------------------------------------------------------
# Training wrappers
# -------------------------------------------------------

def train_basic(
    model_name: str,
    output_dir: Path,
    data_dir: Path,
    args: argparse.Namespace,
):
    dataset = load_dataset(
        "csv", data_files=str(data_dir / "game_strategy_and_term.csv")
    )["train"].map(format_basic, remove_columns=["prompt", "response"])
    data_split = dataset.train_test_split(test_size=0.1, seed=42)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    token_fn = build_tokenize_fn(tokenizer, args.max_seq_len)
    tokenised = data_split.map(token_fn, batched=True, remove_columns=["text"])

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=get_bnb_4bit_config(),
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, build_lora_cfg())

    train_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        fp16=True,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        logging_steps=1,
        save_strategy="steps",
        save_steps=args.save_steps,
        report_to="none",  # SwanLab handled via callback
    )

    # 设置回调
    callbacks = []
    if args.use_swanlab:
        if args.swanlab_api_key:
            # 直接通过代码登录SwanLab
            try:
                print(f"正在使用API密钥登录SwanLab...")
                swanlab.login(api_key=args.swanlab_api_key)
                callbacks.append(SwanLabCallback(
                    project=args.swanlab_project or "Werewolf-LoRA",
                    experiment_name=args.stage
                ))
                print("SwanLab登录成功!")
            except Exception as e:
                print(f"SwanLab登录失败: {e}")
                print("继续训练但不使用SwanLab记录")
        elif args.use_local_swanlab:
            # 使用本地模式
            print("使用SwanLab本地模式记录")
            callbacks.append(SwanLabCallback(
                mode="local",
                project=args.swanlab_project or "Werewolf-LoRA",
                experiment_name=args.stage
            ))
        else:
            print("警告: 启用了SwanLab但未提供API密钥。尝试使用已缓存的凭据或交互登录...")
            try:
                callbacks.append(SwanLabCallback(
                    project=args.swanlab_project or "Werewolf-LoRA",
                    experiment_name=args.stage
                ))
            except Exception as e:
                print(f"SwanLab初始化失败: {e}")
                print("继续训练但不使用SwanLab记录")

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=tokenised["train"],
        eval_dataset=tokenised["test"],
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        callbacks=callbacks,
    )
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def train_sft(
    model_name: str,
    basic_lora_dir: Path,
    output_dir: Path,
    data_dir: Path,
    args: argparse.Namespace,
):
    """Second-stage SFT; loads LoRA from `basic_lora_dir`."""

    print(f"加载SFT数据集: {data_dir / 'train_zh.csv'}")
    dataset = load_dataset("csv", data_files=str(data_dir / "train_zh.csv"))["train"]
    
    # 可选: 只使用部分数据进行训练
    if args.max_train_samples > 0:
        print(f"限制训练样本数量为 {args.max_train_samples} (原始数量: {len(dataset)})")
        dataset = dataset.select(range(min(args.max_train_samples, len(dataset))))
    
    print(f"开始处理SFT数据: {len(dataset)} 样本")
    dataset = dataset.map(
        format_sft,
        remove_columns=["instruction", "prompt", "response", "meta"],
    )
    data_split = dataset.train_test_split(test_size=0.1, seed=42)

    print(f"加载tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    
    print(f"使用最大序列长度: {args.max_seq_len}")
    token_fn = build_tokenize_fn(tokenizer, args.max_seq_len, truncate_only_first=True)
    
    print("开始tokenize数据...")
    tokenised = data_split.map(token_fn, batched=True, remove_columns=["text"])
    print(f"Tokenize完成，训练集大小: {len(tokenised['train'])} 样本")

    print(f"加载基础模型: {model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=get_bnb_4bit_config(),
        device_map="auto",
        trust_remote_code=True,
    )
    base_model = prepare_model_for_kbit_training(base_model)
    
    print(f"加载LoRA适配器: {basic_lora_dir}")
    model = get_peft_model(base_model, LoraConfig.from_pretrained(str(basic_lora_dir)))
    model.print_trainable_parameters()

    # 优化内存使用并禁用fp16混合精度训练
    print("配置训练参数 - 禁用FP16混合精度")
    train_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        # 禁用FP16混合精度训练，改用BF16或纯FP32
        fp16=False,  # 禁用FP16
        bf16=args.use_bf16,  # 可选使用BF16，在A100/H100上更稳定
        # 内存优化
        gradient_checkpointing=True,
        optim="adamw_torch",  # 使用PyTorch原生优化器
        # 其他训练参数
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        logging_steps=1,
        save_strategy="steps",
        save_steps=args.save_steps,
        # 不使用内置的报告
        report_to="none",
        # 额外内存优化
        ddp_find_unused_parameters=False,
        dataloader_pin_memory=False,
    )

    # 设置回调
    callbacks = []
    if args.use_swanlab:
        if args.swanlab_api_key:
            # 直接通过代码登录SwanLab
            try:
                print(f"正在使用API密钥登录SwanLab...")
                swanlab.login(api_key=args.swanlab_api_key)
                callbacks.append(SwanLabCallback(
                    project=args.swanlab_project or "Werewolf-LoRA",
                    experiment_name=args.stage
                ))
                print("SwanLab登录成功!")
            except Exception as e:
                print(f"SwanLab登录失败: {e}")
                print("继续训练但不使用SwanLab记录")
        elif args.use_local_swanlab:
            # 使用本地模式
            print("使用SwanLab本地模式记录")
            callbacks.append(SwanLabCallback(
                mode="local",
                project=args.swanlab_project or "Werewolf-LoRA",
                experiment_name=args.stage
            ))
        else:
            print("警告: 启用了SwanLab但未提供API密钥。尝试使用已缓存的凭据或交互登录...")
            try:
                callbacks.append(SwanLabCallback(
                    project=args.swanlab_project or "Werewolf-LoRA",
                    experiment_name=args.stage
                ))
            except Exception as e:
                print(f"SwanLab初始化失败: {e}")
                print("继续训练但不使用SwanLab记录")

    print("开始SFT训练...")
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=tokenised["train"],
        eval_dataset=tokenised["test"],
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        callbacks=callbacks,
    )
    trainer.train()
    
    print(f"训练完成，保存模型到 {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


# -------------------------------------------------------
# Main
# -------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Werewolf LoRA fine-tune")
    p.add_argument("--stage", choices=["basic", "sft", "full"], default="basic")
    p.add_argument("--model_dir", default="Qwen/Qwen2.5-1.5B")
    p.add_argument("--data_dir", default=".")
    p.add_argument("--output_dir", default="output", help="root output folder")
    p.add_argument("--per_device_train_batch_size", type=int, default=4)
    p.add_argument("--per_device_eval_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--num_train_epochs", type=int, default=3)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--max_seq_len", type=int, default=2048)
    p.add_argument("--eval_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=100)
    
    # SwanLab相关参数
    p.add_argument("--use_swanlab", action="store_true", help="开启SwanLab记录")
    p.add_argument("--swanlab_api_key", type=str, default="", help="SwanLab API密钥")
    p.add_argument("--swanlab_project", type=str, default="", help="SwanLab项目名称")
    p.add_argument("--use_local_swanlab", action="store_true", help="使用SwanLab本地模式")
    
    # 新增参数
    p.add_argument("--max_train_samples", type=int, default=-1, help="限制训练样本数量，-1表示使用全部")
    p.add_argument("--use_bf16", action="store_true", help="使用BF16代替FP16，在A100或H100上更好")
    
    return p.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    root_out = Path(args.output_dir)
    root_out.mkdir(parents=True, exist_ok=True)

    # 打印训练信息
    print(f"开始训练 - 阶段: {args.stage}")
    print(f"数据目录: {data_dir}")
    print(f"输出目录: {root_out}")
    print(f"模型路径: {args.model_dir}")
    print(f"批次大小: {args.per_device_train_batch_size}，梯度累积: {args.gradient_accumulation_steps}")
    print(f"评估批次大小: {args.per_device_eval_batch_size}")
    if args.use_swanlab:
        if args.swanlab_api_key:
            print("SwanLab: 使用提供的API密钥")
        elif args.use_local_swanlab:
            print("SwanLab: 使用本地模式")
        else:
            print("SwanLab: 尝试使用已缓存凭据")

    if args.stage in {"basic", "full"}:
        basic_dir = root_out / "basic"
        basic_dir.mkdir(exist_ok=True, parents=True)
        train_basic(args.model_dir, basic_dir, data_dir, args)
    else:
        basic_dir = root_out / "basic"
        if not basic_dir.exists():
            raise FileNotFoundError("Basic LoRA dir not found – run stage 'basic' first")

    if args.stage in {"sft", "full"}:
        sft_dir = root_out / "sft"
        sft_dir.mkdir(exist_ok=True, parents=True)
        train_sft(args.model_dir, basic_dir, sft_dir, data_dir, args)


if __name__ == "__main__":
    main() 