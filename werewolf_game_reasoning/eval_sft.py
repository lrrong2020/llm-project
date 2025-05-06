#!/usr/bin/env python
"""eval_sft.py
Comprehensive evaluation utilities for the second-stage (SFT) LoRA adapter
trained with *train_werewolf.py*.

It provides:
  1. Per-token negative-log-likelihood (NLL) and perplexity on a 10 % test split
     of *train_zh.csv*.
  2. Qualitative generation examples that are saved to *eval_outputs.jsonl* for
     manual review.

Example
-------
python eval_sft.py \
       --model_dir output/sft \
       --base_model Qwen/Qwen2.5-1.5B \
       --data_dir .
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict

import torch
from datasets import load_dataset, Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from peft import PeftModel

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def tokenize_function(tokenizer, max_length: int):
    """Return a mapping function that tokenises raw *text* strings."""

    def _fn(examples):
        toks = tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        toks["labels"] = toks["input_ids"].clone()
        return toks

    return _fn


def calc_perplexity(model, dataset: Dataset, batch_size: int = 4) -> float:
    """Compute perplexity (exp(mean NLL)) over *dataset*.

    The dataset must already contain *input_ids* and *labels* tensors.
    """
    model.eval()
    nlls: List[float] = []
    with torch.no_grad():
        for i in tqdm(range(0, len(dataset), batch_size)):
            batch = dataset[i : i + batch_size]
            input_ids = torch.tensor(batch["input_ids"], device=model.device)
            labels = torch.tensor(batch["labels"], device=model.device)
            outputs = model(input_ids=input_ids, labels=labels)
            # outputs.loss is already mean NLL over tokens
            nlls.append(outputs.loss.item())
    mean_nll = sum(nlls) / len(nlls)
    return float(torch.exp(torch.tensor(mean_nll)).item())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate Werewolf SFT LoRA model")
    p.add_argument("--model_dir", default="output/sft", help="Folder that contains the LoRA adapter weights from SFT stage")
    p.add_argument("--base_model", default="Qwen/Qwen2.5-1.5B", help="The unchanged backbone that adapters were trained on")
    p.add_argument("--data_dir", default=".", help="Path that contains train_zh.csv")
    p.add_argument("--max_seq_len", type=int, default=2048)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_examples", type=int, default=20, help="How many qualitative generations to dump")
    return p.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)

    # ---------------------------------------------------------------------
    # Dataset preparation (identical split logic as train_werewolf.py)
    # ---------------------------------------------------------------------
    raw_ds = load_dataset("csv", data_files=str(data_dir / "train_zh.csv"))["train"]
    from train_werewolf import format_sft  # reuse formatting logic

    formatted = raw_ds.map(
        format_sft,
        remove_columns=["instruction", "prompt", "response", "meta"],
    )
    data_split = formatted.train_test_split(test_size=0.1, seed=42)
    test_ds: Dataset = data_split["test"]

    # ---------------------------------------------------------------------
    # Model & tokenizer
    # ---------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=False)

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map="auto",
        trust_remote_code=True,
    )

    # 确保使用绝对路径加载本地模型，而不是从HF Hub加载
    model_path = Path(args.model_dir).absolute()
    print(f"加载本地LoRA适配器: {model_path}")
    model = PeftModel.from_pretrained(
        base_model, 
        str(model_path),
        is_trainable=False,
        local_files_only=True,  # 强制只使用本地文件
    )
    model.eval()

    # ---------------------------------------------------------------------
    # Perplexity
    # ---------------------------------------------------------------------
    tokenised_test = test_ds.map(
        tokenize_function(tokenizer, args.max_seq_len),
        batched=True,
        remove_columns=["text"],
    )
    ppl = calc_perplexity(model, tokenised_test, batch_size=args.batch_size)
    print(f"Perplexity on 10% test split: {ppl:.2f}")

    # ---------------------------------------------------------------------
    # Qualitative generations
    # ---------------------------------------------------------------------
    examples = test_ds.shuffle(seed=0).select(range(min(args.num_examples, len(test_ds))))
    outputs: List[Dict[str, str]] = []
    for ex in tqdm(examples, desc="Generating"):
        # 只保留 Instruction 和 Prompt 部分，去掉 Response
        prompt_text = ex["text"].split("<Response>")[0]
        input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(model.device)
        gen_out = model.generate(
            input_ids=input_ids,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
        )
        gen_out = model.generate(...)
        decoded = tokenizer.decode(gen_out[0], skip_special_tokens=True)
        outputs.append({"prompt": prompt_text, "generated": decoded})

    with Path("eval_outputs.jsonl").open("w", encoding="utf-8") as f:
        for item in outputs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("Saved qualitative generations to eval_outputs.jsonl")


if __name__ == "__main__":
    main() 