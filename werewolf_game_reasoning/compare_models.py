#!/usr/bin/env python
"""compare_models.py
Compare performance between basic-only and SFT models on the same test samples.
"""
import argparse
import json
from pathlib import Path
import re

import torch
from datasets import load_dataset, Dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def parse_args():
    p = argparse.ArgumentParser(description="Compare Basic vs SFT models")
    p.add_argument("--basic_model_dir", default="output/basic", help="Basic model checkpoint")
    p.add_argument("--sft_model_dir", default="output/sft/checkpoint-1400", help="SFT model checkpoint")
    p.add_argument("--base_model", default="Qwen/Qwen2.5-1.5B", help="Base model path")
    p.add_argument("--data_dir", default=".", help="Data directory")
    p.add_argument("--num_examples", type=int, default=10, help="Number of examples to evaluate")
    p.add_argument("--max_new_tokens", type=int, default=128)
    return p.parse_args()

def format_sft(example):
    # From original train_werewolf.py
    full_prompt = (
        f"<Instruction>{example['instruction']}</Instruction>\n"
        f"<Prompt>{example['prompt']}</Prompt>\n"
    )
    # Exclude the response part for generation
    return {"text": full_prompt}

def main():
    args = parse_args()
    
    # Load dataset and prepare test samples
    raw_ds = load_dataset("csv", data_files=str(Path(args.data_dir) / "train_zh.csv"))["train"]
    formatted = raw_ds.map(
        format_sft,
        remove_columns=["instruction", "prompt", "response", "meta"],
    )
    # Use a fixed seed for reproducibility
    test_ds = formatted.shuffle(seed=42).select(range(args.num_examples))
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=False)
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Load both models with absolute paths
    basic_path = Path(args.basic_model_dir).absolute()
    sft_path = Path(args.sft_model_dir).absolute()
    
    print(f"Loading Basic model from: {basic_path}")
    basic_model = PeftModel.from_pretrained(
        base_model, 
        str(basic_path),
        is_trainable=False,
        local_files_only=True,
    )
    basic_model.eval()
    
    # Create a fresh base model instance for the SFT model to avoid issues
    sft_base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map="auto",
        trust_remote_code=True,
    )
    
    print(f"Loading SFT model from: {sft_path}")
    sft_model = PeftModel.from_pretrained(
        sft_base_model, 
        str(sft_path),
        is_trainable=False,
        local_files_only=True,
    )
    sft_model.eval()
    
    # Compare generations
    results = []
    
    for ex in tqdm(test_ds, desc="Generating comparisons"):
        prompt = ex["text"]
        
        # Prepare input
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(basic_model.device)
        
        # Generate with basic model
        with torch.no_grad():
            basic_out = basic_model.generate(
                input_ids=input_ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                eos_token_id=tokenizer.eos_token_id,
            )
            basic_text = tokenizer.decode(basic_out[0], skip_special_tokens=True)
            
            # Generate with SFT model
            sft_out = sft_model.generate(
                input_ids=input_ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                eos_token_id=tokenizer.eos_token_id,
            )
            sft_text = tokenizer.decode(sft_out[0], skip_special_tokens=True)
        
        # Extract just the generated part (after the prompt)
        basic_gen = basic_text[len(prompt):].strip()
        sft_gen = sft_text[len(prompt):].strip()
        
        # Try to extract JSON from both outputs for structured comparison
        basic_json = extract_json(basic_gen)
        sft_json = extract_json(sft_gen)
        
        results.append({
            "prompt": prompt,
            "basic_model": {
                "raw_output": basic_gen,
                "parsed_json": basic_json
            },
            "sft_model": {
                "raw_output": sft_gen,
                "parsed_json": sft_json
            }
        })
    
    # Save results
    with open("model_comparison.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"Saved comparison results to model_comparison.json")
    
    # Simple metrics: format correctness
    basic_format_correct = sum(1 for r in results if r["basic_model"]["parsed_json"] is not None)
    sft_format_correct = sum(1 for r in results if r["sft_model"]["parsed_json"] is not None)
    
    print(f"Basic model format correctness: {basic_format_correct}/{len(results)} ({basic_format_correct/len(results):.1%})")
    print(f"SFT model format correctness: {sft_format_correct}/{len(results)} ({sft_format_correct/len(results):.1%})")

def extract_json(text):
    """Try to extract JSON object from text"""
    try:
        # Try to find JSON object with regex
        match = re.search(r'{.*}', text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return None
    except json.JSONDecodeError:
        return None

if __name__ == "__main__":
    main()