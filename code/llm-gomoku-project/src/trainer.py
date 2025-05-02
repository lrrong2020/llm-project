import os
import argparse
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType
import wandb
from tqdm import tqdm

class GomokuTrainer:
    def __init__(
        self,
        model_name: str = "gpt2-medium",
        data_dir: str = "data/processed/gomoku_dataset",
        output_dir: str = "models",
        wandb_project: Optional[str] = "gomoku-gpt2",
    ):
        """
        Initialize the Gomoku GPT-2 trainer.
        
        Args:
            model_name: Base model to fine-tune
            data_dir: Directory containing processed dataset
            output_dir: Directory to save model checkpoints
            wandb_project: W&B project name (None to disable)
        """
        self.model_name = model_name
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.wandb_project = wandb_project
        
        # Initialize W&B if enabled
        if self.wandb_project:
            wandb.init(project=self.wandb_project)
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Add special tokens for game representation
        special_tokens = ["<board>", "</board>", "<move>", "</move>"]
        self.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        
        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_8bit=False,  # Set to True for quantization if needed
            torch_dtype=torch.float16,
            device_map="auto",
        )
        
        # Resize token embeddings for the added special tokens
        self.model.resize_token_embeddings(len(self.tokenizer))
        
    def load_dataset(self):
        """Load the processed Gomoku dataset."""
        self.dataset = load_from_disk(self.data_dir)
        print(f"Loaded dataset with {len(self.dataset['train'])} training examples")
        
    def preprocess_dataset(self, max_length: int = 512):
        """Preprocess the dataset for training."""
        def tokenize_function(examples):
            # Format each example as: "<board>{state}</board><move>{decision}</move>"
            texts = []
            for state, decision in zip(examples["state"], examples["decision"]):
                text = f"<board>{state}</board><move>{decision}</move>"
                texts.append(text)
            
            # Tokenize with padding to max_length
            tokenized = self.tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            
            # Set labels equal to input_ids for causal LM
            tokenized["labels"] = tokenized["input_ids"].clone()
            
            return tokenized
        
        # Apply tokenization to each split
        self.tokenized_dataset = self.dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=self.dataset["train"].column_names,
        )
        
        print(f"Preprocessed dataset: {self.tokenized_dataset}")
    
    def setup_peft(
        self,
        r: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        frozen_layers: int = 12,
    ):
        """Set up parameter-efficient fine-tuning (LoRA)."""
        # Configure LoRA
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["c_attn"],  # For GPT-2
            bias="none",
        )
        
        # Convert model to PEFT
        self.model = get_peft_model(self.model, peft_config)
        
        # Freeze bottom layers
        if frozen_layers > 0:
            modules = [self.model.base_model.model.h[i] for i in range(frozen_layers)]
            for module in modules:
                for param in module.parameters():
                    param.requires_grad = False
        
        # Enable gradient checkpointing
        self.model.gradient_checkpointing_enable()
        
        # Print trainable parameters
        self.model.print_trainable_parameters()
    
    def train(
        self,
        output_name: str = "gomoku-gpt2",
        epochs: int = 3,
        batch_size: int = 8,
        gradient_accumulation_steps: int = 4,
        learning_rate: float = 3e-5,
        warmup_steps: int = 100,
        max_grad_norm: float = 1.0,
        fp16: bool = True,
        logging_steps: int = 50,
        save_steps: int = 500,
    ):
        """Train the model."""
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir / output_name),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            weight_decay=0.01,
            logging_dir=str(self.output_dir / f"{output_name}_logs"),
            logging_steps=logging_steps,
            evaluation_strategy="steps",
            eval_steps=save_steps,
            save_strategy="steps",
            save_steps=save_steps,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=fp16,
            max_grad_norm=max_grad_norm,
            report_to="wandb" if self.wandb_project else "none",
            run_name=output_name if self.wandb_project else None,
        )
        
        # Data collator for language modeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Not using masked language modeling
        )
        
        # Initialize Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.tokenized_dataset["train"],
            eval_dataset=self.tokenized_dataset["validation"],
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        # Start training
        print("Starting training...")
        trainer.train()
        
        # Save the model
        trainer.save_model(str(self.output_dir / f"{output_name}_final"))
        self.tokenizer.save_pretrained(str(self.output_dir / f"{output_name}_final"))
        
        print(f"Training completed. Model saved to {self.output_dir / f'{output_name}_final'}")
        
        # End W&B run if enabled
        if self.wandb_project:
            wandb.finish()
    
def main():
    parser = argparse.ArgumentParser(description="Train GPT-2 for Gomoku")
    parser.add_argument("--model", type=str, default="gpt2-medium", help="Base model to fine-tune")
    parser.add_argument("--data_dir", type=str, default="data/processed/gomoku_dataset", help="Path to processed dataset")
    parser.add_argument("--output_dir", type=str, default="models", help="Directory to save model checkpoints")
    parser.add_argument("--wandb_project", type=str, default="gomoku-gpt2", help="W&B project name (None to disable)")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--frozen_layers", type=int, default=12, help="Number of frozen layers")
    
    args = parser.parse_args()
    
    trainer = GomokuTrainer(
        model_name=args.model,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        wandb_project=args.wandb_project,
    )
    
    trainer.load_dataset()
    trainer.preprocess_dataset()
    trainer.setup_peft(
        r=args.r,
        lora_alpha=args.lora_alpha,
        frozen_layers=args.frozen_layers,
    )
    trainer.train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )

if __name__ == "__main__":
    main() 