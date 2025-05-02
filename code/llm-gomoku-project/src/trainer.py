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
import wandb
from tqdm import tqdm

class GomokuTrainer:
    def __init__(
        self,
        model_name: str = "gpt2-medium",
        data_dir: str = "data/processed/gomoku_dataset",
        output_dir: str = "models",
        wandb_project: Optional[str] = None,
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
            try:
                wandb.init(project=self.wandb_project)
            except Exception as e:
                print(f"Warning: Failed to initialize wandb: {e}")
                print("Training will continue without wandb logging.")
                self.wandb_project = None
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Add special tokens for game representation
        special_tokens = ["<board>", "</board>", "<move>", "</move>"]
        self.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        
        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_8bit=False,
            torch_dtype=torch.float32,
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
        frozen_layers: int = 12,
    ):
        """Set up fine-tuning by freezing most layers and training only a few."""
        print("Using traditional fine-tuning on the final layers instead of LoRA...")
        
        # First, freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Unfreeze final transformer blocks
        if hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            # Calculate how many layers to unfreeze (total layers - frozen layers)
            num_layers = len(self.model.transformer.h)
            layers_to_unfreeze = max(0, num_layers - frozen_layers)
            
            # Unfreeze the final N transformer blocks
            for i in range(num_layers - layers_to_unfreeze, num_layers):
                for param in self.model.transformer.h[i].parameters():
                    param.requires_grad = True
            
            print(f"Unfroze the final {layers_to_unfreeze} transformer blocks out of {num_layers} total")
        else:
            print("Warning: Could not locate transformer blocks. Model architecture may not be compatible.")
        
        # Always unfreeze the LM head
        if hasattr(self.model, 'lm_head'):
            for param in self.model.lm_head.parameters():
                param.requires_grad = True
            print("Unfroze the language model head")
        
        # Disable gradient checkpointing since it's causing issues
        self.model.config.use_cache = True
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"trainable params: {trainable_params:,} || all params: {total_params:,} || trainable%: {100 * trainable_params / total_params:.4f}")
    
    def train(
        self,
        output_name: str = "gomoku-gpt2",
        epochs: int = 3,
        batch_size: int = 8,
        gradient_accumulation_steps: int = 4,
        learning_rate: float = 2e-5,  # Slightly higher learning rate for standard fine-tuning
        warmup_steps: int = 100,
        max_grad_norm: float = 1.0,
        fp16: bool = False,  # Disable FP16 training to avoid gradient scaling issues
        logging_steps: int = 50,
        save_steps: int = 500,
    ):
        """Train the model."""
        # Training arguments - simplified for standard fine-tuning
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
            save_steps=save_steps,
            save_total_limit=3,
            fp16=fp16,  # Now defaults to False
            max_grad_norm=max_grad_norm,
            report_to="none",  # Disable all reporting to avoid wandb issues
            run_name=output_name,
            optim="adamw_torch",  # Use PyTorch's AdamW instead of the transformers one
            remove_unused_columns=False,  # Keep all columns
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
    parser.add_argument("--wandb_project", type=str, default=None, help="W&B project name (None to disable)")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size (reduced to save memory)")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate (lowered for stability)")
    parser.add_argument("--frozen_layers", type=int, default=18, help="Number of frozen layers at the bottom (increased)")
    
    args = parser.parse_args()
    
    trainer = GomokuTrainer(
        model_name=args.model,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        wandb_project=args.wandb_project,
    )
    
    trainer.load_dataset()
    trainer.preprocess_dataset()
    trainer.setup_peft(frozen_layers=args.frozen_layers)
    trainer.train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )

if __name__ == "__main__":
    main() 