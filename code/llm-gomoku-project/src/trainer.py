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
import subprocess

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
        
        # Verify CUDA availability
        self.check_gpu_availability()
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Add special tokens for game representation
        special_tokens = ["<board>", "</board>", "<move>", "</move>"]
        self.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        
        # Load base model with explicit GPU settings
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_8bit=False,
            torch_dtype=torch.float32,
            device_map="auto",
        )
        
        # Verify model is on correct device
        if torch.cuda.is_available():
            print(f"Model is on device: {next(self.model.parameters()).device}")
        
        # Resize token embeddings for the added special tokens
        self.model.resize_token_embeddings(len(self.tokenizer))
        
    def check_gpu_availability(self):
        """Check if GPU is available and print GPU info."""
        print("\n=== GPU Information ===")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"Current CUDA device: {torch.cuda.current_device()}")
            print(f"Number of GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"GPU {i} memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
            
            # Run nvidia-smi for detailed GPU info
            try:
                print("\nnvidia-smi output:")
                result = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, text=True)
                print(result.stdout)
            except Exception as e:
                print(f"Failed to run nvidia-smi: {e}")
        else:
            print("WARNING: CUDA is not available. Training will be on CPU only!")
            print("Check your CUDA installation and environment variables.")
            
        print("=== End GPU Information ===\n")
    
    def monitor_gpu_usage(self):
        """Monitor current GPU memory usage."""
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                print(f"GPU {i} memory allocated: {allocated:.2f} GB, reserved: {reserved:.2f} GB")
    
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
        batch_size: int = 16,
        gradient_accumulation_steps: int = 1,
        learning_rate: float = 2e-5, 
        warmup_steps: int = 100,
        max_grad_norm: float = 1.0,
        fp16: bool = True,
        logging_steps: int = 50,
        save_steps: int = 500,
    ):
        """Train the model."""
        # Force CUDA if available - MORE AGGRESSIVE APPROACH
        if torch.cuda.is_available():
            # Clear cache before training
            torch.cuda.empty_cache()
            
            device = torch.device("cuda")
            print(f"Using GPU device: {torch.cuda.get_device_name(device)}")
            
            # Force model to GPU and verify
            if next(self.model.parameters()).device.type != 'cuda':
                self.model = self.model.cuda()
                print(f"Model forcibly moved to {next(self.model.parameters()).device}")
            
            # Run a small test tensor through the model to verify GPU usage
            test_input = torch.randint(0, 100, (1, 10)).to(device)
            with torch.no_grad():
                _ = self.model(test_input)
            print(f"Test forward pass successfully run on {device}")
            
            # Print memory usage after test
            self.monitor_gpu_usage()
        else:
            print("WARNING: Training on CPU. This will be very slow!")
        
        # Training arguments - GPU OPTIMIZED
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
            fp16=fp16,
            fp16_full_eval=fp16,
            tf32=True,  # Enable TF32 for faster computation on Ampere+ GPUs
            max_grad_norm=max_grad_norm,
            report_to="none",
            run_name=output_name,
            optim="adamw_torch",
            remove_unused_columns=False,
            dataloader_num_workers=4,
            dataloader_pin_memory=True,
            bf16=False,  # Disable bf16 as it's not needed with fp16
            ddp_find_unused_parameters=False,  # Avoid unnecessary overhead
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
        
        # Monitor GPU before training
        print("\nGPU status before training:")
        self.monitor_gpu_usage()
        
        # Start training
        print("\nStarting training...")
        trainer.train()
        
        # Monitor GPU after training
        print("\nGPU status after training:")
        self.monitor_gpu_usage()
        
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
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--frozen_layers", type=int, default=12, help="Number of frozen layers")
    
    args = parser.parse_args()
    
    # Set CUDA environment variables inside Python as well
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    
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