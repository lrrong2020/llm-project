#!/usr/bin/env python
"""
Full pipeline script to train and evaluate a GPT-2 model for Gomoku.
"""
import os
import argparse
import subprocess
from pathlib import Path
import time

def run_command(command, description=None):
    """Run a shell command and print output."""
    if description:
        print(f"\n{'='*50}")
        print(f"  {description}")
        print(f"{'='*50}")
    
    print(f"Running: {command}")
    result = subprocess.run(command, shell=True, check=True)
    return result.returncode == 0

def main():
    parser = argparse.ArgumentParser(description="Run the complete Gomoku GPT-2 training pipeline")
    parser.add_argument("--generate_data", action="store_true", help="Generate synthetic Gomoku data")
    parser.add_argument("--num_games", type=int, default=1000, help="Number of games to generate")
    parser.add_argument("--process_data", action="store_true", help="Process the raw data")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the model")
    parser.add_argument("--demo", action="store_true", help="Run the demo")
    parser.add_argument("--all", action="store_true", help="Run the complete pipeline")
    parser.add_argument("--model", type=str, default="gpt2-medium", help="Base model to fine-tune")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--output_dir", type=str, default="models", help="Directory to save models")
    parser.add_argument("--model_path", type=str, help="Path to a saved model for evaluation/demo (overrides default path)")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--frozen_layers", type=int, default=18, help="Number of frozen layers")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    args = parser.parse_args()
    
    # Check if at least one action is specified
    if not any([args.generate_data, args.process_data, args.train, args.evaluate, args.demo, args.all]):
        parser.error("Please specify at least one action to perform")
    
    # Set paths
    model_name = "gomoku-gpt2"
    default_model_path = Path(args.output_dir) / f"{model_name}_final"
    # Use the provided model_path if available, otherwise use the default
    model_path = args.model_path if args.model_path else default_model_path
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Run the complete pipeline or individual steps
    if args.all or args.generate_data:
        run_command(
            f"python src/data_generation.py --num_games {args.num_games} --output_dir data/raw",
            "Generating synthetic Gomoku data"
        )
    
    if args.all or args.process_data:
        run_command(
            "python src/data_processor.py",
            "Processing raw data into training format"
        )
    
    if args.all or args.train:
        run_command(
            f"python src/trainer.py --model {args.model} --epochs {args.epochs} --output_dir {args.output_dir} --batch_size {args.batch_size} --frozen_layers {args.frozen_layers} --lr {args.lr}",
            "Training the model"
        )
    
    if args.all or args.evaluate:
        run_command(
            f"python src/evaluate.py --model_path {model_path} --data_dir data/processed/gomoku_dataset --output_dir evaluation_results",
            "Evaluating the model"
        )
    
    if args.all or args.demo:
        run_command(
            f"python src/inference.py --model_path {model_path} --interactive",
            "Running interactive demo"
        )
    
    print("\nPipeline completed successfully!")

if __name__ == "__main__":
    main() 