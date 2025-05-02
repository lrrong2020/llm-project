import os
import argparse
from pathlib import Path
import json
from typing import Dict, List, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

class GomokuPredictor:
    def __init__(self, model_path: str):
        """
        Initialize the Gomoku predictor for inference.
        
        Args:
            model_path: Path to the fine-tuned model
        """
        self.model_path = Path(model_path)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        # Load model
        if os.path.exists(os.path.join(self.model_path, "adapter_config.json")):
            # This is a PEFT/LoRA model
            base_model_path = self._get_base_model_path()
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            self.model = PeftModel.from_pretrained(
                self.model,
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto",
            )
        else:
            # Standard model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto",
            )
        
        self.model.eval()
    
    def _get_base_model_path(self) -> str:
        """Get base model path from adapter config."""
        config_path = os.path.join(self.model_path, "adapter_config.json")
        with open(config_path, "r") as f:
            config = json.load(f)
        return config.get("base_model_name_or_path", "gpt2-medium")
    
    def predict(
        self,
        board_state: str,
        max_new_tokens: int = 10,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        num_return_sequences: int = 1,
    ) -> List[str]:
        """
        Generate a move prediction for the given board state.
        
        Args:
            board_state: String representation of the Gomoku board
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            num_return_sequences: Number of candidate sequences to return
            
        Returns:
            List of predicted moves in descending order of model confidence
        """
        # Format the prompt
        prompt = f"<board>{board_state}</board><move>"
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # Generate prediction
        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                num_return_sequences=num_return_sequences,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # Extract predictions
        predictions = []
        for output in outputs:
            text = self.tokenizer.decode(output, skip_special_tokens=False)
            # Extract the move part
            try:
                move = text.split("<move>")[1].split("</move>")[0].strip()
                predictions.append(move)
            except IndexError:
                predictions.append("[INVALID]")
        
        return predictions
    
    def format_board_from_array(self, board_array: List[List[int]]) -> str:
        """
        Format a board array into a string representation.
        
        Args:
            board_array: 2D array representing the board state
                         0: empty, 1: black, 2: white
                         
        Returns:
            String representation of the board
        """
        board_size = len(board_array)
        rows = []
        for i in range(board_size):
            row = []
            for j in range(board_size):
                if board_array[i][j] == 0:
                    row.append('.')
                elif board_array[i][j] == 1:
                    row.append('X')
                elif board_array[i][j] == 2:
                    row.append('O')
            rows.append(' '.join(row))
        return '\n'.join(rows)

def interactive_demo(model_path: str):
    """Run an interactive demo of the Gomoku predictor."""
    predictor = GomokuPredictor(model_path)
    
    print("=== Gomoku GPT-2 Interactive Demo ===")
    print("Enter a board state or 'q' to quit.")
    print("Format: Use '.' for empty, 'X' for black, 'O' for white, separated by spaces.")
    print("Example:")
    print(". . . . . . . . . . . . . . .")
    print(". . . . . . . . . . . . . . .")
    print(". . . . . . . . . . . . . . .")
    print(". . . . . . X . . . . . . . .")
    print(". . . . . . O . . . . . . . .")
    print(". . . . . . . . . . . . . . .")
    print(". . . . . . . . . . . . . . .")
    print()
    
    while True:
        print("\nEnter board state (empty line to end input, 'q' to quit):")
        board_lines = []
        while True:
            line = input()
            if line.lower() == 'q':
                return
            if not line:
                break
            board_lines.append(line)
        
        if not board_lines:
            continue
        
        board_state = '\n'.join(board_lines)
        
        # Generate prediction
        predictions = predictor.predict(
            board_state,
            num_return_sequences=3,
            temperature=0.8,
        )
        
        # Display results
        print("\nPredicted moves:")
        for i, move in enumerate(predictions):
            print(f"{i+1}. {move}")

def main():
    parser = argparse.ArgumentParser(description="Gomoku GPT-2 inference")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the fine-tuned model")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_demo(args.model_path)
    else:
        # Simple demo with a hard-coded example
        predictor = GomokuPredictor(args.model_path)
        
        # Example board state (15x15 board)
        board_state = "\n".join([
            ". . . . . . . . . . . . . . .",
            ". . . . . . . . . . . . . . .",
            ". . . . . . . . . . . . . . .",
            ". . . . . . . . . . . . . . .",
            ". . . . . . . . . . . . . . .",
            ". . . . . X . . . . . . . . .",
            ". . . . . O . . . . . . . . .",
            ". . . . . . . . . . . . . . .",
            ". . . . . . . . . . . . . . .",
            ". . . . . . . . . . . . . . .",
            ". . . . . . . . . . . . . . .",
            ". . . . . . . . . . . . . . .",
            ". . . . . . . . . . . . . . .",
            ". . . . . . . . . . . . . . .",
            ". . . . . . . . . . . . . . ."
        ])
        
        print("Board state:")
        print(board_state)
        
        predictions = predictor.predict(board_state, num_return_sequences=3)
        
        print("\nPredicted moves:")
        for i, move in enumerate(predictions):
            print(f"{i+1}. {move}")

if __name__ == "__main__":
    main() 