#!/bin/bash
#SBATCH --job-name=gomoku_eval_fixed
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --partition=normal
#SBATCH --account=msccsit2024
#SBATCH --gres=gpu:1
#SBATCH --output=gomoku_eval_fixed_%j.log

# Load modules
module purge  
module load cuda12.2/toolkit/12.2.2  
module load Anaconda3/2023.09-0

# Verify GPU availability
nvidia-smi

# Set up environment
source $(conda info --base)/etc/profile.d/conda.sh
conda create -n gomoku_env python=3.10 -y
conda activate gomoku_env

# Install compatible versions with CUDA support
pip install torch==2.1.0 torchvision==0.16.0 --extra-index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install datasets pydantic pandas tqdm wandb peft numpy packaging psutil pyyaml transformers pytz matplotlib

# Force GPU settings
export CUDA_VISIBLE_DEVICES=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TOKENIZERS_PARALLELISM=false

# Make sure the fixed file is being used
cat > verify_evaluate.py << 'EOL'
import re

def verify_and_fix():
    file_path = "src/evaluate.py"
    with open(file_path, 'r') as file:
        content = file.read()
    
    if "def evaluate(self, num_examples: Optional[int] = None, do_sample: bool = True, temperature: float = 0.7, top_p: float = 0.9):" in content:
        print("Evaluate method already fixed. Proceeding with evaluation.")
    else:
        print("Evaluate method needs to be fixed. Fixing now.")
        # Replace evaluate method
        pattern = r"def evaluate\(self, num_examples: Optional\[int\] = None\):(.*?)def main\(\):"
        replacement = """def evaluate(self, num_examples: Optional[int] = None, do_sample: bool = True, temperature: float = 0.7, top_p: float = 0.9):
        \"\"\"Run the full evaluation pipeline.\"\"\"
        # Load dataset
        self.load_dataset()
        
        # Preprocess examples
        examples = self.preprocess_examples(num_examples)
        
        # Generate predictions
        results = self.generate_predictions(examples, do_sample=do_sample, temperature=temperature, top_p=top_p)
        
        # Calculate metrics
        metrics = self.calculate_metrics(results)
        
        # Save results and metrics
        self.save_results(results, metrics)
        
        return metrics

def main():"""
        
        # Use re.DOTALL to make the dot match newlines
        modified_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
        
        # Also update the main function
        main_call_pattern = r"evaluator\.evaluate\(num_examples=args\.num_examples\)"
        main_call_replacement = "evaluator.evaluate(num_examples=args.num_examples, do_sample=args.do_sample, temperature=args.temperature, top_p=args.top_p)"
        
        modified_content = modified_content.replace(main_call_pattern, main_call_replacement)
        
        with open(file_path, 'w') as file:
            file.write(modified_content)
        
        print("Fixed evaluate method in src/evaluate.py")

if __name__ == "__main__":
    verify_and_fix()
EOL

# Run the verification script
python verify_evaluate.py

# Run evaluation with fixed parameters
python src/evaluate.py --model_path models/gomoku-gpt2_final --data_dir data/processed/gomoku_dataset --output_dir evaluation_results --num_examples 500 --do_sample=True --temperature=0.7 --top_p=0.9

# Create a simple demo script
cat > demo_gomoku.py << 'EOL'
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from matplotlib.colors import ListedColormap

def display_board(board_str):
    """Display the Gomoku board using matplotlib."""
    # Convert the board string to a 2D array
    board_lines = board_str.strip().split('\n')
    board_size = len(board_lines)
    board = np.zeros((board_size, board_size))
    
    for i, line in enumerate(board_lines):
        for j, char in enumerate(line):
            if char == 'X':
                board[i, j] = 1   # Black stone
            elif char == 'O':
                board[i, j] = 2   # White stone
    
    # Create a custom colormap for the board
    cmap = ListedColormap(['#D9B382', 'black', 'white'])
    
    # Set up the plot
    plt.figure(figsize=(10, 10))
    plt.imshow(board, cmap=cmap, vmin=0, vmax=2)
    
    # Add gridlines
    plt.grid(color='black', linestyle='-', linewidth=0.5)
    
    # Set ticks
    plt.xticks(np.arange(-.5, board_size, 1), [])
    plt.yticks(np.arange(-.5, board_size, 1), [])
    
    # Add coordinates
    for i in range(board_size):
        plt.text(-0.7, i, f"{i+1}", va='center')
        plt.text(i, -0.7, chr(65+i), ha='center')
    
    plt.title('Gomoku Board')
    plt.tight_layout()
    plt.savefig('gomoku_board.png')
    plt.close()

def predict_move(model, tokenizer, board_str):
    """Use the model to predict the next move."""
    prompt = f"<board>{board_str}</board><move>"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    output = model.generate(
        inputs["input_ids"],
        max_new_tokens=10,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
    )
    
    prediction_text = tokenizer.decode(output[0], skip_special_tokens=False)
    
    # Extract the move
    if "<move>" in prediction_text and "</move>" in prediction_text:
        move = prediction_text.split("<move>")[1].split("</move>")[0].strip()
    else:
        move = "Invalid move"
    
    return move

def make_move(board_str, move, player_symbol):
    """Apply a move to the board."""
    try:
        # Parse the move (e.g., 'H8' -> row=7, col=7)
        col = ord(move[0].upper()) - ord('A')
        row = int(move[1:]) - 1
        
        # Convert board to array
        board_lines = board_str.strip().split('\n')
        board_size = len(board_lines)
        
        # Check boundaries
        if row < 0 or row >= board_size or col < 0 or col >= board_size:
            return board_str, False
        
        # Check if position is empty
        if board_lines[row][col] != '.':
            return board_str, False
        
        # Make the move
        new_lines = []
        for i, line in enumerate(board_lines):
            if i == row:
                new_line = line[:col] + player_symbol + line[col+1:]
                new_lines.append(new_line)
            else:
                new_lines.append(line)
        
        return '\n'.join(new_lines), True
    except Exception as e:
        print(f"Error making move: {e}")
        return board_str, False

def run_demo(model_path, num_moves=5):
    """Run a Gomoku game demo."""
    # Load model and tokenizer
    print(f"Loading model from {model_path}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map="auto",
    )
    print("Model loaded successfully")
    
    # Initialize empty board (15x15)
    board_size = 15
    board = '\n'.join(['.' * board_size for _ in range(board_size)])
    
    print("\nStarting Gomoku Demo")
    print("-----------------")
    print("X: AI (Black)\nO: Human (White)")
    print(f"Board size: {board_size}x{board_size}")
    print("-----------------\n")
    
    print(board)
    display_board(board)
    print("\nInitial board saved as 'gomoku_board.png'")
    
    # Demo: AI plays against itself for num_moves
    current_player = 'X'  # Black goes first
    
    for move_num in range(num_moves):
        print(f"\nMove {move_num+1}. {current_player}'s turn:")
        
        # AI makes a move
        predicted_move = predict_move(model, tokenizer, board)
        print(f"AI predicts: {predicted_move}")
        
        # Apply the move
        new_board, success = make_move(board, predicted_move, current_player)
        if success:
            board = new_board
            print(f"Move {predicted_move} applied successfully")
        else:
            print(f"Invalid move: {predicted_move}. Generating a random valid move instead.")
            # Find an empty position
            board_lines = board.strip().split('\n')
            empty_positions = []
            for i in range(board_size):
                for j in range(board_size):
                    if board_lines[i][j] == '.':
                        empty_positions.append((i, j))
            
            if empty_positions:
                row, col = empty_positions[np.random.randint(0, len(empty_positions))]
                move = f"{chr(65+col)}{row+1}"
                board, _ = make_move(board, move, current_player)
                print(f"Random move applied: {move}")
        
        # Display board
        print("\nCurrent board:")
        print(board)
        display_board(board)
        print(f"Board after move {move_num+1} saved as 'gomoku_board.png'")
        
        # Switch player
        current_player = 'O' if current_player == 'X' else 'X'
    
    print("\nDemo complete!")
    print(f"Final board state after {num_moves} moves saved as 'gomoku_board.png'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo for Gomoku GPT-2 model")
    parser.add_argument("--model_path", type=str, default="models/gomoku-gpt2_final", help="Path to the fine-tuned model")
    parser.add_argument("--num_moves", type=int, default=10, help="Number of moves to simulate")
    
    args = parser.parse_args()
    run_demo(args.model_path, args.num_moves)
EOL

# Run the demo
echo -e "\n\n========== RUNNING DEMO ==========\n\n"
python demo_gomoku.py --model_path models/gomoku-gpt2_final --num_moves 10