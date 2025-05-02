import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import random
import pandas as pd
from datasets import Dataset, DatasetDict
from tqdm import tqdm

from schema import GameExample

class GomokuDataProcessor:
    def __init__(self, data_dir: str, output_dir: str):
        """
        Initialize the Gomoku data processor.
        
        Args:
            data_dir: Directory containing raw game data
            output_dir: Directory to save processed data
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
    def load_raw_data(self, file_pattern: str = "*.json") -> List[Dict]:
        """Load raw game data from JSON files."""
        raw_data = []
        for file_path in tqdm(list(self.data_dir.glob(file_pattern))):
            with open(file_path, "r") as f:
                game_data = json.load(f)
                raw_data.append(game_data)
        return raw_data
    
    def convert_to_examples(self, raw_data: List[Dict]) -> List[GameExample]:
        """Convert raw data to GameExample objects."""
        examples = []
        for game in tqdm(raw_data):
            game_id = game.get("id", f"game_{len(examples)}")
            moves = game.get("moves", [])
            board_states = game.get("board_states", [])
            
            for turn_idx, (state, move) in enumerate(zip(board_states, moves)):
                # Format the board state as a string
                state_str = self._format_board_state(state)
                
                # Format the move as a string (e.g., "H8")
                move_str = self._format_move(move)
                
                # Get legal moves if available
                legal_moves = self._get_legal_moves(state)
                
                example = GameExample(
                    game_id=game_id,
                    turn_idx=turn_idx,
                    state=state_str,
                    legal_moves=legal_moves,
                    decision=move_str
                )
                examples.append(example.dict())
        
        return examples
    
    def _format_board_state(self, state) -> str:
        """
        Format the board state as a string.
        Override this method based on your specific data format.
        """
        # Example implementation - adjust to your actual data format
        if isinstance(state, str):
            return state
        
        # Assuming state is a 2D grid where:
        # 0 = empty, 1 = black, 2 = white
        if isinstance(state, list):
            board_size = len(state)
            rows = []
            for i in range(board_size):
                row = []
                for j in range(board_size):
                    if state[i][j] == 0:
                        row.append('.')
                    elif state[i][j] == 1:
                        row.append('X')
                    elif state[i][j] == 2:
                        row.append('O')
                rows.append(' '.join(row))
            return '\n'.join(rows)
        
        return str(state)
    
    def _format_move(self, move) -> str:
        """
        Format the move as a string.
        Override this method based on your specific data format.
        """
        # Example implementation - adjust to your actual data format
        if isinstance(move, str):
            return move
        
        # If move is represented as [row, col]
        if isinstance(move, list) and len(move) == 2:
            row, col = move
            # Convert to letter-number format (e.g., "H8")
            col_letter = chr(65 + col)
            return f"{col_letter}{row+1}"
        
        return str(move)
    
    def _get_legal_moves(self, state) -> Optional[str]:
        """
        Get legal moves based on the current board state.
        Override this method based on your specific game rules.
        """
        # This would implement game-specific logic to determine legal moves
        # For Gomoku, legal moves are typically empty spaces on the board
        return None  # For simplicity, we return None for now
    
    def split_data(self, examples: List[Dict], 
                  train_ratio: float = 0.8, 
                  val_ratio: float = 0.1) -> Dict[str, List[Dict]]:
        """Split data into train/val/test sets by game_id."""
        # Get unique game IDs
        game_ids = list(set(ex["game_id"] for ex in examples))
        random.shuffle(game_ids)
        
        # Calculate split sizes
        n_games = len(game_ids)
        n_train = int(n_games * train_ratio)
        n_val = int(n_games * val_ratio)
        
        # Split game IDs
        train_ids = set(game_ids[:n_train])
        val_ids = set(game_ids[n_train:n_train+n_val])
        test_ids = set(game_ids[n_train+n_val:])
        
        # Split examples by game ID
        splits = {
            "train": [ex for ex in examples if ex["game_id"] in train_ids],
            "validation": [ex for ex in examples if ex["game_id"] in val_ids],
            "test": [ex for ex in examples if ex["game_id"] in test_ids]
        }
        
        return splits
    
    def create_dataset(self, examples: List[Dict]) -> DatasetDict:
        """Create a HuggingFace dataset from examples."""
        splits = self.split_data(examples)
        
        dataset_dict = {
            split: Dataset.from_pandas(pd.DataFrame(data))
            for split, data in splits.items()
        }
        
        return DatasetDict(dataset_dict)
    
    def save_dataset(self, dataset: DatasetDict, name: str = "gomoku_dataset"):
        """Save the dataset to disk."""
        dataset.save_to_disk(self.output_dir / name)
        print(f"Dataset saved to {self.output_dir / name}")
    
    def process(self, file_pattern: str = "*.json", name: str = "gomoku_dataset"):
        """Process the data from raw files to HuggingFace dataset."""
        raw_data = self.load_raw_data(file_pattern)
        examples = self.convert_to_examples(raw_data)
        dataset = self.create_dataset(examples)
        self.save_dataset(dataset, name)
        return dataset

if __name__ == "__main__":
    # Example usage
    processor = GomokuDataProcessor(
        data_dir="data/raw", 
        output_dir="data/processed"
    )
    dataset = processor.process() 