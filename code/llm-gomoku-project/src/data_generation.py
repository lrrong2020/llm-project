import random
import json
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm
import argparse

class GomokuGame:
    """Simple Gomoku game implementation for data generation."""
    
    def __init__(self, board_size=15):
        """Initialize a new Gomoku game.
        
        Args:
            board_size: Size of the board (default: 15x15)
        """
        self.board_size = board_size
        self.reset()
    
    def reset(self):
        """Reset the game to initial state."""
        # 0: empty, 1: black, 2: white
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        self.current_player = 1  # Black goes first
        self.moves = []
        self.board_states = []
        self.game_over = False
        self.winner = None
        
        # Store initial board state
        self.board_states.append(self.board.copy())
    
    def make_move(self, row, col):
        """Make a move on the board.
        
        Args:
            row: Row index (0-based)
            col: Column index (0-based)
            
        Returns:
            True if the move is valid, False otherwise
        """
        if self.game_over:
            return False
        
        if not (0 <= row < self.board_size and 0 <= col < self.board_size):
            return False
        
        if self.board[row, col] != 0:
            return False
        
        # Place the stone
        self.board[row, col] = self.current_player
        
        # Record the move
        self.moves.append([row, col])
        
        # Check for win
        if self._check_win(row, col):
            self.game_over = True
            self.winner = self.current_player
        
        # Switch player
        self.current_player = 3 - self.current_player  # 1->2, 2->1
        
        # Store board state after move
        self.board_states.append(self.board.copy())
        
        return True
    
    def _check_win(self, row, col):
        """Check if the last move at (row, col) resulted in a win."""
        player = self.board[row, col]
        
        # Define 4 directions: horizontal, vertical, diagonal, anti-diagonal
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        for dr, dc in directions:
            count = 1  # Count the current stone
            
            # Check in one direction
            r, c = row + dr, col + dc
            while 0 <= r < self.board_size and 0 <= c < self.board_size and self.board[r, c] == player:
                count += 1
                r += dr
                c += dc
            
            # Check in the opposite direction
            r, c = row - dr, col - dc
            while 0 <= r < self.board_size and 0 <= c < self.board_size and self.board[r, c] == player:
                count += 1
                r -= dr
                c -= dc
            
            # Check if we have 5 in a row
            if count >= 5:
                return True
        
        return False
    
    def get_legal_moves(self):
        """Get all legal moves on the current board."""
        if self.game_over:
            return []
        
        legal_moves = []
        for r in range(self.board_size):
            for c in range(self.board_size):
                if self.board[r, c] == 0:
                    legal_moves.append((r, c))
        
        return legal_moves
    
    def play_random_game(self):
        """Play a random game until completion."""
        self.reset()
        
        while not self.game_over:
            legal_moves = self.get_legal_moves()
            
            # End game if board is full
            if not legal_moves:
                self.game_over = True
                break
            
            # Make a random move
            row, col = random.choice(legal_moves)
            self.make_move(row, col)
    
    def get_game_data(self):
        """Get the game data in a structured format.
        
        Returns:
            Dict containing game data
        """
        return {
            "id": f"game_{random.randint(0, 1000000)}",
            "board_size": self.board_size,
            "moves": self.moves,
            "board_states": [board.tolist() for board in self.board_states[:-1]],  # Exclude final state
            "winner": self.winner
        }


def generate_games(num_games, board_size=15, output_dir="data/raw"):
    """Generate synthetic Gomoku games.
    
    Args:
        num_games: Number of games to generate
        board_size: Size of the Gomoku board
        output_dir: Directory to save the generated games
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    game = GomokuGame(board_size=board_size)
    
    for i in tqdm(range(num_games)):
        # Play a random game
        game.play_random_game()
        
        # Get game data
        game_data = game.get_game_data()
        
        # Save to file
        with open(output_path / f"game_{i}.json", "w") as f:
            json.dump(game_data, f, indent=2)
    
    print(f"Generated {num_games} games in {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic Gomoku game data")
    parser.add_argument("--num_games", type=int, default=100, help="Number of games to generate")
    parser.add_argument("--board_size", type=int, default=15, help="Size of the Gomoku board")
    parser.add_argument("--output_dir", type=str, default="data/raw", help="Directory to save the generated games")
    
    args = parser.parse_args()
    
    generate_games(
        num_games=args.num_games,
        board_size=args.board_size,
        output_dir=args.output_dir
    ) 