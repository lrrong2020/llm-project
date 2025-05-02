from typing import List, Optional
from pydantic import BaseModel

class GameExample(BaseModel):
    game_id: str
    turn_idx: int
    state: str  # textual representation of the Gomoku board
    legal_moves: Optional[str] = None  # optional; available positions
    decision: str  # ground-truth move (e.g., "H8")
    
    class Config:
        frozen = True 