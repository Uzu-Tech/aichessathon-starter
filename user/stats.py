from dataclasses import dataclass
import chess
from typing import Optional


@dataclass
class SearchResult:
    depth: int
    budget_ms: float
    best_score: float = -float('inf')
    best_move: Optional[chess.Move] = None
    nodes: int = 0
    leaves: int = 0
    cutoffs: int = 0