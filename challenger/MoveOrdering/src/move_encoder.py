"""Encoding for UCI chess moves."""

from __future__ import annotations

from collections.abc import Sequence

import chess
import torch
from torch import Tensor

# python-chess uses None for no promotion.  Its piece constants fit 1..5.
NO_PROMOTION = 0
PROMOTION_VOCAB_SIZE = 6


def move_to_components(move: chess.Move) -> tuple[int, int, int]:
    """Convert a move to from-square, to-square, and promotion IDs."""
    promotion = NO_PROMOTION if move.promotion is None else move.promotion
    return move.from_square, move.to_square, promotion


def moves_to_tensor(moves: Sequence[chess.Move], device: torch.device | None = None) -> Tensor:
    """Return an ``[moves, 3]`` long tensor suitable for the model."""
    return torch.tensor(
        [move_to_components(move) for move in moves], dtype=torch.long, device=device
    )
