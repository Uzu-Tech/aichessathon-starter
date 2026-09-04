"""Drop-in board evaluator backed by the quantized 1024x32x1 NNUE."""

from __future__ import annotations

import chess

if __package__:
    from .nnue_1024 import evaluate_board
else:
    from nnue_1024 import evaluate_board


def evaluate(board: chess.Board) -> float:
    """Return the NNUE score in centipawns from the side-to-move viewpoint."""
    return evaluate_board(board)


__all__ = ["evaluate"]
