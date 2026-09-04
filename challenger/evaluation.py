"""Static evaluation backed by the quantized HalfKP 1024x32x1 NNUE."""

from __future__ import annotations

import sys
from pathlib import Path

import chess

# The local harness adds only ``challenger/`` to its import path.
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(1, project_root)

from nnues.evaluate_1024 import evaluate as evaluate_nnue


def evaluate(board: chess.Board, config: object | None = None) -> float:
    """Return a centipawn score from the side-to-move point of view."""
    del config  # Keep the existing search interface unchanged.
    return evaluate_nnue(board)
