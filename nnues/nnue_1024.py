"""Quantized incremental evaluator for the HalfKP 1024x32x1 model."""

from __future__ import annotations

from pathlib import Path

import chess

if __package__:
    from .nnue_runtime import NNUEBackend, NNUEPosition
else:
    from nnue_runtime import NNUEBackend, NNUEPosition

MODEL_DIR = Path(__file__).resolve().parent / "weights"
BACKEND = NNUEBackend(
    MODEL_DIR / "halfkp_1024_transformer_q8.npy",
    MODEL_DIR / "halfkp_1024_bias_q32.npy",
    MODEL_DIR / "halfkp_1024_scale_f32.npy",
    MODEL_DIR / "halfkp_1024_head_q8.onnx",
    width=1024,
)


def position(fen: str) -> NNUEPosition:
    """Create a search position with fresh HalfKP accumulators."""
    return BACKEND.position(fen)


def evaluate_fen(fen: str) -> float:
    """Evaluate one FEN in centipawns from its side-to-move point of view."""
    return position(fen).evaluate()


def evaluate_board(board: chess.Board) -> float:
    """Evaluate an existing board without serializing and reparsing its FEN."""
    return NNUEPosition(BACKEND, board).evaluate()


__all__ = ["evaluate_board", "evaluate_fen", "position"]
