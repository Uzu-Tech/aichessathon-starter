"""Move-ordering inference API; it deliberately does not search or evaluate."""

from __future__ import annotations

import chess
import torch

from .model import MoveOrderingModel


@torch.no_grad()
def order_moves(board: chess.Board, model: MoveOrderingModel) -> list[chess.Move]:
    """Return all legal moves, descending by the model's one scalar score."""
    moves = list(board.legal_moves)
    if not moves:
        return []
    was_training = model.training
    model.eval()
    scores = model.score_moves(model.encode_position(board), moves).cpu().tolist()
    model.train(was_training)
    return [
        move
        for _, move in sorted(
            zip(scores, moves, strict=True), reverse=True, key=lambda item: item[0]
        )
    ]
