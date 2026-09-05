"""Ranking metrics for datasets containing all preference pairs per position."""

from __future__ import annotations

from collections import defaultdict

import chess
import torch

from .dataset import MovePairDataset, PairRecord
from .inference import order_moves
from .model import MoveOrderingModel


@torch.no_grad()
def top_k_agreement(model: MoveOrderingModel, dataset: MovePairDataset, k: int) -> float:
    """Fraction where the pair-derived Stockfish best move is in network top ``k``.

    The generator writes every ordered pair, so the sole move never appearing as
    ``bad_move`` is the recorded ranking's best move.
    """
    records_by_fen: dict[str, list[PairRecord]] = defaultdict(list)
    for record in dataset.records:
        records_by_fen[record["fen"]].append(record)
    hits = 0
    for fen, records in records_by_fen.items():
        candidates = {record["good_move"] for record in records} | {
            record["bad_move"] for record in records
        }
        bad_moves = {record["bad_move"] for record in records}
        best_moves = candidates - bad_moves
        if len(best_moves) != 1:
            raise ValueError("top-k metrics require complete, strict pair rankings per FEN")
        top_moves = {move.uci() for move in order_moves(chess.Board(fen), model)[:k]}
        hits += next(iter(best_moves)) in top_moves
    return hits / len(records_by_fen) if records_by_fen else 0.0
