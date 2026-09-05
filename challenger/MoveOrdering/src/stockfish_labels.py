"""Offline Stockfish labels for training only; never import this in a submission."""

from __future__ import annotations

from collections.abc import Iterable

import chess
import chess.engine


def rank_legal_moves(
    board: chess.Board, engine: chess.engine.SimpleEngine, depth: int
) -> list[chess.Move]:
    """Rank every legal move by a root-restricted Stockfish search from the mover's POV."""
    mover = board.turn
    scored_moves: list[tuple[int, chess.Move]] = []
    for move in board.legal_moves:
        info = engine.analyse(board, chess.engine.Limit(depth=depth), root_moves=[move])
        score = info["score"].pov(mover).score(mate_score=100_000)
        if score is not None:
            scored_moves.append((score, move))
    return [move for _, move in sorted(scored_moves, key=lambda item: item[0], reverse=True)]


def move_pairs(ranked_moves: Iterable[chess.Move]) -> list[tuple[str, str]]:
    """Turn a descending ranking into every preferred UCI move pair."""
    moves = list(ranked_moves)
    return [
        (moves[better].uci(), moves[worse].uci())
        for better in range(len(moves))
        for worse in range(better + 1, len(moves))
    ]
