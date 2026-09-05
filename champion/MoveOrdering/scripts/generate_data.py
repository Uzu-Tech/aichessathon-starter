"""Extract PGN positions and write Stockfish-derived move-pair JSONL labels."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import chess
import chess.engine
import chess.pgn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.stockfish_labels import move_pairs, rank_legal_moves


def sampled_positions(pgn_path: Path, sample_rate: float, seed: int):
    """Yield ordinary, non-terminal positions sampled from main-line PGN games."""
    randomizer = random.Random(seed)
    with pgn_path.open(encoding="utf-8", errors="replace") as pgn_file:
        while game := chess.pgn.read_game(pgn_file):
            board = game.board()
            for move in game.mainline_moves():
                board.push(move)
                if not board.is_game_over() and randomizer.random() < sample_rate:
                    yield board.copy(stack=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create relative move-ordering labels from PGN positions."
    )
    parser.add_argument("--pgn", required=True, type=Path)
    parser.add_argument(
        "--stockfish", required=True, type=Path, help="Path to a local Stockfish executable"
    )
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sample-rate", type=float, default=0.02)
    parser.add_argument("--max-positions", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()
    if not 0.0 < arguments.sample_rate <= 1.0:
        parser.error("--sample-rate must be in (0, 1]")

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    written_positions = 0
    with (
        chess.engine.SimpleEngine.popen_uci(str(arguments.stockfish)) as engine,
        arguments.output.open("w", encoding="utf-8") as output,
    ):
        for board in sampled_positions(arguments.pgn, arguments.sample_rate, arguments.seed):
            ranking = rank_legal_moves(board, engine, arguments.depth)
            for good_move, bad_move in move_pairs(ranking):
                output.write(
                    json.dumps({"fen": board.fen(), "good_move": good_move, "bad_move": bad_move})
                    + "\n"
                )
            written_positions += 1
            if written_positions >= arguments.max_positions:
                break
    print(f"wrote pairs for {written_positions} positions to {arguments.output}")


if __name__ == "__main__":
    main()
