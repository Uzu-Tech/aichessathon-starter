"""JSONL pair dataset and batching helpers for move-ordering training."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict

import chess
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .halfkp import halfkp_indices
from .move_encoder import move_to_components


class PairRecord(TypedDict):
    fen: str
    good_move: str
    bad_move: str


class MovePairDataset(Dataset[dict[str, Tensor]]):
    """One pair is a preference: ``good_move`` must score above ``bad_move``."""

    def __init__(self, path: str | Path) -> None:
        self.records: list[PairRecord] = []
        with Path(path).open(encoding="utf-8") as data_file:
            for line_number, line in enumerate(data_file, start=1):
                if line.strip():
                    record = json.loads(line)
                    if not {"fen", "good_move", "bad_move"} <= record.keys():
                        raise ValueError(f"line {line_number} is not a move-pair record")
                    self.records.append(record)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        record = self.records[index]
        board = chess.Board(record["fen"])
        return {
            "features": torch.tensor(halfkp_indices(board), dtype=torch.long),
            "good_move": torch.tensor(
                move_to_components(chess.Move.from_uci(record["good_move"])), dtype=torch.long
            ),
            "bad_move": torch.tensor(
                move_to_components(chess.Move.from_uci(record["bad_move"])), dtype=torch.long
            ),
        }


def collate_move_pairs(items: Sequence[dict[str, Tensor]]) -> dict[str, Tensor]:
    """Pad variable-length active-feature lists with ``-1`` for a DataLoader."""
    if not items:
        raise ValueError("cannot collate an empty batch")
    max_features = max(item["features"].numel() for item in items)
    features = torch.full((len(items), max_features), -1, dtype=torch.long)
    for index, item in enumerate(items):
        features[index, : item["features"].numel()] = item["features"]
    return {
        "features": features,
        "good_move": torch.stack([item["good_move"] for item in items]),
        "bad_move": torch.stack([item["bad_move"] for item in items]),
    }
