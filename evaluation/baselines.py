"""Fast, reproducible validation baselines for NNUE experiments."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import TypedDict

import numpy as np

from .training import DEFAULT_DATA_DIR, EvaluationMetrics, Split, WDL_EXPONENT, WDL_SCALE_CP

MATERIAL_VALUES_CP = np.asarray([100, 320, 330, 500, 900, 0], dtype=np.float64)


class _Totals(TypedDict):
    count: int
    wdl: float
    square: float
    absolute: float
    clipped_square: float
    sign_correct: int
    sign_count: int


def _empty_totals() -> _Totals:
    return {
        "count": 0,
        "wdl": 0.0,
        "square": 0.0,
        "absolute": 0.0,
        "clipped_square": 0.0,
        "sign_correct": 0,
        "sign_count": 0,
    }


def _white_to_move(archive: np.lib.npyio.NpzFile, path: Path) -> np.ndarray:
    if "white_to_move" in archive.files:
        return np.asarray(archive["white_to_move"], dtype=np.bool_)
    fens = np.asarray(archive["fen"])

    def is_white(value: object) -> bool:
        if isinstance(value, (bytes, np.bytes_)):
            fen = bytes(value).rstrip(b"\0").decode("ascii")
        else:
            fen = str(value).rstrip("\0")
        fields = fen.split()
        if len(fields) < 2 or fields[1] not in {"w", "b"}:
            raise ValueError(f"{path}: invalid FEN side-to-move field")
        return fields[1] == "w"

    return np.fromiter((is_white(value) for value in fens), dtype=np.bool_, count=len(fens))


def _update(totals: _Totals, predictions: np.ndarray, targets: np.ndarray) -> None:
    difference = predictions - targets
    prediction_wdl = 1.0 / (
        1.0 + np.exp(-np.clip(predictions / WDL_SCALE_CP, -50.0, 50.0))
    )
    target_wdl = 1.0 / (
        1.0 + np.exp(-np.clip(targets / WDL_SCALE_CP, -50.0, 50.0))
    )
    decisive = np.abs(targets) >= 50.0
    totals["count"] += len(targets)
    totals["wdl"] += float(
        np.power(np.abs(prediction_wdl - target_wdl), WDL_EXPONENT).sum()
    )
    totals["square"] += float(np.square(difference).sum())
    totals["absolute"] += float(np.abs(difference).sum())
    totals["clipped_square"] += float(
        np.square(
            np.clip(predictions, -2000.0, 2000.0)
            - np.clip(targets, -2000.0, 2000.0)
        ).sum()
    )
    totals["sign_correct"] += int(
        np.count_nonzero((np.sign(predictions) == np.sign(targets)) & decisive)
    )
    totals["sign_count"] += int(np.count_nonzero(decisive))


def _finish(name: str, totals: _Totals, seconds: float) -> EvaluationMetrics:
    count = totals["count"]
    if count == 0:
        raise ValueError(f"baseline {name!r} evaluated no positions")
    return EvaluationMetrics(
        split=name,
        positions=count,
        wdl_loss=totals["wdl"] / count,
        rmse_cp=math.sqrt(totals["square"] / count),
        mae_cp=totals["absolute"] / count,
        clipped_rmse_cp=math.sqrt(totals["clipped_square"] / count),
        sign_accuracy=(
            totals["sign_correct"] / totals["sign_count"]
            if totals["sign_count"]
            else math.nan
        ),
        seconds=seconds,
    )


def evaluate_baselines(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    *,
    split: Split = "validation",
    max_shards: int | None = None,
) -> dict[str, EvaluationMetrics]:
    """Evaluate zero and material baselines against exactly the same split as models."""
    if max_shards is not None and max_shards < 1:
        raise ValueError("max_shards must be positive or None")
    paths = sorted((Path(data_dir).expanduser().resolve() / split).glob("shard_*.npz"))
    if max_shards is not None:
        paths = paths[:max_shards]
    if not paths:
        raise FileNotFoundError(f"no {split} shards found beneath {data_dir}")

    started = time.perf_counter()
    totals = {"zero": _empty_totals(), "material": _empty_totals()}
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            targets_white = np.asarray(archive["evaluation_cp"], dtype=np.float64)
            white_to_move = _white_to_move(archive, path)
            piece_counts = np.asarray(archive["piece768"]).reshape(-1, 12, 64).sum(
                axis=2, dtype=np.int16
            )
        targets = np.where(white_to_move, targets_white, -targets_white)
        material_white = (piece_counts[:, :6] - piece_counts[:, 6:]) @ MATERIAL_VALUES_CP
        material = np.where(white_to_move, material_white, -material_white)
        _update(totals["zero"], np.zeros_like(targets), targets)
        _update(totals["material"], material, targets)

    seconds = time.perf_counter() - started
    return {name: _finish(name, values, seconds) for name, values in totals.items()}


def format_metrics(name: str, metrics: EvaluationMetrics) -> str:
    return (
        f"{name:20s}  WDL={metrics.wdl_loss:.9f}  "
        f"RMSE={metrics.rmse_cp:8.2f} cp  MAE={metrics.mae_cp:8.2f} cp  "
        f"clip-RMSE={metrics.clipped_rmse_cp:7.2f} cp  "
        f"sign={metrics.sign_accuracy:6.2%}"
    )


__all__ = ["evaluate_baselines", "format_metrics"]
