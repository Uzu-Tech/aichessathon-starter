"""A small, practical training toolkit for the evaluation networks.

The common path is deliberately one function::

    from training import train

    run = train(epochs=10)              # HalfKP_hm NNUE by default
    print(run.test_rmse_cp)
    model = run.load()

The public helpers ``inspect_data()``, ``evaluate()``, ``load_model()``,
``predict()`` and ``plot_history()`` cover the usual follow-up work.

The HalfKP implementation follows the important NNUE structure: one shared
sparse affine feature transformer creates separate accumulators for both
king perspectives, the accumulators are ordered side-to-move first, and a
clipped activation precedes the small dense head. Existing ``halfkp32``
shards are remapped while loading, so they do not need to be regenerated.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import statistics
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, cast

import chess
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

Representation = Literal["piece768", "halfkp32"]
Split = Literal["train", "validation", "test"]
Perspective = Literal["side_to_move", "white"]
Objective = Literal["wdl", "rmse", "mae"]

MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = MODULE_DIR / "data"
DEFAULT_OUTPUT_DIR = MODULE_DIR / "models"
DEFAULT_ARCHITECTURE = "256x32x32x1"

PIECE_FEATURES = 768
HALFKP_FEATURES_PER_PERSPECTIVE = 64 * 10 * 64
HALFKP_FEATURES = 2 * HALFKP_FEATURES_PER_PERSPECTIVE
HALFKP_MIRRORED_FEATURES = 32 * 10 * 64
HALFKP_SLOTS = 64
PADDING_INDEX = np.uint32(0xFFFFFFFF)
WDL_SCALE_CP = 400.0
WDL_EXPONENT = 2.5
CHECKPOINT_VERSION = 2

_FEATURE_COLUMN: dict[Representation, str] = {
    "piece768": "piece768",
    "halfkp32": "halfkp32",
}


@dataclass(frozen=True)
class Architecture:
    """Feature-transformer width, dense widths, and scalar output width."""

    widths: tuple[int, ...]

    @property
    def name(self) -> str:
        return "x".join(str(width) for width in self.widths)


def parse_architecture(spec: str | Sequence[int] | Architecture) -> Architecture:
    """Parse ``"256x32x32x1"`` or a sequence such as ``(256, 32, 32, 1)``."""
    if isinstance(spec, Architecture):
        return spec
    try:
        if isinstance(spec, str):
            parts = re.split(r"[xX, ]+", spec.strip())
            widths = tuple(int(part) for part in parts if part)
        else:
            widths = tuple(int(width) for width in spec)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid architecture: {spec!r}") from error
    if widths and widths[-1] != 1:
        widths += (1,)
    if len(widths) < 2 or any(width <= 0 for width in widths):
        raise ValueError("architecture needs a positive transformer width and output width 1")
    if widths[-1] != 1:
        raise ValueError("the final architecture width must be 1")
    return Architecture(widths)


def _representation(value: str) -> Representation:
    if value not in _FEATURE_COLUMN:
        choices = ", ".join(_FEATURE_COLUMN)
        raise ValueError(f"unknown representation {value!r}; choose {choices}")
    return cast(Representation, value)


@dataclass(frozen=True)
class SplitInfo:
    name: str
    positions: int
    shards: int


@dataclass(frozen=True)
class DatasetInfo:
    """A compact description returned by :func:`inspect_data`."""

    data_dir: str
    splits: tuple[SplitInfo, ...]
    representations: tuple[str, ...]
    metadata_found: bool

    @property
    def positions(self) -> int:
        return sum(split.positions for split in self.splits)

    def __str__(self) -> str:
        split_text = ", ".join(
            f"{item.name}={item.positions:,} ({item.shards} shards)" for item in self.splits
        )
        metadata = "yes" if self.metadata_found else "no"
        columns = ",".join(self.representations)
        return f"Dataset({split_text}; columns={columns}; metadata={metadata})"


def _validate_shard(archive: Any, path: Path, *, check_values: bool) -> int:
    required = {"fen", "evaluation_cp", "piece768", "halfkp32"}
    missing = required.difference(archive.files)
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")

    targets = np.asarray(archive["evaluation_cp"])
    fens = np.asarray(archive["fen"])
    pieces = np.asarray(archive["piece768"])
    halfkp = np.asarray(archive["halfkp32"])
    rows = len(targets)
    if targets.shape != (rows,) or not np.issubdtype(targets.dtype, np.floating):
        raise ValueError(f"{path}: evaluation_cp must be a one-dimensional float array")
    if fens.shape != (rows,) or fens.dtype.kind not in {"U", "S"}:
        raise ValueError(f"{path}: fen must be a one-dimensional string array")
    if pieces.shape != (rows, PIECE_FEATURES) or pieces.dtype != np.uint8:
        raise ValueError(f"{path}: piece768 must have shape (N, 768) and dtype uint8")
    if halfkp.shape != (rows, HALFKP_SLOTS) or halfkp.dtype != np.uint32:
        raise ValueError(f"{path}: halfkp32 must have shape (N, 64) and dtype uint32")
    if check_values:
        if np.any((pieces != 0) & (pieces != 1)):
            raise ValueError(f"{path}: piece768 contains values other than zero and one")
        valid_halfkp = (halfkp < HALFKP_FEATURES) | (halfkp == PADDING_INDEX)
        if not bool(np.all(valid_halfkp)):
            raise ValueError(f"{path}: halfkp32 contains an invalid feature index")
        if not bool(np.all(np.isfinite(targets))):
            raise ValueError(f"{path}: evaluation_cp contains a non-finite value")
    return rows


def inspect_data(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    *,
    validate_all: bool = False,
) -> DatasetInfo:
    """Validate the synchronized dataset and report its split sizes.

    The first shard of every split is always checked fully. ``validate_all``
    performs value checks on every shard and is intentionally slower.
    """
    root = Path(data_dir).expanduser().resolve()
    split_info: list[SplitInfo] = []
    for split in ("train", "validation", "test"):
        files = sorted((root / split).glob("shard_*.npz"))
        if not files:
            raise FileNotFoundError(f"no shard_*.npz files found in {root / split}")
        positions = 0
        for index, path in enumerate(files):
            try:
                with np.load(path, allow_pickle=False) as archive:
                    if index == 0 or validate_all:
                        positions += _validate_shard(
                            archive, path, check_values=validate_all or index == 0
                        )
                    else:
                        targets = np.asarray(archive["evaluation_cp"])
                        if targets.ndim != 1:
                            raise ValueError(
                                f"{path}: evaluation_cp must be a one-dimensional array"
                            )
                        positions += len(targets)
            except (OSError, ValueError) as error:
                raise ValueError(f"could not read dataset shard {path}: {error}") from error
        split_info.append(SplitInfo(split, positions, len(files)))
    metadata_found = all((root / split.name / "metadata.json").exists() for split in split_info)
    return DatasetInfo(
        data_dir=str(root),
        splits=tuple(split_info),
        representations=("halfkp32", "piece768"),
        metadata_found=metadata_found,
    )


def _side_to_move(fens: np.ndarray, path: Path) -> np.ndarray:
    result = np.empty(len(fens), dtype=np.bool_)
    for index, fen_value in enumerate(fens):
        if isinstance(fen_value, (bytes, np.bytes_)):
            try:
                fen = bytes(fen_value).rstrip(b"\0").decode("ascii")
            except UnicodeDecodeError as error:
                raise ValueError(
                    f"{path}: FEN at row {index} is not ASCII"
                ) from error
        elif isinstance(fen_value, (str, np.str_)):
            fen = str(fen_value).rstrip("\0")
        else:
            raise ValueError(
                f"{path}: unsupported FEN value at row {index}: "
                f"{type(fen_value).__name__}"
            )
        fields = fen.split()
        if len(fields) < 2 or fields[1] not in {"w", "b"}:
            raise ValueError(f"{path}: invalid side-to-move field at row {index}")
        result[index] = fields[1] == "w"
    return result


Batch = tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]


class ShardDataset(IterableDataset[Batch]):
    """Stream already-batched slices from compressed NPZ shards.

    Batching inside the dataset avoids millions of Python-level per-position
    yields and keeps the stored uint8/uint32 feature dtypes until each batch is
    moved to the selected device.
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        representation: Representation,
        batch_size: int = 4096,
        *,
        shuffle: bool = True,
        seed: int = 42,
    ) -> None:
        super().__init__()
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.root = Path(root).expanduser().resolve()
        self.split = split
        self.representation = _representation(representation)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.files = sorted((self.root / split).glob("shard_*.npz"))
        if not self.files:
            raise FileNotFoundError(f"no shards found in {self.root / split}")
        self.shard_rows: list[int] = []
        for path in self.files:
            with np.load(path, allow_pickle=False) as archive:
                if _FEATURE_COLUMN[self.representation] not in archive.files:
                    raise ValueError(
                        f"{path} does not contain {_FEATURE_COLUMN[self.representation]!r}"
                    )
                self.shard_rows.append(len(np.asarray(archive["evaluation_cp"])))
        self.count = sum(self.shard_rows)
        self._iteration = 0

    def __len__(self) -> int:
        return sum(math.ceil(rows / self.batch_size) for rows in self.shard_rows)

    def __iter__(self) -> Iterator[Batch]:
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        worker_count = 1 if worker is None else worker.num_workers
        files = self.files[worker_id::worker_count]
        iteration = self._iteration
        self._iteration += 1
        rng = np.random.default_rng(self.seed + worker_id + iteration * 1_000_003)
        if self.shuffle:
            files = [files[index] for index in rng.permutation(len(files))]

        column = _FEATURE_COLUMN[self.representation]
        for path in files:
            with np.load(path, allow_pickle=False) as archive:
                features = np.asarray(archive[column])
                target_white_cp = np.asarray(archive["evaluation_cp"], dtype=np.float32)
                if "white_to_move" in archive.files:
                    white_to_move = np.asarray(archive["white_to_move"], dtype=np.bool_)
                else:
                    white_to_move = _side_to_move(np.asarray(archive["fen"]), path)

            if len(features) != len(target_white_cp) or len(white_to_move) != len(features):
                raise ValueError(f"{path}: feature, target, and side-to-move lengths differ")
            # Lichess evaluations are White POV. NNUE/negamax convention is STM POV.
            target_cp = np.where(white_to_move, target_white_cp, -target_white_cp).astype(
                np.float32, copy=False
            )
            order: np.ndarray | None = None
            if self.shuffle:
                order = rng.permutation(len(target_cp))
            for start in range(0, len(target_cp), self.batch_size):
                stop = min(start + self.batch_size, len(target_cp))
                selection = slice(start, stop) if order is None else order[start:stop]
                batch_features = np.ascontiguousarray(features[selection])
                batch_side = np.ascontiguousarray(white_to_move[selection])
                batch_targets = np.ascontiguousarray(target_cp[selection])
                yield (
                    (torch.from_numpy(batch_features), torch.from_numpy(batch_side)),
                    torch.from_numpy(batch_targets),
                )


class ClippedReLU(nn.Module):
    """Quantization-friendly ReLU with output range [0, 1]."""

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return torch.clamp(values, 0.0, 1.0)


class NNUE(nn.Module):
    """A compact dual-perspective NNUE evaluator.

    ``forward(features, white_to_move)`` returns centipawns from the side-to-
    move point of view. For ``halfkp32``, the large first affine transform is
    implemented as sums of embedding rows, which is mathematically identical
    to applying a Linear layer to the sparse binary feature vector.
    """

    def __init__(
        self,
        representation: Representation,
        architecture: str | Sequence[int] | Architecture = DEFAULT_ARCHITECTURE,
        *,
        mirror_halfkp: bool = True,
        output_scale_cp: float = WDL_SCALE_CP,
        sparse_gradients: bool = False,
    ) -> None:
        super().__init__()
        self.representation = _representation(representation)
        self.architecture = parse_architecture(architecture)
        self.mirror_halfkp = mirror_halfkp
        self.output_scale_cp = float(output_scale_cp)
        self.sparse_gradients = sparse_gradients
        if not math.isfinite(self.output_scale_cp) or self.output_scale_cp <= 0:
            raise ValueError("output_scale_cp must be positive and finite")

        transformer_width = self.architecture.widths[0]
        if self.representation == "halfkp32":
            feature_count = (
                HALFKP_MIRRORED_FEATURES
                if self.mirror_halfkp
                else HALFKP_FEATURES_PER_PERSPECTIVE
            )
            self.feature_transformer: nn.Module = nn.EmbeddingBag(
                feature_count,
                transformer_width,
                mode="sum",
                include_last_offset=True,
                sparse=sparse_gradients,
            )
            nn.init.normal_(self.feature_transformer.weight, mean=0.0, std=0.01)
            self.feature_bias = nn.Parameter(torch.zeros(transformer_width))
        else:
            self.feature_transformer = nn.Linear(PIECE_FEATURES, transformer_width)
            nn.init.normal_(self.feature_transformer.weight, mean=0.0, std=0.01)
            nn.init.zeros_(self.feature_transformer.bias)
            self.register_parameter("feature_bias", None)

        layers: list[nn.Module] = []
        input_width = transformer_width * 2
        for output_width in self.architecture.widths[1:]:
            layers.append(nn.Linear(input_width, output_width))
            if output_width != 1:
                layers.append(ClippedReLU())
            input_width = output_width
        self.input_activation = ClippedReLU()
        self.head = nn.Sequential(*layers)
        rank_flip = torch.arange(64, dtype=torch.long).bitwise_xor(56)
        self.register_buffer("rank_flip", rank_flip, persistent=False)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def size_mib_float32(self) -> float:
        return self.parameter_count * 4 / (1024**2)

    def _piece_accumulators(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        transformer = cast(nn.Linear, self.feature_transformer)
        board = features.to(dtype=transformer.weight.dtype).reshape(-1, 12, 64)
        white_view = board
        black_view = torch.cat((board[:, 6:], board[:, :6]), dim=1).index_select(
            2, self.rank_flip
        )
        return (
            transformer(white_view.flatten(1)),
            transformer(black_view.flatten(1)),
        )

    def _normalize_halfkp(self, indices: torch.Tensor, *, black: bool) -> torch.Tensor:
        king = torch.div(indices, 640, rounding_mode="floor")
        remainder = indices.remainder(640)
        piece_square = remainder.remainder(64)
        piece_plane = remainder - piece_square
        if black:
            # prepare_datasets already flips Black's piece squares, but not its king.
            king = king.bitwise_xor(56)
        if not self.mirror_halfkp:
            return king * 640 + piece_plane + piece_square

        mirror = king.remainder(8) < 4
        king = torch.where(mirror, king.bitwise_xor(7), king)
        piece_square = torch.where(mirror, piece_square.bitwise_xor(7), piece_square)
        king_bucket = torch.div(king, 8, rounding_mode="floor") * 4 + king.remainder(8) - 4
        return king_bucket * 640 + piece_plane + piece_square

    def _embedding_bag(self, indices: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        transformer = cast(nn.EmbeddingBag, self.feature_transformer)
        counts = mask.sum(dim=1, dtype=torch.long)
        offsets = torch.cat(
            (torch.zeros(1, dtype=torch.long, device=indices.device), counts.cumsum(dim=0))
        )
        packed = indices.masked_select(mask)
        return transformer(packed, offsets) + cast(torch.Tensor, self.feature_bias)

    def _halfkp_accumulators(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        indices = features.to(dtype=torch.long)
        white_mask = (indices >= 0) & (indices < HALFKP_FEATURES_PER_PERSPECTIVE)
        black_mask = (indices >= HALFKP_FEATURES_PER_PERSPECTIVE) & (
            indices < HALFKP_FEATURES
        )
        white_indices = self._normalize_halfkp(indices, black=False)
        black_indices = self._normalize_halfkp(
            (indices - HALFKP_FEATURES_PER_PERSPECTIVE).clamp_min(0), black=True
        )
        return (
            self._embedding_bag(white_indices, white_mask),
            self._embedding_bag(black_indices, black_mask),
        )

    def forward(self, features: torch.Tensor, white_to_move: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2:
            raise ValueError("features must have shape (batch, feature_width)")
        side = white_to_move.to(device=features.device, dtype=torch.bool).reshape(-1)
        if len(side) != len(features):
            raise ValueError("white_to_move must contain one value per position")
        if self.representation == "piece768":
            if features.shape[1] != PIECE_FEATURES:
                raise ValueError("piece768 features must have width 768")
            white, black = self._piece_accumulators(features)
        else:
            if features.shape[1] != HALFKP_SLOTS:
                raise ValueError("halfkp32 features must have width 64")
            white, black = self._halfkp_accumulators(features)

        first = torch.where(side[:, None], white, black)
        second = torch.where(side[:, None], black, white)
        accumulator = self.input_activation(torch.cat((first, second), dim=1))
        return self.head(accumulator).squeeze(-1) * self.output_scale_cp


@dataclass(frozen=True)
class EvaluationMetrics:
    split: str
    positions: int
    wdl_loss: float
    rmse_cp: float
    mae_cp: float
    clipped_rmse_cp: float
    sign_accuracy: float
    seconds: float


@dataclass(frozen=True)
class ModelInfo:
    representation: str
    architecture: str
    parameters: int
    float32_mib: float
    float16_or_int16_mib: float
    mirror_halfkp: bool


@dataclass(frozen=True)
class BenchmarkResult:
    device: str
    batch_size: int
    milliseconds_per_batch: float
    microseconds_per_position: float
    positions_per_second: float


@dataclass(frozen=True)
class AccumulatorBenchmarkResult:
    device: str
    iterations: int
    parameter_mib: float
    accumulator_kib: float
    full_microseconds: float
    accumulated_microseconds: float
    accumulated_positions_per_second: float
    speedup: float


@dataclass(frozen=True)
class GameBenchmarkResult:
    """Timing for a legal move followed by an incremental NNUE evaluation."""

    device: str
    games: int
    plies_per_game: int
    positions: int
    update_microseconds: float
    inference_microseconds: float
    total_microseconds: float
    positions_per_second: float
    max_error_cp: float


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    train_loss: float
    validation_rmse_cp: float
    validation_loss: float = math.nan
    validation_mae_cp: float = math.nan
    seconds: float = 0.0
    learning_rate: float = 0.0
    validation_objective: float = math.nan


@dataclass
class TrainingResult:
    """Everything needed to inspect or reuse a completed training run."""

    # The first five fields intentionally preserve the original notebook API.
    architecture: str
    representation: str
    epochs: list[EpochMetrics]
    test_rmse_cp: float
    model_path: str
    data_dir: str = str(DEFAULT_DATA_DIR)
    objective: str = "wdl"
    best_epoch: int = 0
    best_validation_loss: float = math.inf
    best_validation_metric: float = math.inf
    test_metrics: EvaluationMetrics | None = None
    parameters: int = 0
    model_size_mib: float = 0.0
    training_seconds: float = 0.0
    device: str = "cpu"

    def load(self, device: str | torch.device | None = "auto") -> NNUE:
        return load_model(self.model_path, device=device)

    def evaluate(
        self,
        split: Split = "test",
        *,
        batch_size: int = 4096,
        workers: int = 0,
        device: str | torch.device | None = "auto",
    ) -> EvaluationMetrics:
        return evaluate(
            self,
            data_dir=self.data_dir,
            split=split,
            batch_size=batch_size,
            workers=workers,
            device=device,
        )

    def plot(self, output_path: str | Path | None = None) -> Any:
        return plot_history(self, output_path)

    def info(self, device: str | torch.device | None = "cpu") -> ModelInfo:
        return model_info(self, device=device)

    def benchmark(
        self,
        *,
        batch_size: int = 1,
        device: str | torch.device | None = "cpu",
    ) -> BenchmarkResult:
        return benchmark(self, batch_size=batch_size, device=device)


def _loader(
    root: str | Path,
    split: str,
    representation: Representation,
    batch_size: int,
    workers: int,
    *,
    shuffle: bool = False,
    seed: int = 42,
    pin_memory: bool = False,
) -> DataLoader[Batch]:
    if workers < 0:
        raise ValueError("workers cannot be negative")
    dataset = ShardDataset(
        root,
        split,
        representation,
        batch_size,
        shuffle=shuffle,
        seed=seed,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    worker_options: dict[str, Any] = {}
    if workers > 0:
        worker_options.update(persistent_workers=True, prefetch_factor=2)
    return DataLoader(
        dataset,
        batch_size=None,
        num_workers=workers,
        pin_memory=pin_memory,
        generator=generator,
        **worker_options,
    )


def recommended_workers(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    *,
    split: Split = "train",
    cap: int = 8,
) -> int:
    """Choose useful loader parallelism without creating more workers than shards."""
    if cap < 1:
        raise ValueError("worker cap must be positive")
    shard_count = len(list((Path(data_dir).expanduser().resolve() / split).glob("shard_*.npz")))
    if shard_count == 0:
        return 0
    available_cpus = max(1, os.cpu_count() or 1)
    return min(cap, shard_count, max(1, available_cpus - 1))


def _device(value: str | torch.device | None) -> torch.device:
    if value is None or str(value).lower() == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    selected = torch.device(value)
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    if selected.type == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise ValueError("MPS was requested but is not available")
    return selected


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def wdl_loss(
    predictions_cp: torch.Tensor,
    targets_cp: torch.Tensor,
    *,
    scale_cp: float = WDL_SCALE_CP,
    exponent: float = WDL_EXPONENT,
) -> torch.Tensor:
    """Return the bounded NNUE loss after mapping centipawns to WDL space."""
    prediction_wdl = torch.sigmoid(predictions_cp / scale_cp)
    target_wdl = torch.sigmoid(targets_cp / scale_cp)
    return torch.mean(torch.abs(prediction_wdl - target_wdl).pow(exponent))


def objective_loss(
    predictions_cp: torch.Tensor,
    targets_cp: torch.Tensor,
    objective: Objective = "wdl",
) -> torch.Tensor:
    """Return the differentiable loss used by one training objective."""
    if objective == "wdl":
        return wdl_loss(predictions_cp, targets_cp)
    difference = predictions_cp - targets_cp
    if objective == "rmse":
        return torch.sqrt(torch.mean(difference.square()) + 1e-12)
    if objective == "mae":
        return torch.mean(difference.abs())
    raise ValueError("objective must be 'wdl', 'rmse', or 'mae'")


def _validation_objective(metrics: EvaluationMetrics, objective: Objective) -> float:
    if objective == "wdl":
        return metrics.wdl_loss
    if objective == "rmse":
        return metrics.rmse_cp
    return metrics.mae_cp


def _evaluate_loader(
    model: NNUE,
    loader: DataLoader[Batch],
    device: torch.device,
    *,
    split: str,
    max_batches: int | None = None,
) -> EvaluationMetrics:
    model.eval()
    started = time.perf_counter()
    squared_error = 0.0
    absolute_error = 0.0
    clipped_squared_error = 0.0
    total_loss = 0.0
    count = 0
    sign_correct = 0
    sign_count = 0
    with torch.inference_mode():
        for batch_number, ((features, white_to_move), targets) in enumerate(loader, start=1):
            if max_batches is not None and batch_number > max_batches:
                break
            features = features.to(device, non_blocking=True)
            white_to_move = white_to_move.to(device, non_blocking=True)
            targets = targets.to(device, dtype=torch.float32, non_blocking=True)
            predictions = model(features, white_to_move).float()
            difference = predictions.double() - targets.double()
            squared_error += torch.sum(difference.square()).item()
            absolute_error += torch.sum(difference.abs()).item()
            clipped_difference = predictions.clamp(-2000.0, 2000.0).double() - targets.clamp(
                -2000.0, 2000.0
            ).double()
            clipped_squared_error += torch.sum(clipped_difference.square()).item()
            loss = wdl_loss(predictions, targets)
            total_loss += loss.item() * len(targets)
            decisive = targets.abs() >= 50.0
            sign_correct += int(torch.sum((predictions.sign() == targets.sign()) & decisive).item())
            sign_count += int(torch.sum(decisive).item())
            count += len(targets)
    if count == 0:
        raise ValueError(f"the {split!r} loader produced no positions")
    return EvaluationMetrics(
        split=split,
        positions=count,
        wdl_loss=total_loss / count,
        rmse_cp=math.sqrt(squared_error / count),
        mae_cp=absolute_error / count,
        clipped_rmse_cp=math.sqrt(clipped_squared_error / count),
        sign_accuracy=sign_correct / sign_count if sign_count else math.nan,
        seconds=time.perf_counter() - started,
    )


def evaluate_model(
    model: nn.Module,
    loader: DataLoader[Batch],
    device: torch.device,
) -> float:
    """Compatibility helper returning RMSE in centipawns for an existing loader."""
    if not isinstance(model, NNUE):
        raise TypeError("evaluate_model expects an NNUE model")
    return _evaluate_loader(model, loader, device, split="custom").rmse_cp


def save_model(
    model: NNUE,
    path: str | Path,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Save a self-describing, weights-only checkpoint that :func:`load_model` can open."""
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": CHECKPOINT_VERSION,
        "representation": model.representation,
        "architecture": list(model.architecture.widths),
        "mirror_halfkp": model.mirror_halfkp,
        "output_scale_cp": model.output_scale_cp,
        "state_dict": model.state_dict(),
        "metadata": dict(metadata or {}),
    }
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)
    return destination


def load_model(
    path: str | Path,
    device: str | torch.device | None = "auto",
    *,
    sparse_gradients: bool = False,
) -> NNUE:
    """Load a checkpoint without separately specifying its architecture."""
    source = Path(path).expanduser().resolve()
    selected_device = _device(device)
    try:
        payload = torch.load(source, map_location=selected_device, weights_only=True)
    except TypeError:  # pragma: no cover - compatibility with older local PyTorch
        payload = torch.load(source, map_location=selected_device)
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise ValueError(
            f"{source} is a legacy state_dict, not a self-describing training checkpoint"
        )
    version = int(payload.get("format_version", 0))
    if version != CHECKPOINT_VERSION:
        raise ValueError(f"unsupported checkpoint format {version} in {source}")
    model = NNUE(
        _representation(str(payload["representation"])),
        cast(Sequence[int], payload["architecture"]),
        mirror_halfkp=bool(payload["mirror_halfkp"]),
        output_scale_cp=float(payload["output_scale_cp"]),
        sparse_gradients=sparse_gradients,
    )
    model.load_state_dict(cast(dict[str, torch.Tensor], payload["state_dict"]))
    model.to(selected_device)
    model.eval()
    return model


ModelSource = NNUE | TrainingResult | str | Path


def _resolve_model(
    source: ModelSource,
    device: str | torch.device | None,
    *,
    sparse_gradients: bool = False,
) -> tuple[NNUE, torch.device]:
    selected_device = _device(device)
    if isinstance(source, TrainingResult):
        model = load_model(
            source.model_path, selected_device, sparse_gradients=sparse_gradients
        )
    elif isinstance(source, NNUE):
        model = source.to(selected_device)
    else:
        model = load_model(source, selected_device, sparse_gradients=sparse_gradients)
    return model, selected_device


def model_info(
    model_or_checkpoint: ModelSource,
    *,
    device: str | torch.device | None = "cpu",
) -> ModelInfo:
    """Report the model's parameter and weight-memory budget."""
    model, _ = _resolve_model(model_or_checkpoint, device)
    return ModelInfo(
        representation=model.representation,
        architecture=model.architecture.name,
        parameters=model.parameter_count,
        float32_mib=model.size_mib_float32,
        float16_or_int16_mib=model.parameter_count * 2 / (1024**2),
        mirror_halfkp=model.mirror_halfkp,
    )


def evaluate(
    model_or_checkpoint: ModelSource,
    data_dir: str | Path | None = None,
    *,
    split: Split = "test",
    batch_size: int = 4096,
    workers: int = 0,
    device: str | torch.device | None = "auto",
    max_batches: int | None = None,
) -> EvaluationMetrics:
    """Evaluate a model, checkpoint, or :class:`TrainingResult` on one split."""
    model, selected_device = _resolve_model(model_or_checkpoint, device)
    if data_dir is None and isinstance(model_or_checkpoint, TrainingResult):
        data_dir = model_or_checkpoint.data_dir
    root = DEFAULT_DATA_DIR if data_dir is None else data_dir
    loader = _loader(
        root,
        split,
        model.representation,
        batch_size,
        workers,
        shuffle=False,
        pin_memory=selected_device.type == "cuda",
    )
    return _evaluate_loader(
        model,
        loader,
        selected_device,
        split=split,
        max_batches=max_batches,
    )


def predict(
    model_or_checkpoint: ModelSource,
    features: np.ndarray | torch.Tensor,
    white_to_move: bool | Sequence[bool] | np.ndarray | torch.Tensor,
    *,
    perspective: Perspective = "side_to_move",
    batch_size: int = 4096,
    device: str | torch.device | None = "auto",
) -> np.ndarray:
    """Predict centipawns for encoded positions.

    ``features`` is either an ``(N, 64)`` halfkp32 array or an ``(N, 768)``
    piece768 array matching the checkpoint. The default result is side-to-
    move POV; set ``perspective="white"`` for White POV scores.
    """
    if perspective not in {"side_to_move", "white"}:
        raise ValueError("perspective must be 'side_to_move' or 'white'")
    model, selected_device = _resolve_model(model_or_checkpoint, device)
    values = torch.as_tensor(features)
    single = values.ndim == 1
    if single:
        values = values.unsqueeze(0)
    if values.ndim != 2:
        raise ValueError("features must be a one- or two-dimensional array")
    if isinstance(white_to_move, bool):
        sides = torch.full((len(values),), white_to_move, dtype=torch.bool)
    else:
        sides = torch.as_tensor(white_to_move, dtype=torch.bool).reshape(-1)
    if len(sides) != len(values):
        raise ValueError("white_to_move must have one value per position")

    outputs: list[torch.Tensor] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(values), batch_size):
            batch_values = values[start : start + batch_size].to(selected_device)
            batch_sides = sides[start : start + batch_size].to(selected_device)
            scores = model(batch_values, batch_sides)
            if perspective == "white":
                scores = torch.where(batch_sides, scores, -scores)
            outputs.append(scores.cpu())
    result = torch.cat(outputs).numpy()
    return result.reshape(1) if single else result


def _benchmark_features(model: NNUE, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    if model.representation == "piece768":
        features = torch.zeros((batch_size, PIECE_FEATURES), dtype=torch.uint8)
        # Two kings and representative pawns/minors keep both perspectives active.
        active = (5 * 64 + 4, 11 * 64 + 60, 8, 17, 6 * 64 + 48, 7 * 64 + 57)
        features[:, active] = 1
    else:
        # A real starting position exercises all 30 non-king features in each
        # perspective; a smaller synthetic board makes full inference look too fast.
        _, one_position = _board_halfkp_features(model, chess.Board())
        features = one_position.repeat(batch_size, 1)
    sides = torch.arange(batch_size).remainder(2) == 0
    return features, sides


def benchmark(
    model_or_checkpoint: ModelSource,
    *,
    batch_size: int = 1,
    warmup: int = 20,
    iterations: int = 200,
    device: str | torch.device | None = "cpu",
) -> BenchmarkResult:
    """Measure forward-pass speed using one CPU thread by default.

    Batch size 1 approximates an ordinary alpha-beta leaf evaluation; larger
    batches show the benefit available to batched search code.
    """
    if batch_size < 1 or warmup < 0 or iterations < 1:
        raise ValueError("batch_size and iterations must be positive; warmup cannot be negative")
    model, selected_device = _resolve_model(model_or_checkpoint, device)
    features, sides = _benchmark_features(model, batch_size)
    features = features.to(selected_device)
    sides = sides.to(selected_device)
    previous_threads = torch.get_num_threads()
    if selected_device.type == "cpu":
        torch.set_num_threads(1)

    def synchronize() -> None:
        if selected_device.type == "cuda":
            torch.cuda.synchronize(selected_device)

    try:
        model.eval()
        with torch.inference_mode():
            for _ in range(warmup):
                model(features, sides)
            synchronize()
            started = time.perf_counter()
            for _ in range(iterations):
                model(features, sides)
            synchronize()
        elapsed = time.perf_counter() - started
    finally:
        if selected_device.type == "cpu":
            torch.set_num_threads(previous_threads)
    positions = iterations * batch_size
    return BenchmarkResult(
        device=str(selected_device),
        batch_size=batch_size,
        milliseconds_per_batch=elapsed * 1000 / iterations,
        microseconds_per_position=elapsed * 1_000_000 / positions,
        positions_per_second=positions / elapsed,
    )


def benchmark_inference(
    model_or_checkpoint: ModelSource,
    *,
    warmup: int = 100,
    iterations: int = 1000,
    device: str | torch.device | None = "cpu",
) -> BenchmarkResult:
    """Benchmark repeated batch-size-one inference calls."""
    return benchmark(
        model_or_checkpoint,
        batch_size=1,
        warmup=warmup,
        iterations=iterations,
        device=device,
    )


def benchmark_accumulator(
    model_or_checkpoint: ModelSource,
    *,
    warmup: int = 100,
    iterations: int = 1000,
    trials: int = 5,
    device: str | torch.device | None = "cpu",
) -> AccumulatorBenchmarkResult:
    """Benchmark the HalfKP dense head after its accumulators have been maintained."""
    if warmup < 0 or iterations < 1 or trials < 1:
        raise ValueError("warmup cannot be negative; iterations and trials must be positive")
    model, selected_device = _resolve_model(model_or_checkpoint, device)
    if model.representation != "halfkp32":
        raise ValueError("accumulator benchmarking requires a halfkp32 model")
    features, sides = _benchmark_features(model, 1)
    features = features.to(selected_device)
    sides = sides.to(selected_device)
    previous_threads = torch.get_num_threads()
    if selected_device.type == "cpu":
        torch.set_num_threads(1)

    def synchronize() -> None:
        if selected_device.type == "cuda":
            torch.cuda.synchronize(selected_device)

    def measure(call: Any) -> float:
        with torch.inference_mode():
            for _ in range(warmup):
                call()
            samples = []
            for _ in range(trials):
                synchronize()
                started = time.perf_counter()
                for _ in range(iterations):
                    call()
                synchronize()
                samples.append(time.perf_counter() - started)
        return statistics.median(samples)

    try:
        model.eval()
        with torch.inference_mode():
            white, black = model._halfkp_accumulators(features)
            first = torch.where(sides[:, None], white, black)
            second = torch.where(sides[:, None], black, white)
            oriented = torch.cat((first, second), dim=1)

        def full_prediction() -> torch.Tensor:
            return model(features, sides)

        def accumulated_prediction() -> torch.Tensor:
            return (
                model.head(model.input_activation(oriented)).squeeze(-1)
                * model.output_scale_cp
            )

        full_seconds = measure(full_prediction)
        accumulated_seconds = measure(accumulated_prediction)
    finally:
        if selected_device.type == "cpu":
            torch.set_num_threads(previous_threads)

    full_us = full_seconds * 1_000_000 / iterations
    accumulated_us = accumulated_seconds * 1_000_000 / iterations
    parameter_bytes = sum(
        parameter.numel() * parameter.element_size() for parameter in model.parameters()
    )
    accumulator_bytes = 2 * model.architecture.widths[0] * 4
    return AccumulatorBenchmarkResult(
        device=str(selected_device),
        iterations=iterations,
        parameter_mib=parameter_bytes / (1024**2),
        accumulator_kib=accumulator_bytes / 1024,
        full_microseconds=full_us,
        accumulated_microseconds=accumulated_us,
        accumulated_positions_per_second=1_000_000 / accumulated_us,
        speedup=full_us / accumulated_us,
    )


_BENCHMARK_GAME_MOVES = (
    "e2e4",
    "e7e5",
    "g1f3",
    "b8c6",
    "f1b5",
    "a7a6",
    "b5c6",
    "d7c6",
    "e1g1",
    "g8f6",
    "d2d4",
    "e5d4",
    "d1d4",
    "f8e7",
    "f1e1",
    "e8g8",
    "b1c3",
    "c6c5",
    "d4d3",
    "h7h6",
)


def _normalise_halfkp_index(model: NNUE, raw: int, *, black: bool) -> int:
    king, remainder = divmod(raw, 640)
    piece_plane, piece_square = divmod(remainder, 64)
    if black:
        king ^= 56
    if model.mirror_halfkp:
        if king % 8 < 4:
            king ^= 7
            piece_square ^= 7
        king = (king // 8) * 4 + king % 8 - 4
    return king * 640 + piece_plane * 64 + piece_square


def _board_halfkp_features(
    model: NNUE, board: chess.Board
) -> tuple[tuple[frozenset[int], frozenset[int]], torch.Tensor]:
    white_king = board.king(chess.WHITE)
    black_king = board.king(chess.BLACK)
    if white_king is None or black_king is None:
        raise ValueError("benchmark positions must contain both kings")

    raw_white: list[int] = []
    raw_black: list[int] = []
    for square, piece in board.piece_map().items():
        if piece.piece_type == chess.KING:
            continue
        colour = 0 if piece.color == chess.WHITE else 1
        piece_type = piece.piece_type - 1
        raw_white.append(white_king * 640 + (colour * 5 + piece_type) * 64 + square)
        raw_black.append(
            black_king * 640
            + ((colour ^ 1) * 5 + piece_type) * 64
            + (square ^ 56)
        )

    active_white = frozenset(
        _normalise_halfkp_index(model, index, black=False) for index in raw_white
    )
    active_black = frozenset(
        _normalise_halfkp_index(model, index, black=True) for index in raw_black
    )
    features = torch.full((1, HALFKP_SLOTS), int(PADDING_INDEX), dtype=torch.uint32)
    count = len(raw_white)
    features[0, :count] = torch.tensor(raw_white, dtype=torch.uint32)
    features[0, count : 2 * count] = torch.tensor(
        [HALFKP_FEATURES_PER_PERSPECTIVE + index for index in raw_black],
        dtype=torch.uint32,
    )
    return (active_white, active_black), features


class _GameAccumulator:
    def __init__(self, model: NNUE, board: chess.Board) -> None:
        self.model = model
        self.board = board.copy(stack=False)
        self.active, _ = _board_halfkp_features(model, self.board)
        self.white = self._build(self.active[0])
        self.black = self._build(self.active[1])

    @property
    def weight(self) -> torch.Tensor:
        return cast(nn.EmbeddingBag, self.model.feature_transformer).weight

    def _rows(self, indices: frozenset[int]) -> torch.Tensor:
        selected = torch.tensor(sorted(indices), dtype=torch.long, device=self.weight.device)
        return self.weight.index_select(0, selected).sum(dim=0)

    def _build(self, active: frozenset[int]) -> torch.Tensor:
        bias = cast(torch.Tensor, self.model.feature_bias)
        return bias + self._rows(active)

    def _update(
        self,
        accumulator: torch.Tensor,
        old: frozenset[int],
        new: frozenset[int],
    ) -> torch.Tensor:
        removed = old - new
        added = new - old
        if removed:
            accumulator = accumulator - self._rows(removed)
        if added:
            accumulator = accumulator + self._rows(added)
        return accumulator

    def push(self, move: chess.Move) -> None:
        old_kings = (self.board.king(chess.WHITE), self.board.king(chess.BLACK))
        self.board.push(move)
        new_active, _ = _board_halfkp_features(self.model, self.board)
        new_kings = (self.board.king(chess.WHITE), self.board.king(chess.BLACK))
        self.white = (
            self._build(new_active[0])
            if old_kings[0] != new_kings[0]
            else self._update(self.white, self.active[0], new_active[0])
        )
        self.black = (
            self._build(new_active[1])
            if old_kings[1] != new_kings[1]
            else self._update(self.black, self.active[1], new_active[1])
        )
        self.active = new_active

    def evaluate(self) -> torch.Tensor:
        first, second = (
            (self.white, self.black) if self.board.turn else (self.black, self.white)
        )
        oriented = torch.cat((first, second)).unsqueeze(0)
        return (
            self.model.head(self.model.input_activation(oriented)).squeeze()
            * self.model.output_scale_cp
        )


def _game_moves(moves: Sequence[str]) -> tuple[chess.Move, ...]:
    board = chess.Board()
    parsed = []
    for uci in moves:
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            raise ValueError(f"illegal benchmark move {uci!r} after {board.fen()}")
        parsed.append(move)
        board.push(move)
    if not parsed:
        raise ValueError("moves cannot be empty")
    return tuple(parsed)


def benchmark_game(
    model_or_checkpoint: ModelSource,
    *,
    moves: Sequence[str] = _BENCHMARK_GAME_MOVES,
    warmup_games: int = 2,
    games: int = 20,
    trials: int = 5,
    device: str | torch.device | None = "cpu",
) -> GameBenchmarkResult:
    """Benchmark legal moves, accumulator updates, and one inference per move."""
    if warmup_games < 0 or games < 1 or trials < 1:
        raise ValueError("warmup_games cannot be negative; games and trials must be positive")
    model, selected_device = _resolve_model(model_or_checkpoint, device)
    if model.representation != "halfkp32":
        raise ValueError("game accumulator benchmarking requires a halfkp32 model")
    parsed_moves = _game_moves(moves)
    previous_threads = torch.get_num_threads()
    if selected_device.type == "cpu":
        torch.set_num_threads(1)

    def synchronize() -> None:
        if selected_device.type == "cuda":
            torch.cuda.synchronize(selected_device)

    def play_once() -> tuple[float, float, float]:
        state = _GameAccumulator(model, chess.Board())
        update_seconds = 0.0
        inference_seconds = 0.0
        synchronize()
        total_started = time.perf_counter()
        for move in parsed_moves:
            started = time.perf_counter()
            state.push(move)
            synchronize()
            update_seconds += time.perf_counter() - started

            started = time.perf_counter()
            state.evaluate()
            synchronize()
            inference_seconds += time.perf_counter() - started
        total_seconds = time.perf_counter() - total_started
        return update_seconds, inference_seconds, total_seconds

    try:
        model.eval()
        max_error_cp = 0.0
        with torch.inference_mode():
            state = _GameAccumulator(model, chess.Board())
            for move in parsed_moves:
                state.push(move)
                incremental = state.evaluate()
                _, features = _board_halfkp_features(model, state.board)
                side = torch.tensor([state.board.turn], device=selected_device)
                full = model(features.to(selected_device), side).squeeze()
                max_error_cp = max(max_error_cp, abs((incremental - full).item()))
            if max_error_cp > 0.05:
                raise RuntimeError(
                    f"incremental accumulator differs from full inference by {max_error_cp:.3f} cp"
                )

            for _ in range(warmup_games):
                play_once()
            samples = []
            for _ in range(trials):
                update = inference = total = 0.0
                for _ in range(games):
                    one_update, one_inference, one_total = play_once()
                    update += one_update
                    inference += one_inference
                    total += one_total
                samples.append((update, inference, total))
    finally:
        if selected_device.type == "cpu":
            torch.set_num_threads(previous_threads)

    positions = games * len(parsed_moves)
    update_us = statistics.median(sample[0] for sample in samples) * 1_000_000 / positions
    inference_us = statistics.median(sample[1] for sample in samples) * 1_000_000 / positions
    total_us = statistics.median(sample[2] for sample in samples) * 1_000_000 / positions
    return GameBenchmarkResult(
        device=str(selected_device),
        games=games,
        plies_per_game=len(parsed_moves),
        positions=positions,
        update_microseconds=update_us,
        inference_microseconds=inference_us,
        total_microseconds=total_us,
        positions_per_second=1_000_000 / total_us,
        max_error_cp=max_error_cp,
    )


def _optimizers(
    model: NNUE,
    *,
    learning_rate: float,
    weight_decay: float,
) -> tuple[list[torch.optim.Optimizer], list[nn.Parameter]]:
    if model.representation == "halfkp32" and model.sparse_gradients:
        transformer = cast(nn.EmbeddingBag, model.feature_transformer)
        sparse = torch.optim.SparseAdam([transformer.weight], lr=learning_rate)
        sparse_id = id(transformer.weight)
        dense_parameters = [
            parameter for parameter in model.parameters() if id(parameter) != sparse_id
        ]
        dense = torch.optim.AdamW(
            dense_parameters, lr=learning_rate, weight_decay=weight_decay
        )
        return [sparse, dense], dense_parameters
    parameters = list(model.parameters())
    optimizer = torch.optim.AdamW(
        parameters,
        lr=learning_rate,
        weight_decay=weight_decay,
        fused=next(model.parameters()).device.type == "cuda",
    )
    return [optimizer], parameters


def _checkpoint_name(
    representation: Representation,
    architecture: Architecture,
    mirror_halfkp: bool,
    objective: Objective,
    run_name: str | None,
) -> str:
    if run_name is not None:
        cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_name).strip("_.")
        if not cleaned:
            raise ValueError("run_name must contain a letter or number")
        return f"{cleaned}.pt"
    feature_name = "halfkp_hm" if representation == "halfkp32" and mirror_halfkp else representation
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"{feature_name}_{architecture.name}_{objective}_{timestamp}.pt"


def train(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    *,
    representation: Representation = "halfkp32",
    architecture: str | Sequence[int] | Architecture = DEFAULT_ARCHITECTURE,
    objective: Objective = "wdl",
    epochs: int = 10,
    batch_size: int = 4096,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    workers: int = 0,
    device: str | torch.device | None = "auto",
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    mirror_halfkp: bool = True,
    sparse_gradients: bool | None = None,
    seed: int = 42,
    patience: int | None = None,
    gradient_clip: float = 1.0,
    max_train_batches: int | None = None,
    max_validation_batches: int | None = None,
    test: bool = True,
    allow_incomplete_data: bool = False,
    run_name: str | None = None,
    resume_from: str | Path | None = None,
    verbose: bool = True,
) -> TrainingResult:
    """Train, select the best validation checkpoint, optionally test, and return one result.

    With no arguments this finds ``evaluation/data``, trains the recommended
    compact HalfKP_hm network, saves a self-describing checkpoint beneath
    ``evaluation/models``, and evaluates the best checkpoint on the test set.
    Set ``representation="piece768"`` to train from the synchronized dense
    column instead.
    """
    representation = _representation(representation)
    if objective not in {"wdl", "rmse", "mae"}:
        raise ValueError("objective must be 'wdl', 'rmse', or 'mae'")
    parsed = parse_architecture(architecture)
    if epochs < 1 or batch_size < 1 or workers < 0:
        raise ValueError("epochs and batch_size must be positive; workers cannot be negative")
    if learning_rate <= 0 or weight_decay < 0:
        raise ValueError("learning_rate must be positive and weight_decay cannot be negative")
    if patience is not None and patience < 1:
        raise ValueError("patience must be positive or None")
    if gradient_clip <= 0:
        raise ValueError("gradient_clip must be positive")
    if max_train_batches is not None and max_train_batches < 1:
        raise ValueError("max_train_batches must be positive or None")
    if max_validation_batches is not None and max_validation_batches < 1:
        raise ValueError("max_validation_batches must be positive or None")

    root = Path(data_dir).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    selected_device = _device(device)
    if sparse_gradients is None:
        sparse_gradients = selected_device.type == "cpu" and representation == "halfkp32"
    _seed_everything(seed)

    dataset = inspect_data(root)
    if not dataset.metadata_found and not allow_incomplete_data:
        raise RuntimeError(
            "dataset metadata.json is missing, so preparation may still be running; "
            "wait for it to finish or pass allow_incomplete_data=True intentionally"
        )
    if verbose:
        print(dataset, flush=True)
        print(
            f"Training {representation} {parsed.name} for {objective.upper()} on "
            f"{selected_device} "
            f"(batch={batch_size:,}, workers={workers}, seed={seed})",
            flush=True,
        )

    train_loader = _loader(
        root,
        "train",
        representation,
        batch_size,
        workers,
        shuffle=True,
        seed=seed,
        pin_memory=selected_device.type == "cuda",
    )
    validation_loader = _loader(
        root,
        "validation",
        representation,
        batch_size,
        workers,
        shuffle=False,
        seed=seed,
        pin_memory=selected_device.type == "cuda",
    )
    if resume_from is None:
        model = NNUE(
            representation,
            parsed,
            mirror_halfkp=mirror_halfkp,
            sparse_gradients=sparse_gradients,
        ).to(selected_device)
    else:
        model = load_model(
            resume_from,
            selected_device,
            sparse_gradients=sparse_gradients,
        )
        if model.representation != representation:
            raise ValueError("resume checkpoint representation does not match")
        if model.architecture != parsed:
            raise ValueError("resume checkpoint architecture does not match")
        if model.mirror_halfkp != mirror_halfkp:
            raise ValueError("resume checkpoint mirror_halfkp setting does not match")
    optimizers, clipped_parameters = _optimizers(
        model, learning_rate=learning_rate, weight_decay=weight_decay
    )
    use_amp = selected_device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    destination.mkdir(parents=True, exist_ok=True)
    model_path = destination / _checkpoint_name(
        representation, parsed, mirror_halfkp, objective, run_name
    )
    if run_name is not None and model_path.exists():
        raise FileExistsError(f"refusing to overwrite existing run {model_path}")

    history: list[EpochMetrics] = []
    best_metric = math.inf
    best_wdl_loss = math.inf
    best_epoch = 0
    epochs_without_improvement = 0
    if resume_from is not None:
        baseline = _evaluate_loader(
            model,
            validation_loader,
            selected_device,
            split="validation",
            max_batches=max_validation_batches,
        )
        best_metric = _validation_objective(baseline, objective)
        best_wdl_loss = baseline.wdl_loss
        save_model(
            model,
            model_path,
            metadata={
                "best_epoch": 0,
                "objective": objective,
                "best_validation_metric": best_metric,
                "best_validation_loss": best_wdl_loss,
                "data_dir": str(root),
                "seed": seed,
                "resume_from": str(Path(resume_from).expanduser().resolve()),
            },
        )
        if verbose:
            print(
                f"Resumed baseline: {objective}={best_metric:.6f}, "
                f"WDL={best_wdl_loss:.6f}",
                flush=True,
            )
    training_started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        total_loss = 0.0
        seen = 0
        available_batches = len(train_loader)
        total_batches = (
            available_batches
            if max_train_batches is None
            else min(available_batches, max_train_batches)
        )
        report_every = max(1, total_batches // 20)
        for batch_number, ((features, white_to_move), targets) in enumerate(
            train_loader, start=1
        ):
            if batch_number > total_batches:
                break
            features = features.to(selected_device, non_blocking=True)
            white_to_move = white_to_move.to(selected_device, non_blocking=True)
            targets = targets.to(selected_device, dtype=torch.float32, non_blocking=True)
            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=selected_device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                predictions = model(features, white_to_move)
                loss = objective_loss(predictions.float(), targets.float(), objective)
            scaler.scale(loss).backward()
            for optimizer in optimizers:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(clipped_parameters, gradient_clip)
            for optimizer in optimizers:
                scaler.step(optimizer)
            scaler.update()
            total_loss += loss.detach().item() * len(targets)
            seen += len(targets)

            if verbose and (
                batch_number == 1
                or batch_number == total_batches
                or batch_number % report_every == 0
            ):
                elapsed = time.perf_counter() - epoch_started
                rate = seen / elapsed if elapsed else 0.0
                print(
                    f"epoch {epoch}/{epochs}  batch {batch_number}/{total_batches}  "
                    f"loss={total_loss / seen:.6f}  {rate:,.0f} pos/s",
                    flush=True,
                )

        if seen == 0:
            raise ValueError("training loader produced no positions")
        validation = _evaluate_loader(
            model,
            validation_loader,
            selected_device,
            split="validation",
            max_batches=max_validation_batches,
        )
        epoch_seconds = time.perf_counter() - epoch_started
        validation_objective = _validation_objective(validation, objective)
        history.append(
            EpochMetrics(
                epoch=epoch,
                train_loss=total_loss / seen,
                validation_rmse_cp=validation.rmse_cp,
                validation_loss=validation.wdl_loss,
                validation_mae_cp=validation.mae_cp,
                seconds=epoch_seconds,
                learning_rate=learning_rate,
                validation_objective=validation_objective,
            )
        )
        improved = validation_objective < best_metric
        if improved:
            best_metric = validation_objective
            best_wdl_loss = validation.wdl_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            save_model(
                model,
                model_path,
                metadata={
                    "best_epoch": best_epoch,
                    "objective": objective,
                    "best_validation_metric": best_metric,
                    "best_validation_loss": best_wdl_loss,
                    "data_dir": str(root),
                    "seed": seed,
                    "resume_from": (
                        None
                        if resume_from is None
                        else str(Path(resume_from).expanduser().resolve())
                    ),
                },
            )
        else:
            epochs_without_improvement += 1
        if verbose:
            marker = "  saved best" if improved else ""
            print(
                f"epoch {epoch}/{epochs}  {objective}={validation_objective:.6f}  "
                f"WDL={validation.wdl_loss:.6f}  "
                f"RMSE={validation.rmse_cp:.1f} cp  MAE={validation.mae_cp:.1f} cp"
                f"{marker}",
                flush=True,
            )
        if patience is not None and epochs_without_improvement >= patience:
            if verbose:
                print(f"Early stopping after {patience} unimproved epochs.", flush=True)
            break

    training_seconds = time.perf_counter() - training_started
    best_model = load_model(model_path, selected_device)
    test_metrics: EvaluationMetrics | None = None
    if test:
        test_metrics = evaluate(
            best_model,
            root,
            split="test",
            batch_size=batch_size,
            workers=workers,
            device=selected_device,
        )
    result = TrainingResult(
        architecture=parsed.name,
        representation=representation,
        epochs=history,
        test_rmse_cp=math.nan if test_metrics is None else test_metrics.rmse_cp,
        model_path=str(model_path),
        data_dir=str(root),
        objective=objective,
        best_epoch=best_epoch,
        best_validation_loss=best_wdl_loss,
        best_validation_metric=best_metric,
        test_metrics=test_metrics,
        parameters=best_model.parameter_count,
        model_size_mib=best_model.size_mib_float32,
        training_seconds=training_seconds,
        device=str(selected_device),
    )
    result_path = model_path.with_suffix(".json")
    result_path.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    if verbose:
        test_text = (
            "test skipped"
            if test_metrics is None
            else f"test RMSE={test_metrics.rmse_cp:.1f} cp, MAE={test_metrics.mae_cp:.1f} cp"
        )
        print(
            f"Best epoch {best_epoch} by {objective}={best_metric:.6f}; {test_text}; "
            f"saved {model_path} "
            f"({best_model.size_mib_float32:.1f} MiB float32)",
            flush=True,
        )
    return result


def train_model(
    root: str | Path,
    representation: Representation,
    architecture: str | list[int] | tuple[int, ...] = DEFAULT_ARCHITECTURE,
    epochs: int = 10,
    batch_size: int = 4096,
    learning_rate: float = 1e-3,
    workers: int = 0,
    device: str | torch.device | None = None,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    objective: Objective = "wdl",
) -> TrainingResult:
    """Backward-compatible wrapper for notebooks that used ``train_model``."""
    return train(
        data_dir=root,
        representation=representation,
        architecture=architecture,
        objective=objective,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        workers=workers,
        device=device,
        output_dir=output_dir,
    )


def plot_history(result: TrainingResult, output_path: str | Path | None = None) -> Any:
    """Plot loss and validation error; matplotlib is imported only when requested."""
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "plot_history requires matplotlib; install it in the training environment"
        ) from error

    epochs = [item.epoch for item in result.epochs]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, [item.train_loss for item in result.epochs], label="train")
    axes[0].plot(
        epochs,
        [item.validation_loss for item in result.epochs],
        label="validation",
    )
    axes[0].set(
        title=f"{result.objective.upper()} training objective",
        xlabel="Epoch",
        ylabel="Loss",
    )
    axes[0].legend()
    axes[1].plot(
        epochs,
        [item.validation_rmse_cp for item in result.epochs],
        marker="o",
    )
    axes[1].set(title="Validation error", xlabel="Epoch", ylabel="RMSE (cp)")
    figure.suptitle(f"{result.representation} {result.architecture}")
    figure.tight_layout()
    if output_path is not None:
        destination = Path(output_path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=150)
    return figure


__all__ = [
    "NNUE",
    "AccumulatorBenchmarkResult",
    "Architecture",
    "BenchmarkResult",
    "DatasetInfo",
    "EpochMetrics",
    "EvaluationMetrics",
    "GameBenchmarkResult",
    "ModelInfo",
    "Objective",
    "Representation",
    "ShardDataset",
    "TrainingResult",
    "benchmark",
    "benchmark_accumulator",
    "benchmark_game",
    "benchmark_inference",
    "evaluate",
    "evaluate_model",
    "inspect_data",
    "load_model",
    "model_info",
    "objective_loss",
    "parse_architecture",
    "plot_history",
    "predict",
    "recommended_workers",
    "save_model",
    "train",
    "train_model",
    "wdl_loss",
]


if __name__ == "__main__":
    print(inspect_data())
    print("Import train() to start a run; no-argument train() uses the recommended defaults.")
