"""Fast, resumable preparation of synchronized NNUE training shards.

The normal entry point is a single function::

    from evaluation.prepare_datasets import prepare

    result = prepare(examples=10_000_000)

Each source position is decoded once and both ``piece768`` and legacy-compatible
``halfkp32`` features are written to the same row.  Work is bounded in memory,
shards are replaced atomically, and progress is committed only after the shard
writes for a batch have completed.  An interrupted run can therefore resume
without trusting a partial file.

For a dataset whose shards were completed by the former script but whose final
metadata was never published, use ``finalize_existing()``.  It validates and
describes the existing shards; it never rewrites them.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import sys
import threading
import time
import zipfile
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

try:  # Optional: the pure-Python encoder remains fully supported.
    import numba as _numba
except ImportError:  # pragma: no cover - depends on the local training machine
    _numba = None


Split = Literal["train", "validation", "test"]
Encoder = Literal["auto", "numba", "python"]

MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = MODULE_DIR / "data"
SOURCE_PATH = DEFAULT_DATA_DIR / "lichess_db_eval.jsonl (1).zst"
OUTPUT_ROOT = DEFAULT_DATA_DIR

SPLITS: tuple[Split, ...] = ("train", "validation", "test")
FEN_BYTES = 100
PIECE_FEATURES = 768
HALFKP_FEATURES_PER_PERSPECTIVE = 64 * 10 * 64
HALFKP_FEATURES = 2 * HALFKP_FEATURES_PER_PERSPECTIVE
HALFKP_SLOTS = 64
PADDING_INDEX = np.uint32(0xFFFFFFFF)
MATE_SCORE_CP = 10_000.0
SCHEMA_VERSION = 2
PIPELINE_VERSION = 2

_FEN_RE = re.compile(rb'"fen"\s*:\s*"([^"\\]{1,100})"')
_PVS_RE = re.compile(rb'"pvs"\s*:\s*\[\s*\{')
_SCORE_RE = re.compile(
    rb'"(cp|mate)"\s*:\s*(-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)'
)


@dataclass(frozen=True)
class EncodedBatch:
    """Both synchronized feature representations for a sequence of FENs."""

    fen: np.ndarray
    halfkp32: np.ndarray
    piece768: np.ndarray
    white_to_move: np.ndarray
    valid: np.ndarray


@dataclass(frozen=True)
class PreparationResult:
    """Compact result returned by :func:`prepare_dataset` and finalization."""

    output_dir: str
    selected: int
    source_records: int
    split_counts: dict[str, int]
    shard_counts: dict[str, int]
    skipped: dict[str, int]
    elapsed_seconds: float
    complete: bool
    resumed: bool
    metadata_path: str

    @property
    def positions(self) -> int:
        return self.selected

    def __str__(self) -> str:
        splits = ", ".join(f"{name}={count:,}" for name, count in self.split_counts.items())
        return f"PreparedDataset(total={self.selected:,}; {splits}; {self.elapsed_seconds:.1f}s)"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _piece_details(code: int) -> tuple[int, int]:
    """Return ``(colour, piece_type)`` or ``(-1, -1)`` for a non-piece byte."""
    if code == 80:  # P
        return 0, 0
    if code == 78:  # N
        return 0, 1
    if code == 66:  # B
        return 0, 2
    if code == 82:  # R
        return 0, 3
    if code == 81:  # Q
        return 0, 4
    if code == 75:  # K
        return 0, 5
    if code == 112:  # p
        return 1, 0
    if code == 110:  # n
        return 1, 1
    if code == 98:  # b
        return 1, 2
    if code == 114:  # r
        return 1, 3
    if code == 113:  # q
        return 1, 4
    if code == 107:  # k
        return 1, 5
    return -1, -1


def _encode_matrix_impl(
    matrix: np.ndarray, lengths: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Inner encoder kept Numba-friendly (integer loops and fixed arrays only)."""
    rows = matrix.shape[0]
    piece768 = np.zeros((rows, PIECE_FEATURES), dtype=np.uint8)
    halfkp32 = np.full((rows, HALFKP_SLOTS), PADDING_INDEX, dtype=np.uint32)
    white_to_move = np.zeros(rows, dtype=np.bool_)
    valid = np.zeros(rows, dtype=np.bool_)
    piece_squares = np.zeros((rows, 32), dtype=np.uint8)
    piece_types = np.zeros((rows, 32), dtype=np.uint8)
    piece_colours = np.zeros((rows, 32), dtype=np.uint8)
    piece_counts = np.zeros(rows, dtype=np.uint8)
    king_squares = np.zeros((rows, 2), dtype=np.int16)
    king_seen = np.zeros((rows, 2), dtype=np.uint8)

    for row in range(rows):
        length = int(lengths[row])
        if length < 3 or length > FEN_BYTES:
            continue
        rank = 7
        file_index = 0
        cursor = 0
        ok = True
        while cursor < length and matrix[row, cursor] != 32:
            code = int(matrix[row, cursor])
            cursor += 1
            if code == 47:  # /
                if file_index != 8 or rank == 0:
                    ok = False
                    break
                rank -= 1
                file_index = 0
                continue
            if 49 <= code <= 56:  # 1..8
                file_index += code - 48
                if file_index > 8:
                    ok = False
                    break
                continue
            colour, piece_type = _piece_details(code)
            if colour < 0 or file_index >= 8:
                ok = False
                break
            square = rank * 8 + file_index
            file_index += 1
            piece768[row, (colour * 6 + piece_type) * 64 + square] = 1
            if piece_type == 5:
                if king_seen[row, colour] != 0:
                    ok = False
                    break
                king_seen[row, colour] = 1
                king_squares[row, colour] = square
            else:
                piece_count = int(piece_counts[row])
                if piece_count >= 32:
                    ok = False
                    break
                piece_squares[row, piece_count] = square
                piece_types[row, piece_count] = piece_type
                piece_colours[row, piece_count] = colour
                piece_counts[row] = piece_count + 1

        if not ok or rank != 0 or file_index != 8 or cursor >= length:
            continue
        if king_seen[row, 0] != 1 or king_seen[row, 1] != 1:
            continue
        cursor += 1  # board/side separator
        if cursor >= length:
            continue
        side = int(matrix[row, cursor])
        if side == 119:  # w
            white_to_move[row] = True
        elif side == 98:  # b
            white_to_move[row] = False
        else:
            continue
        if cursor + 1 < length and matrix[row, cursor + 1] != 32:
            continue

        piece_count = int(piece_counts[row])
        for piece_index in range(piece_count):
            square = int(piece_squares[row, piece_index])
            piece_type = int(piece_types[row, piece_index])
            colour = int(piece_colours[row, piece_index])
            white_feature = (
                int(king_squares[row, 0]) * 640
                + (colour * 5 + piece_type) * 64
                + square
            )
            # Keep the legacy on-disk convention: Black's piece square is
            # rank-flipped while its king square remains raw. training.py
            # normalizes that quirk while loading.
            black_feature = (
                HALFKP_FEATURES_PER_PERSPECTIVE
                + int(king_squares[row, 1]) * 640
                + ((colour ^ 1) * 5 + piece_type) * 64
                + (square ^ 56)
            )
            halfkp32[row, piece_index] = white_feature
            halfkp32[row, piece_count + piece_index] = black_feature
        valid[row] = True

    return halfkp32, piece768, white_to_move, valid


if _numba is not None:  # pragma: no branch - chosen once at import
    _encode_matrix_numba = _numba.njit(cache=True, nogil=True)(_encode_matrix_impl)
else:  # pragma: no cover - name is present for clear error handling
    _encode_matrix_numba = None


def _normalise_fens(
    fens: Sequence[str | bytes | np.str_ | np.bytes_],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw_values: list[bytes] = []
    lengths = np.zeros(len(fens), dtype=np.int16)
    forced_invalid = np.zeros(len(fens), dtype=np.bool_)
    for index, value in enumerate(fens):
        try:
            if isinstance(value, (bytes, np.bytes_)):
                raw = bytes(value).rstrip(b"\0")
            else:
                raw = str(value).encode("ascii")
        except (UnicodeEncodeError, ValueError):
            raw = b""
            forced_invalid[index] = True
        if len(raw) > FEN_BYTES:
            raw = b""
            forced_invalid[index] = True
        raw_values.append(raw)
        lengths[index] = len(raw)
    fixed = np.asarray(raw_values, dtype=f"S{FEN_BYTES}")
    return fixed, lengths, forced_invalid


def encode_fens(
    fens: Sequence[str | bytes | np.str_ | np.bytes_], *, encoder: Encoder = "auto"
) -> EncodedBatch:
    """Encode FENs once into synchronized Piece768 and HalfKP arrays.

    Invalid rows are retained in the returned arrays with ``valid=False`` so a
    caller can preserve exact input ordering.  Dataset preparation filters them.
    ``encoder='auto'`` uses Numba when installed and the equivalent Python loop
    otherwise.
    """
    if encoder not in {"auto", "numba", "python"}:
        raise ValueError("encoder must be 'auto', 'numba', or 'python'")
    fixed, lengths, forced_invalid = _normalise_fens(fens)
    matrix = fixed.view(np.uint8).reshape(len(fixed), FEN_BYTES)
    use_numba = encoder == "numba" or (encoder == "auto" and _encode_matrix_numba is not None)
    if use_numba:
        if _encode_matrix_numba is None:
            raise RuntimeError("encoder='numba' requested, but numba is not installed")
        halfkp32, piece768, white_to_move, valid = _encode_matrix_numba(matrix, lengths)
    else:
        halfkp32, piece768, white_to_move, valid = _encode_matrix_impl(matrix, lengths)
    valid &= ~forced_invalid
    return EncodedBatch(fixed, halfkp32, piece768, white_to_move, valid)


def split_for_fen(fen: str | bytes | np.str_ | np.bytes_) -> Split:
    """Assign represented inputs deterministically to an 80/10/10 split.

    Only board placement and side to move are hashed.  Castling and en-passant
    fields are deliberately excluded because neither stored representation
    encodes them; identical model inputs therefore cannot leak across splits.
    """
    if isinstance(fen, (bytes, np.bytes_)):
        raw = bytes(fen).rstrip(b"\0")
    else:
        try:
            raw = str(fen).encode("ascii")
        except UnicodeEncodeError as error:
            raise ValueError("FEN must be ASCII") from error
    fields = raw.split()
    if len(fields) < 2 or fields[1] not in {b"w", b"b"}:
        raise ValueError("FEN must contain board placement and side to move")
    key = fields[0] + b" " + fields[1]
    bucket = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "little") % 10
    if bucket < 8:
        return "train"
    if bucket == 8:
        return "test"
    return "validation"


def _extract_record(line: bytes) -> tuple[bytes, float, bool] | None:
    fen_match = _FEN_RE.search(line)
    if fen_match is None:
        return None
    fen = fen_match.group(1)
    try:
        fen.decode("ascii")
    except UnicodeDecodeError:
        return None
    pvs_match = _PVS_RE.search(line)
    if pvs_match is None:
        return None
    first_pv_end = line.find(b"}", pvs_match.end())
    if first_pv_end < 0:
        return None
    score_match = _SCORE_RE.search(line, pvs_match.end(), first_pv_end + 1)
    if score_match is None:
        return None
    try:
        value = float(score_match.group(2))
    except ValueError:
        return None
    if not math.isfinite(value):
        return None
    is_mate = score_match.group(1) == b"mate"
    if is_mate:
        if value > 0:
            value = MATE_SCORE_CP
        elif value < 0:
            value = -MATE_SCORE_CP
        else:
            value = 0.0
    return fen, value, is_mate


def _source_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{threading.get_ident()}")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_text(path: Path, value: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{threading.get_ident()}")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_npz_atomic(
    path: Path, arrays: Mapping[str, np.ndarray], compression_level: int
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{threading.get_ident()}")
    compression = zipfile.ZIP_STORED if compression_level == 0 else zipfile.ZIP_DEFLATED
    options: dict[str, Any] = {"compression": compression, "allowZip64": True}
    if compression != zipfile.ZIP_STORED:
        options["compresslevel"] = compression_level
    try:
        with temporary.open("w+b") as raw_stream:
            with zipfile.ZipFile(raw_stream, mode="w", **options) as archive:
                for name, array in arrays.items():
                    with archive.open(f"{name}.npy", mode="w", force_zip64=True) as member:
                        np.lib.format.write_array(member, np.ascontiguousarray(array), allow_pickle=False)
            raw_stream.flush()
            os.fsync(raw_stream.fileno())
        os.replace(temporary, path)
        return path.stat().st_size
    finally:
        temporary.unlink(missing_ok=True)


def _acquire_lock(root: Path) -> tuple[int, Path]:
    root.mkdir(parents=True, exist_ok=True)
    path = root / ".prepare.lock"
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise RuntimeError(
            f"{path} exists; another preparation may be running. "
            "If no process owns it, remove only that stale lock file."
        ) from error
    os.write(descriptor, f"pid={os.getpid()} started={_utc_now()}\n".encode("ascii"))
    return descriptor, path


def _release_lock(descriptor: int, path: Path) -> None:
    os.close(descriptor)
    path.unlink(missing_ok=True)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return cast(dict[str, Any], value)


def _result_from_metadata(path: Path, *, resumed: bool) -> PreparationResult:
    metadata = _load_json(path)
    return PreparationResult(
        output_dir=str(path.parent),
        selected=int(metadata["total_positions"]),
        source_records=int(metadata.get("source_records", 0)),
        split_counts={key: int(value) for key, value in metadata["split_counts"].items()},
        shard_counts={key: int(value) for key, value in metadata["shard_counts"].items()},
        skipped={key: int(value) for key, value in metadata.get("skipped", {}).items()},
        elapsed_seconds=float(metadata.get("elapsed_seconds", 0.0)),
        complete=bool(metadata.get("complete", False)),
        resumed=resumed,
        metadata_path=str(path),
    )


def _open_zstd_lines(source: Path) -> tuple[Any, io.BufferedReader, Any]:
    try:
        import zstandard
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "zstandard is required to read .zst input; install it on the training machine"
        ) from error
    compressed = source.open("rb")
    reader = zstandard.ZstdDecompressor().stream_reader(compressed, read_across_frames=True)
    buffered = io.BufferedReader(reader, buffer_size=4 * 1024 * 1024)
    return compressed, buffered, reader


def _snapshot_state(
    *,
    next_batch: int,
    source_records: int,
    selected: int,
    split_counts: Mapping[str, int],
    shard_counts: Mapping[str, int],
    skipped: Mapping[str, int],
    started_at: str,
    elapsed_seconds: float,
) -> dict[str, Any]:
    return {
        "version": PIPELINE_VERSION,
        "next_batch": next_batch,
        "source_records": source_records,
        "selected": selected,
        "split_counts": dict(split_counts),
        "shard_counts": dict(shard_counts),
        "skipped": dict(skipped),
        "started_at": started_at,
        "updated_at": _utc_now(),
        "elapsed_seconds": elapsed_seconds,
    }


def _metadata(
    root: Path,
    source: Path | None,
    *,
    total: int,
    source_records: int,
    split_counts: Mapping[str, int],
    shard_counts: Mapping[str, int],
    skipped: Mapping[str, int],
    elapsed_seconds: float,
    pipeline: str,
    columns: Sequence[str],
    split_policy: str,
) -> dict[str, Any]:
    source_info: dict[str, Any] | None = None
    if source is not None:
        source_info = _source_fingerprint(source) if source.exists() else {"path": str(source)}
    return {
        "complete": True,
        "schema_version": SCHEMA_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "pipeline": pipeline,
        "created_at": _utc_now(),
        "data_dir": str(root),
        "source": source_info,
        "source_records": source_records,
        "total_positions": total,
        "selected": total,
        "split_counts": dict(split_counts),
        "shard_counts": dict(shard_counts),
        "skipped": dict(skipped),
        "columns": list(columns),
        "dtypes": {
            "fen": f"S{FEN_BYTES}" if "white_to_move" in columns else "unicode/bytes",
            "halfkp32": "uint32[N,64]",
            "piece768": "uint8[N,768]",
            "evaluation_cp": "float32[N]",
            **({"white_to_move": "bool[N]", "is_mate": "bool[N]"} if "white_to_move" in columns else {}),
        },
        "label": {
            "name": "evaluation_cp",
            "point_of_view": "white",
            "source_point_of_view": "side_to_move",
            "mate_mapping_cp": MATE_SCORE_CP,
        },
        "representations": ["halfkp32", "piece768"],
        "split_policy": split_policy,
        "elapsed_seconds": elapsed_seconds,
    }


def _publish_metadata(root: Path, metadata: Mapping[str, Any]) -> Path:
    for split in SPLITS:
        split_metadata = dict(metadata)
        split_metadata["split"] = split
        split_metadata["positions"] = int(cast(Mapping[str, Any], metadata["split_counts"])[split])
        split_metadata["shards"] = int(cast(Mapping[str, Any], metadata["shard_counts"])[split])
        _atomic_json(root / split / "metadata.json", split_metadata)
    metadata_path = root / "metadata.json"
    _atomic_json(metadata_path, metadata)
    _atomic_text(root / "_SUCCESS", json.dumps({"complete": True, "positions": metadata["total_positions"]}) + "\n")
    return metadata_path


def prepare_dataset(
    source: str | Path = SOURCE_PATH,
    output_dir: str | Path = OUTPUT_ROOT,
    *,
    examples: int = 10_000_000,
    batch_size: int = 100_000,
    compression_level: int = 3,
    writer_threads: int = 3,
    max_pending_batches: int = 2,
    resume: bool = True,
    encoder: Encoder = "auto",
    verbose: bool = True,
) -> PreparationResult:
    """Prepare both NNUE representations with one bounded, resumable call.

    The first ``examples`` valid records are selected in source order.  Three
    split shard writes may run concurrently, while ``max_pending_batches``
    bounds how much encoded data can await compression.  Existing untracked
    shards are never erased: use a new directory or ``finalize_existing``.
    """
    if examples < 1:
        raise ValueError("examples must be positive")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not 0 <= compression_level <= 9:
        raise ValueError("compression_level must be between 0 and 9")
    if writer_threads < 1 or max_pending_batches < 1:
        raise ValueError("writer_threads and max_pending_batches must be positive")
    if encoder not in {"auto", "numba", "python"}:
        raise ValueError("encoder must be 'auto', 'numba', or 'python'")

    source_path = Path(source).expanduser().resolve()
    root = Path(output_dir).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    descriptor, lock_path = _acquire_lock(root)
    start_clock = time.perf_counter()
    try:
        state_dir = root / ".prepare"
        config_path = state_dir / "config.json"
        state_path = state_dir / "state.json"
        metadata_path = root / "metadata.json"
        config = {
            "pipeline_version": PIPELINE_VERSION,
            "schema_version": SCHEMA_VERSION,
            "source": _source_fingerprint(source_path),
            "examples": examples,
            "batch_size": batch_size,
            "compression_level": compression_level,
            "split_policy": "blake2b64-board-and-side-v1",
            "fen_bytes": FEN_BYTES,
        }

        if (root / "_SUCCESS").exists() and metadata_path.exists():
            completed = _result_from_metadata(metadata_path, resumed=True)
            if completed.selected == examples:
                return completed
            raise ValueError(
                f"{root} is already complete with {completed.selected:,} rows; "
                "choose another output_dir for a different target"
            )

        existing_shards = any(
            any((root / split).glob("shard_*.npz")) for split in SPLITS
        )
        resumed = config_path.exists() or state_path.exists()
        if resumed:
            if not resume:
                raise ValueError("partial preparation exists and resume=False; choose a new output_dir")
            if not config_path.exists() or not state_path.exists():
                raise ValueError(f"incomplete resume state under {state_dir}; choose a new output_dir")
            previous_config = _load_json(config_path)
            if previous_config != config:
                raise ValueError(
                    "preparation settings or source changed since the partial run; "
                    "resume with the original settings or choose a new output_dir"
                )
            state = _load_json(state_path)
        else:
            if existing_shards:
                raise ValueError(
                    f"{root} already contains shards without resumable state. "
                    "Call finalize_existing(output_dir=...) to preserve them, or choose a new directory."
                )
            for split in SPLITS:
                (root / split).mkdir(parents=True, exist_ok=True)
            state_dir.mkdir(parents=True, exist_ok=True)
            _atomic_json(config_path, config)
            state = _snapshot_state(
                next_batch=0,
                source_records=0,
                selected=0,
                split_counts={split: 0 for split in SPLITS},
                shard_counts={split: 0 for split in SPLITS},
                skipped={"record": 0, "fen": 0},
                started_at=_utc_now(),
                elapsed_seconds=0.0,
            )
            _atomic_json(state_path, state)

        next_batch = int(state["next_batch"])
        source_records = int(state["source_records"])
        selected = int(state["selected"])
        split_counts = {split: int(state["split_counts"][split]) for split in SPLITS}
        shard_counts = {split: int(state["shard_counts"][split]) for split in SPLITS}
        skipped = {"record": 0, "fen": 0}
        skipped.update({key: int(value) for key, value in state.get("skipped", {}).items()})
        started_at = str(state.get("started_at", _utc_now()))
        previous_elapsed = float(state.get("elapsed_seconds", 0.0))

        compressed, lines, zstd_reader = _open_zstd_lines(source_path)
        try:
            for replayed in range(source_records):
                if not lines.readline():
                    raise ValueError(
                        f"source ended while replaying resume position at record {replayed:,}"
                    )
            if verbose and source_records:
                print(f"Resumed after {source_records:,} source records / {selected:,} selected")

            pending: deque[tuple[list[Future[int]], dict[str, Any]]] = deque()

            def commit_oldest() -> None:
                futures, snapshot = pending.popleft()
                for future in futures:
                    future.result()
                _atomic_json(state_path, snapshot)

            with ThreadPoolExecutor(max_workers=writer_threads, thread_name_prefix="npz") as pool:
                exhausted = False
                while selected < examples:
                    while len(pending) >= max_pending_batches:
                        commit_oldest()

                    capacity = min(batch_size, examples - selected)
                    raw_fens: list[bytes] = []
                    raw_scores: list[float] = []
                    raw_mates: list[bool] = []
                    while len(raw_fens) < capacity:
                        line = lines.readline()
                        if not line:
                            exhausted = True
                            break
                        source_records += 1
                        record = _extract_record(line)
                        if record is None:
                            skipped["record"] += 1
                            continue
                        fen, score, is_mate = record
                        raw_fens.append(fen)
                        raw_scores.append(score)
                        raw_mates.append(is_mate)
                    if not raw_fens:
                        break

                    encoded = encode_fens(raw_fens, encoder=encoder)
                    valid_indices = np.flatnonzero(encoded.valid)
                    skipped["fen"] += len(raw_fens) - len(valid_indices)
                    if len(valid_indices) > examples - selected:
                        valid_indices = valid_indices[: examples - selected]
                    if len(valid_indices) == 0:
                        snapshot = _snapshot_state(
                            next_batch=next_batch + 1,
                            source_records=source_records,
                            selected=selected,
                            split_counts=split_counts,
                            shard_counts=shard_counts,
                            skipped=skipped,
                            started_at=started_at,
                            elapsed_seconds=previous_elapsed + time.perf_counter() - start_clock,
                        )
                        # Keep state commits in the same FIFO as shard writes.
                        # Advancing state directly here could otherwise skip
                        # an earlier batch whose asynchronous write later fails.
                        pending.append(([], snapshot))
                        next_batch += 1
                        if exhausted:
                            break
                        continue

                    fens = encoded.fen[valid_indices]
                    halfkp32 = encoded.halfkp32[valid_indices]
                    piece768 = encoded.piece768[valid_indices]
                    sides = encoded.white_to_move[valid_indices]
                    scores_stm = np.asarray(raw_scores, dtype=np.float32)[valid_indices]
                    # Lichess cloud-eval scores are side-to-move POV.  Storage is
                    # White POV so training can choose its own perspective once.
                    scores_white = np.where(sides, scores_stm, -scores_stm).astype(
                        np.float32, copy=False
                    )
                    mates = np.asarray(raw_mates, dtype=np.bool_)[valid_indices]
                    assignments = np.asarray([split_for_fen(value) for value in fens])

                    futures: list[Future[int]] = []
                    for split in SPLITS:
                        indices = np.flatnonzero(assignments == split)
                        if len(indices) == 0:
                            continue
                        arrays = {
                            "fen": fens[indices],
                            "halfkp32": halfkp32[indices],
                            "piece768": piece768[indices],
                            "evaluation_cp": scores_white[indices],
                            "white_to_move": sides[indices],
                            "is_mate": mates[indices],
                        }
                        shard_path = root / split / f"shard_{next_batch:06d}.npz"
                        futures.append(pool.submit(_write_npz_atomic, shard_path, arrays, compression_level))
                        split_counts[split] += len(indices)
                        shard_counts[split] += 1

                    selected += len(valid_indices)
                    next_batch += 1
                    snapshot = _snapshot_state(
                        next_batch=next_batch,
                        source_records=source_records,
                        selected=selected,
                        split_counts=split_counts,
                        shard_counts=shard_counts,
                        skipped=skipped,
                        started_at=started_at,
                        elapsed_seconds=previous_elapsed + time.perf_counter() - start_clock,
                    )
                    pending.append((futures, snapshot))
                    if verbose:
                        elapsed = max(time.perf_counter() - start_clock, 1e-9)
                        print(
                            f"{selected:,}/{examples:,} positions | "
                            f"{selected / elapsed:,.0f} positions/s | source {source_records:,}"
                        )
                    if exhausted:
                        break

                while pending:
                    commit_oldest()
        finally:
            lines.close()
            zstd_reader.close()
            compressed.close()

        elapsed_seconds = previous_elapsed + time.perf_counter() - start_clock
        if selected < examples:
            state = _snapshot_state(
                next_batch=next_batch,
                source_records=source_records,
                selected=selected,
                split_counts=split_counts,
                shard_counts=shard_counts,
                skipped=skipped,
                started_at=started_at,
                elapsed_seconds=elapsed_seconds,
            )
            _atomic_json(state_path, state)
            raise RuntimeError(
                f"source ended after {selected:,} valid positions; target was {examples:,}. "
                "Partial shards and resumable state were preserved."
            )

        metadata = _metadata(
            root,
            source_path,
            total=selected,
            source_records=source_records,
            split_counts=split_counts,
            shard_counts=shard_counts,
            skipped=skipped,
            elapsed_seconds=elapsed_seconds,
            pipeline="bounded_zstd_nnue_v2",
            columns=(
                "fen",
                "halfkp32",
                "piece768",
                "evaluation_cp",
                "white_to_move",
                "is_mate",
            ),
            split_policy="BLAKE2b-64(board placement + side to move) % 10; 0-7 train, 8 test, 9 validation",
        )
        published = _publish_metadata(root, metadata)
        result = _result_from_metadata(published, resumed=resumed)
        if verbose:
            print(result)
        return result
    finally:
        _release_lock(descriptor, lock_path)


# The shortest public spelling.
prepare = prepare_dataset


def _validate_existing_shard(path: Path, *, validate_values: bool) -> tuple[int, set[str]]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            columns = set(archive.files)
            required = {"fen", "evaluation_cp", "piece768", "halfkp32"}
            missing = required - columns
            if missing:
                raise ValueError(f"missing columns: {', '.join(sorted(missing))}")
            fens = np.asarray(archive["fen"])
            targets = np.asarray(archive["evaluation_cp"])
            pieces = np.asarray(archive["piece768"])
            halfkp = np.asarray(archive["halfkp32"])
            rows = len(targets)
            if targets.shape != (rows,) or not np.issubdtype(targets.dtype, np.floating):
                raise ValueError("evaluation_cp must be a one-dimensional float array")
            if fens.shape != (rows,) or fens.dtype.kind not in {"S", "U"}:
                raise ValueError("fen must be a one-dimensional string array")
            if pieces.shape != (rows, PIECE_FEATURES) or pieces.dtype != np.uint8:
                raise ValueError("piece768 must have shape (N, 768) and dtype uint8")
            if halfkp.shape != (rows, HALFKP_SLOTS) or halfkp.dtype != np.uint32:
                raise ValueError("halfkp32 must have shape (N, 64) and dtype uint32")
            for optional in ("white_to_move", "is_mate"):
                if optional in columns:
                    value = np.asarray(archive[optional])
                    if value.shape != (rows,) or value.dtype != np.bool_:
                        raise ValueError(f"{optional} must have shape (N,) and dtype bool")
            if validate_values:
                if not bool(np.all(np.isfinite(targets))):
                    raise ValueError("evaluation_cp contains a non-finite value")
                if np.any((pieces != 0) & (pieces != 1)):
                    raise ValueError("piece768 contains values other than zero and one")
                if not bool(np.all((halfkp < HALFKP_FEATURES) | (halfkp == PADDING_INDEX))):
                    raise ValueError("halfkp32 contains an invalid feature index")
            return rows, columns
    except (OSError, ValueError) as error:
        raise ValueError(f"could not validate {path}: {error}") from error


def finalize_existing(
    output_dir: str | Path = OUTPUT_ROOT,
    *,
    source: str | Path | None = SOURCE_PATH,
    validate_values: bool = True,
    verbose: bool = True,
) -> PreparationResult:
    """Validate completed legacy shards and atomically publish their metadata.

    No ``.npz`` file is opened for writing.  This is intended for a previous
    preparation that reached a sufficient row count but stopped before writing
    its completion marker.
    """
    root = Path(output_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    source_path = None if source is None else Path(source).expanduser().resolve()
    descriptor, lock_path = _acquire_lock(root)
    started = time.perf_counter()
    try:
        split_counts: dict[str, int] = {}
        shard_counts: dict[str, int] = {}
        common_columns: set[str] | None = None
        for split in SPLITS:
            shards = sorted((root / split).glob("shard_*.npz"))
            if not shards:
                raise FileNotFoundError(f"no shard_*.npz files found in {root / split}")
            rows = 0
            for index, shard in enumerate(shards, start=1):
                shard_rows, columns = _validate_existing_shard(
                    shard, validate_values=validate_values
                )
                rows += shard_rows
                common_columns = columns if common_columns is None else common_columns & columns
                if verbose and (index % 10 == 0 or index == len(shards)):
                    print(f"Validated {split}: {index}/{len(shards)} shards, {rows:,} rows")
            split_counts[split] = rows
            shard_counts[split] = len(shards)

        columns = common_columns or set()
        ordered_columns = [
            name
            for name in (
                "fen",
                "halfkp32",
                "piece768",
                "evaluation_cp",
                "white_to_move",
                "is_mate",
            )
            if name in columns
        ]
        elapsed = time.perf_counter() - started
        metadata = _metadata(
            root,
            source_path,
            total=sum(split_counts.values()),
            source_records=0,
            split_counts=split_counts,
            shard_counts=shard_counts,
            skipped={},
            elapsed_seconds=elapsed,
            pipeline="legacy_recovered",
            columns=ordered_columns,
            split_policy="legacy deterministic FEN split (existing shards preserved)",
        )
        metadata["note"] = "Metadata recovered after validating existing shards; NPZ files were not rewritten."
        published = _publish_metadata(root, metadata)
        result = _result_from_metadata(published, resumed=False)
        if verbose:
            print(result)
        return result
    finally:
        _release_lock(descriptor, lock_path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--examples", type=int, default=10_000_000)
    parser.add_argument("--batch-size", type=int, default=100_000)
    parser.add_argument("--compression-level", type=int, default=3)
    parser.add_argument("--writer-threads", type=int, default=3)
    parser.add_argument("--max-pending-batches", type=int, default=2)
    parser.add_argument("--encoder", choices=("auto", "numba", "python"), default="auto")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--finalize-existing",
        action="store_true",
        help="validate existing shards and publish metadata without regenerating them",
    )
    parser.add_argument(
        "--quick-finalize",
        action="store_true",
        help="skip expensive value scans while still validating every shard's schema",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.finalize_existing:
        finalize_existing(
            arguments.output_dir,
            source=arguments.source,
            validate_values=not arguments.quick_finalize,
            verbose=not arguments.quiet,
        )
    else:
        prepare_dataset(
            arguments.source,
            arguments.output_dir,
            examples=arguments.examples,
            batch_size=arguments.batch_size,
            compression_level=arguments.compression_level,
            writer_threads=arguments.writer_threads,
            max_pending_batches=arguments.max_pending_batches,
            resume=not arguments.no_resume,
            encoder=arguments.encoder,
            verbose=not arguments.quiet,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
