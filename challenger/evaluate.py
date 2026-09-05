"""Team-trained HalfKP NNUE: ``evaluate(board)`` returns side-to-move centipawns.

Only this file and its adjacent weights/ directory are needed for evaluation.
Import loads weights and warms Numba/ONNX; no PyTorch or training code is used.
The private accumulator follows the last evaluated board, including arbitrary
push/pop sequences and unrelated board objects. Calls must be sequential.
Search remains responsible for checkmate, stalemate and draw adjudication.
"""

from __future__ import annotations

import hashlib as _hashlib
import json as _json
from pathlib import Path as _Path

import chess as _chess
import numpy as _np
import onnxruntime as _ort  # type: ignore[import-untyped]
from numba import njit as _njit
from numpy.typing import NDArray as _NDArray

__all__ = ["evaluate"]

_IntArray = _NDArray[_np.int32]
_Bitboards = _NDArray[_np.uint64]
_FloatArray = _NDArray[_np.float32]
_WeightArray = _NDArray[_np.int16]


def _lsb_impl(bits: _np.uint64) -> int:
    """Index of the least significant set bit; input is nonzero uint64."""
    square = 0
    if bits & _np.uint64(0xFFFFFFFF) == 0:
        bits >>= _np.uint64(32)
        square += 32
    if bits & _np.uint64(0xFFFF) == 0:
        bits >>= _np.uint64(16)
        square += 16
    if bits & _np.uint64(0xFF) == 0:
        bits >>= _np.uint64(8)
        square += 8
    if bits & _np.uint64(0xF) == 0:
        bits >>= _np.uint64(4)
        square += 4
    if bits & _np.uint64(3) == 0:
        bits >>= _np.uint64(2)
        square += 2
    if bits & _np.uint64(1) == 0:
        square += 1
    return square


_lsb = _njit(cache=False, inline="always")(_lsb_impl)


def _sync_impl(
    pieces: _Bitboards,
    white_king: int,
    black_king: int,
    previous_pieces: _Bitboards,
    previous_kings: _IntArray,
    accumulators: _IntArray,
    weights: _WeightArray,
    bias: _IntArray,
) -> None:
    # Compare piece bitboards, not move stacks: skipped evaluations and TT
    # cutoffs need no hooks, and switching to any other Board stays correct.
    for perspective in range(2):
        raw_king = white_king if perspective == 0 else black_king
        refresh = raw_king != previous_kings[perspective]
        rank_flip = 0 if perspective == 0 else 56
        king = raw_king ^ rank_flip
        mirror = 7 if king % 8 >= 4 else 0
        king ^= mirror
        bucket_offset = ((king // 8) * 4 + king % 8) * 640
        square_flip = rank_flip ^ mirror
        if refresh:
            # Raw king equality matters: d/e crossings can keep the bucket
            # number while mirroring every non-king feature.
            for neuron in range(bias.size):
                accumulators[perspective, neuron] = bias[neuron]
        for plane in range(10):
            oriented_plane = plane if perspective == 0 else (plane + 5) % 10
            offset = bucket_offset + oriented_plane * 64
            removed = _np.uint64(0) if refresh else previous_pieces[plane] & ~pieces[plane]
            added = pieces[plane] if refresh else pieces[plane] & ~previous_pieces[plane]
            while removed:
                index = offset + (_lsb(removed) ^ square_flip)
                for neuron in range(bias.size):
                    accumulators[perspective, neuron] -= weights[index, neuron]
                removed &= removed - _np.uint64(1)
            while added:
                index = offset + (_lsb(added) ^ square_flip)
                for neuron in range(bias.size):
                    accumulators[perspective, neuron] += weights[index, neuron]
                added &= added - _np.uint64(1)
        previous_kings[perspective] = raw_king
    for plane in range(10):
        previous_pieces[plane] = pieces[plane]


_sync = _njit(cache=False)(_sync_impl)


def _prepare_impl(
    accumulators: _IntArray,
    white_to_move: bool,
    inverse_scale: _np.float32,
    inputs: _FloatArray,
) -> None:
    width = accumulators.shape[1]
    first = 0 if white_to_move else 1
    for column in range(width):
        inputs[0, column] = min(
            _np.float32(1.0),
            max(_np.float32(0.0), _np.float32(accumulators[first, column]) * inverse_scale),
        )
        inputs[0, width + column] = min(
            _np.float32(1.0),
            max(_np.float32(0.0), _np.float32(accumulators[1 - first, column]) * inverse_scale),
        )


_prepare = _njit(cache=False)(_prepare_impl)


def _head_impl(
    accumulators: _IntArray,
    white_to_move: bool,
    inverse_scale: _np.float32,
    output_scale: _np.float32,
    first_weights: _FloatArray,
    first_bias: _FloatArray,
    last_weights: _FloatArray,
    last_bias: _np.float32,
    inputs: _FloatArray,
) -> float:
    _prepare(accumulators, white_to_move, inverse_scale, inputs)
    result = _np.float32(0.0)
    for row in range(first_weights.shape[0]):
        # Adding bias after the reduction lets LLVM vectorize the dot product.
        value = _np.float32(0.0)
        for column in range(first_weights.shape[1]):
            value += first_weights[row, column] * inputs[0, column]
        value = min(_np.float32(1.0), max(_np.float32(0.0), value + first_bias[row]))
        result += value * last_weights[row]
    return float((result + last_bias) * output_scale)


_head = _njit(cache=False, fastmath=True)(_head_impl)


def _onnx_evaluate() -> float:
    """Private benchmark alternative; score the synchronized accumulator."""
    _prepare(_accumulators, _white_to_move, _inverse_scale, _inputs)
    _session.run_with_iobinding(_binding)
    return float(_output[0, 0])


def evaluate(board: _chess.Board, config: object | None = None) -> float:
    """Return a static NNUE score in centipawns, positive for the side to move.

    Pass a normal python-chess Board and use its usual push()/pop(). This
    function updates its own accumulator without changing the board. ``config``
    is accepted for search compatibility; trained weights determine evaluation.
    """
    global _white_to_move
    del config
    white_king = board.king(_chess.WHITE)
    black_king = board.king(_chess.BLACK)
    if white_king is None or black_king is None:
        raise ValueError("NNUE evaluation requires both kings")
    white = board.occupied_co[_chess.WHITE]
    black = board.occupied_co[_chess.BLACK]
    _current_pieces[0] = board.pawns & white
    _current_pieces[1] = board.knights & white
    _current_pieces[2] = board.bishops & white
    _current_pieces[3] = board.rooks & white
    _current_pieces[4] = board.queens & white
    _current_pieces[5] = board.pawns & black
    _current_pieces[6] = board.knights & black
    _current_pieces[7] = board.bishops & black
    _current_pieces[8] = board.rooks & black
    _current_pieces[9] = board.queens & black
    _sync(
        _current_pieces,
        white_king,
        black_king,
        _previous_pieces,
        _previous_kings,
        _accumulators,
        _feature_weights,
        _feature_bias,
    )
    _white_to_move = board.turn
    if _head_backend == "onnx":
        return _onnx_evaluate()
    return float(
        _head(
            _accumulators,
            _white_to_move,
            _inverse_scale,
            _output_scale,
            _first_weights,
            _first_bias,
            _last_weights,
            _last_bias,
            _inputs,
        )
    )


_weights_path = _Path(__file__).resolve().parent / "weights"
with _np.load(_weights_path / "nnue.npz", allow_pickle=False) as _archive:
    _metadata = _json.loads(str(_archive["metadata"].item()))
    if (
        _metadata["format_version"] != 1
        or _metadata["feature_schema"] != "halfkp_hm_left_v1"
        or _metadata["architecture"] != "256x32x1"
        or _metadata["activation"] != "clipped_relu_0_1"
        or _metadata["perspective_order"] != "white_black_then_side_to_move_first"
        or _archive["head_widths"].tolist() != [512, 32, 1]
        or _archive["weight_offsets"].tolist() != [0, 16384]
        or _archive["bias_offsets"].tolist() != [0, 32]
    ):
        raise ValueError("weights do not match the trained HalfKP 256x32x1 runtime")
    _feature_weights: _WeightArray = _np.ascontiguousarray(_archive["feature_weights"])
    _feature_bias: _IntArray = _np.ascontiguousarray(_archive["feature_bias"])
    _head_weights: _FloatArray = _np.ascontiguousarray(_archive["head_weights"])
    _head_bias: _FloatArray = _np.ascontiguousarray(_archive["head_bias"])
    if (
        _feature_weights.shape != (20480, 256)
        or _feature_weights.dtype != _np.int16
        or _feature_bias.shape != (256,)
        or _feature_bias.dtype != _np.int32
        or _head_weights.shape != (16416,)
        or _head_weights.dtype != _np.float32
        or _head_bias.shape != (33,)
        or _head_bias.dtype != _np.float32
        or not _np.isfinite(_head_weights).all()
        or not _np.isfinite(_head_bias).all()
    ):
        raise ValueError("invalid NNUE weight shapes, dtypes or values")
    _quantization_scale = float(_metadata["quantization_scale"])
    _output_scale_cp = float(_metadata["output_scale_cp"])
    if not (0 < _quantization_scale < 1e9 and 0 < _output_scale_cp < 1e9):
        raise ValueError("invalid NNUE scales")

# The piece masks contain at most 64 squares, even for a synthetic board.
if int(_np.max(_np.abs(_feature_bias.astype(_np.int64)))) + 64 * 32768 > _np.iinfo(_np.int32).max:
    raise ValueError("NNUE accumulator could overflow int32")

_inverse_scale = _np.float32(1.0 / _quantization_scale)
_output_scale = _np.float32(_output_scale_cp)
_first_weights = _head_weights[:16384].reshape(32, 512)
_first_bias = _head_bias[:32]
_last_weights = _head_weights[16384:]
_last_bias = _np.float32(_head_bias[32])
_accumulators = _np.zeros((2, 256), dtype=_np.int32)
_current_pieces = _np.empty(10, dtype=_np.uint64)
_previous_pieces = _np.zeros(10, dtype=_np.uint64)
_previous_kings = _np.full(2, -1, dtype=_np.int32)
_inputs = _np.empty((1, 512), dtype=_np.float32)
_output = _np.empty((1, 1), dtype=_np.float32)
_white_to_move = True

_options = _ort.SessionOptions()
_options.intra_op_num_threads = 1
_options.inter_op_num_threads = 1
_options.execution_mode = _ort.ExecutionMode.ORT_SEQUENTIAL
_options.graph_optimization_level = _ort.GraphOptimizationLevel.ORT_ENABLE_ALL
_options.log_severity_level = 3
_session = _ort.InferenceSession(
    str(_weights_path / "head.onnx"),
    sess_options=_options,
    providers=["CPUExecutionProvider"],
)
if (
    _session.get_modelmeta().custom_metadata_map.get("source_npz_sha256")
    != _hashlib.sha256((_weights_path / "nnue.npz").read_bytes()).hexdigest()
):
    raise ValueError("ONNX head and accumulator weights come from different models")
_binding = _session.io_binding()
_binding.bind_cpu_input("features", _inputs)
_binding.bind_output(
    "centipawns",
    device_type="cpu",
    device_id=0,
    element_type=_np.float32,
    shape=_output.shape,
    buffer_ptr=_output.ctypes.data,
)

# Batch-one measurements favor the SIMD Numba head over ONNX I/O binding.
# Keep ONNX private for comparison; forcing both heads into each call is slower.
_head_backend = "numba"
_warm_score = evaluate(_chess.Board())
if abs(_warm_score - _onnx_evaluate()) > 0.002:
    raise ValueError("Numba and ONNX disagree during NNUE initialization")
