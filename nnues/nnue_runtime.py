"""Quantized incremental HalfKP runtime shared by the NNUE evaluators."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import chess
import numba
import numpy as np
import numpy.typing as npt
import onnxruntime as ort

HALFKP_FEATURES: Final = 20_480
MAX_PLY: Final = 320

FloatArray = npt.NDArray[np.float32]
Int8Array = npt.NDArray[np.int8]
Int32Array = npt.NDArray[np.int32]


@numba.njit(inline="always")
def _feature_index(
    king_square: int,
    colour: int,
    piece_type: int,
    square: int,
    black_perspective: bool,
) -> int:
    if black_perspective:
        king_square ^= 56
        square ^= 56
        colour ^= 1
    if king_square % 8 < 4:
        king_square ^= 7
        square ^= 7
    king_bucket = (king_square // 8) * 4 + king_square % 8 - 4
    return king_bucket * 640 + (colour * 5 + piece_type) * 64 + square


@numba.njit(nogil=True, fastmath=True)
def _apply_piece(
    accumulators: Int32Array,
    weights: Int8Array,
    white_king: int,
    black_king: int,
    colour: int,
    piece_type: int,
    square: int,
    sign: int,
) -> None:
    white_index = _feature_index(
        white_king, colour, piece_type, square, False
    )
    black_index = _feature_index(
        black_king, colour, piece_type, square, True
    )
    for column in range(accumulators.shape[1]):
        accumulators[0, column] += sign * int(weights[white_index, column])
        accumulators[1, column] += sign * int(weights[black_index, column])


@numba.njit(nogil=True, fastmath=True)
def _refresh_perspective(
    accumulator: Int32Array,
    weights: Int8Array,
    bias: Int32Array,
    pieces: Int8Array,
    king_square: int,
    black_perspective: bool,
) -> None:
    for column in range(accumulator.shape[0]):
        accumulator[column] = bias[column]
    for square in range(64):
        code = int(pieces[square])
        if code < 0 or code % 6 == 5:
            continue
        colour = code // 6
        piece_type = code % 6
        index = _feature_index(
            king_square, colour, piece_type, square, black_perspective
        )
        for column in range(accumulator.shape[0]):
            accumulator[column] += int(weights[index, column])


@numba.njit(nogil=True, fastmath=True)
def _copy_accumulators(destination: Int32Array, source: Int32Array) -> None:
    for perspective in range(2):
        for column in range(source.shape[1]):
            destination[perspective, column] = source[perspective, column]


@numba.njit(nogil=True, fastmath=True)
def _orient(
    output: FloatArray,
    accumulators: Int32Array,
    scales: FloatArray,
    white_to_move: bool,
) -> None:
    first = 0 if white_to_move else 1
    second = 1 - first
    width = accumulators.shape[1]
    for column in range(width):
        first_value = float(accumulators[first, column]) * scales[column]
        second_value = float(accumulators[second, column]) * scales[column]
        output[0, column] = min(1.0, max(0.0, first_value))
        output[0, width + column] = min(1.0, max(0.0, second_value))


def _board_pieces(board: chess.Board) -> Int8Array:
    pieces = np.full(64, -1, dtype=np.int8)
    for square, piece in board.piece_map().items():
        colour = 0 if piece.color == chess.WHITE else 1
        pieces[square] = colour * 6 + piece.piece_type - 1
    return pieces


def _king_squares(board: chess.Board) -> tuple[int, int]:
    white = board.king(chess.WHITE)
    black = board.king(chess.BLACK)
    if white is None or black is None:
        raise ValueError("NNUE positions must contain both kings")
    return white, black


class NNUEBackend:
    """Load an int8 transformer and quantized ONNX head once."""

    def __init__(
        self,
        transformer_path: str | Path,
        bias_path: str | Path,
        scale_path: str | Path,
        head_path: str | Path,
        width: int,
    ):
        weights = np.load(Path(transformer_path), mmap_mode="r", allow_pickle=False)
        bias = np.load(Path(bias_path), mmap_mode="r", allow_pickle=False)
        scales = np.load(Path(scale_path), mmap_mode="r", allow_pickle=False)
        if weights.shape != (HALFKP_FEATURES, width) or weights.dtype != np.int8:
            raise ValueError(
                f"expected int8 transformer shape {(HALFKP_FEATURES, width)}, "
                f"got {weights.shape} {weights.dtype}"
            )
        if bias.shape != (width,) or bias.dtype != np.int32:
            raise ValueError(f"expected int32 bias shape {(width,)}, got {bias.shape} {bias.dtype}")
        if scales.shape != (width,) or scales.dtype != np.float32:
            raise ValueError(
                f"expected float32 scales shape {(width,)}, got {scales.shape} {scales.dtype}"
            )
        self.width = width
        self.weights: Int8Array = weights
        self.bias: Int32Array = bias
        self.scales: FloatArray = scales
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(Path(head_path)),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self._warm_numba()

    def _warm_numba(self) -> None:
        accumulators = np.zeros((2, self.width), dtype=np.int32)
        stack = np.zeros_like(accumulators)
        oriented = np.zeros((1, self.width * 2), dtype=np.float32)
        pieces = np.full(64, -1, dtype=np.int8)
        _apply_piece(accumulators, self.weights, 4, 60, 0, 0, 8, 1)
        _refresh_perspective(
            accumulators[0], self.weights, self.bias, pieces, 4, False
        )
        _copy_accumulators(stack, accumulators)
        _orient(oriented, accumulators, self.scales, True)

    def position(self, fen: str = chess.STARTING_FEN) -> NNUEPosition:
        return NNUEPosition(self, chess.Board(fen))


class NNUEPosition:
    """A board plus incrementally maintained accumulators and push/pop stack."""

    def __init__(self, backend: NNUEBackend, board: chess.Board):
        if board.chess960:
            raise ValueError("Chess960 castling is not supported")
        self.backend = backend
        self.board = board
        self.accumulators = np.empty((2, backend.width), dtype=np.int32)
        self._oriented = np.empty((1, backend.width * 2), dtype=np.float32)
        self._stack = np.empty((MAX_PLY, 2, backend.width), dtype=np.int32)
        self._ply = 0
        white_king, black_king = _king_squares(board)
        pieces = _board_pieces(board)
        _refresh_perspective(
            self.accumulators[0],
            backend.weights,
            backend.bias,
            pieces,
            white_king,
            False,
        )
        _refresh_perspective(
            self.accumulators[1],
            backend.weights,
            backend.bias,
            pieces,
            black_king,
            True,
        )

    def _piece_delta(
        self,
        white_king: int,
        black_king: int,
        colour: int,
        piece_type: int,
        square: int,
        sign: int,
    ) -> None:
        _apply_piece(
            self.accumulators,
            self.backend.weights,
            white_king,
            black_king,
            colour,
            piece_type,
            square,
            sign,
        )

    def push(self, move: chess.Move) -> None:
        """Push a move after checking that it is legal."""
        if move and not self.board.is_legal(move):
            raise ValueError(f"illegal move {move.uci()} for {self.board.fen()}")
        self.push_unchecked(move)

    def push_unchecked(self, move: chess.Move) -> None:
        """Push a move already obtained from ``board.legal_moves``."""
        if self._ply >= MAX_PLY:
            raise OverflowError(f"NNUE stack is limited to {MAX_PLY} plies")
        if not move:
            _copy_accumulators(self._stack[self._ply], self.accumulators)
            self._ply += 1
            self.board.push(move)
            return
        moving = self.board.piece_at(move.from_square)
        if moving is None:
            raise ValueError(f"no piece on {chess.square_name(move.from_square)}")

        _copy_accumulators(self._stack[self._ply], self.accumulators)
        self._ply += 1
        white_king, black_king = _king_squares(self.board)
        colour = 0 if moving.color == chess.WHITE else 1
        piece_type = moving.piece_type - 1

        if piece_type != chess.KING - 1:
            self._piece_delta(
                white_king, black_king, colour, piece_type, move.from_square, -1
            )
            destination_type = (
                move.promotion - 1 if move.promotion is not None else piece_type
            )
            self._piece_delta(
                white_king, black_king, colour, destination_type, move.to_square, 1
            )

        if self.board.is_en_passant(move):
            captured_square = move.to_square - 8 if moving.color else move.to_square + 8
            captured = self.board.piece_at(captured_square)
        else:
            captured_square = move.to_square
            captured = self.board.piece_at(captured_square)
        if captured is not None:
            captured_colour = 0 if captured.color == chess.WHITE else 1
            self._piece_delta(
                white_king,
                black_king,
                captured_colour,
                captured.piece_type - 1,
                captured_square,
                -1,
            )

        castling = self.board.is_castling(move)
        if castling:
            rank = chess.square_rank(move.from_square)
            kingside = self.board.is_kingside_castling(move)
            rook_from = chess.square(7 if kingside else 0, rank)
            rook_to = chess.square(5 if kingside else 3, rank)
            self._piece_delta(
                white_king, black_king, colour, chess.ROOK - 1, rook_from, -1
            )
            self._piece_delta(
                white_king, black_king, colour, chess.ROOK - 1, rook_to, 1
            )

        self.board.push(move)
        if piece_type == chess.KING - 1:
            pieces = _board_pieces(self.board)
            new_white_king, new_black_king = _king_squares(self.board)
            perspective = colour
            _refresh_perspective(
                self.accumulators[perspective],
                self.backend.weights,
                self.backend.bias,
                pieces,
                new_white_king if perspective == 0 else new_black_king,
                perspective == 1,
            )

    def pop(self) -> chess.Move:
        if self._ply == 0:
            raise IndexError("cannot pop the root NNUE position")
        move = self.board.pop()
        self._ply -= 1
        _copy_accumulators(self.accumulators, self._stack[self._ply])
        return move

    def evaluate(self) -> float:
        _orient(
            self._oriented,
            self.accumulators,
            self.backend.scales,
            self.board.turn,
        )
        output = self.backend.session.run(
            ["score_cp"], {"accumulator": self._oriented}
        )[0]
        return float(output[0, 0])


__all__ = ["NNUEBackend", "NNUEPosition"]
