"""Classic Stockfish-style HalfKP feature extraction.

For the side to move, squares are viewed from that side: Black positions are
flipped vertically (``square ^ 56``), and a piece colour is encoded as us/them.
For every non-king piece, the feature index is::

    piece_square + (piece_slot + king_square * 10) * 64

where ``piece_slot = (piece_type - 1) * 2 + relative_colour`` and
``relative_colour`` is 0 for us and 1 for them.  There are ten piece slots
(pawn through queen, each for us and them), so the valid range is 0..40959.
This is the classic 64 * 64 * 5 * 2 HalfKP layout documented by Stockfish's
NNUE training project; kings select the bucket but are never piece features.
"""

from __future__ import annotations

import chess

FEATURE_COUNT = 64 * 64 * 5 * 2
PIECE_SLOTS = 10


def _orient(square: chess.Square, perspective: chess.Color) -> chess.Square:
    """Return a square viewed from ``perspective`` (White-facing orientation)."""
    return square if perspective == chess.WHITE else chess.square_mirror(square)


def feature_index(
    king_square: chess.Square,
    piece_square: chess.Square,
    piece_type: chess.PieceType,
    piece_color: chess.Color,
    perspective: chess.Color,
) -> int:
    """Return one HalfKP feature index; passing a king is intentionally invalid."""
    if piece_type == chess.KING:
        raise ValueError("HalfKP does not include kings as piece features")
    if not chess.PAWN <= piece_type <= chess.QUEEN:
        raise ValueError(f"unsupported piece type: {piece_type}")
    oriented_king = _orient(king_square, perspective)
    oriented_piece = _orient(piece_square, perspective)
    relative_color = 0 if piece_color == perspective else 1
    piece_slot = (piece_type - chess.PAWN) * 2 + relative_color
    return oriented_piece + (piece_slot + oriented_king * PIECE_SLOTS) * 64


def halfkp_indices(board: chess.Board, perspective: chess.Color | None = None) -> list[int]:
    """Extract active HalfKP indices for a board, defaulting to its side to move."""
    view = board.turn if perspective is None else perspective
    king_square = board.king(view)
    if king_square is None:
        raise ValueError("a valid chess position must contain the perspective king")
    indices: list[int] = []
    for square, piece in board.piece_map().items():
        if piece.piece_type != chess.KING:
            indices.append(feature_index(king_square, square, piece.piece_type, piece.color, view))
    return sorted(indices)
