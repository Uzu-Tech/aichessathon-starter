import chess
import pytest
from src.halfkp import FEATURE_COUNT, feature_index, halfkp_indices


def test_indices_are_in_range_and_exclude_kings() -> None:
    board = chess.Board()
    indices = halfkp_indices(board)
    assert all(0 <= index < FEATURE_COUNT for index in indices)
    assert len(indices) == len(board.piece_map()) - 2


def test_piece_square_changes_feature() -> None:
    board = chess.Board()
    before = halfkp_indices(board)
    board.push_uci("e2e4")
    assert before != halfkp_indices(board, chess.WHITE)


def test_side_to_move_changes_the_relative_representation() -> None:
    board = chess.Board("4k3/8/8/8/3p4/2P5/8/4K3 w - - 0 1")
    white_features = halfkp_indices(board)
    board.turn = chess.BLACK
    assert white_features != halfkp_indices(board)


def test_king_piece_feature_is_rejected() -> None:
    with pytest.raises(ValueError):
        feature_index(chess.E1, chess.E8, chess.KING, chess.BLACK, chess.WHITE)
