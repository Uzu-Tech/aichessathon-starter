import chess
import torch
from src.inference import order_moves
from src.model import MoveOrderingModel


def test_order_moves_returns_all_legal_moves_in_descending_score_order() -> None:
    torch.manual_seed(7)
    board = chess.Board()
    model = MoveOrderingModel(position_dim=16, move_dim=8, hidden_dim=32)
    ordered = order_moves(board, model)
    assert set(ordered) == set(board.legal_moves)
    assert len(ordered) == board.legal_moves.count()
    embedding = model.encode_position(board)
    scores = model.score_moves(embedding, ordered).tolist()
    assert scores == sorted(scores, reverse=True)
