import chess
from src.model import MoveOrderingModel
from src.move_encoder import moves_to_tensor
from src.ranking_loss import pairwise_ranking_loss


def test_model_outputs_one_score_per_move_and_backpropagates() -> None:
    board = chess.Board()
    model = MoveOrderingModel(position_dim=16, move_dim=8, hidden_dim=32)
    moves = moves_to_tensor(list(board.legal_moves)[:3])
    scores = model.score_moves(model.encode_position(board), moves)
    assert scores.shape == (3,)
    pair_loss = pairwise_ranking_loss(scores[:1], scores[1:2])
    pair_loss.backward()
    assert model.halfkp_embedding.weight.grad is not None
