import chess
from config import material_value

def evaluate(board: chess.Board) -> float:
    mover = board.turn
    material = sum(
        value * (len(board.pieces(piece, mover)) - len(board.pieces(piece, not mover)))
        for piece, value in material_value.PIECE_VALUE.items()
    )
    return material