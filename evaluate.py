"""Static evaluation.

Material on its own cannot tell a knight on the rim from a knight in the centre, so
every piece is scored as its value plus a piece-square bonus for where it stands. The
tables come in a middlegame and an endgame set and the two are blended by how much
material is left, which keeps the king's score from jumping the moment a queen trades.
"""

import chess

from config import config

# [piece_type][color][square] -> value + placement bonus, signed for White's point of
# view. Built once at import so the search only ever does a lookup and an add.
def evaluate_pawn():
    pass

def evaluate_knight():
    pass

def evaluate_bishop():
    pass

def evaluate_rook():
    pass

def evaluate_queen():
    pass

def evaluate_king():
    pass

PIECE_HANDLER = {
    chess.PAWN : evaluate_pawn,
    chess.KNIGHT : evaluate_knight,
    chess.BISHOP : evaluate_bishop,
    chess.ROOK : evaluate_rook,
    chess.QUEEN : evaluate_queen,
    chess.KING : evaluate_king,

}


def evaluate(board: chess.Board) -> float:
    """Score the position in centipawns from the point of view of the side to move."""
    middlegame = 0.0
    endgame = 0.0
    phase = 0
    mover = board.turn
    for piece_type in chess.PIECE_TYPES:
        handler = PIECE_HANDLER[piece_type]
        middlegame_squares = config.MIDDLEGAME_TABLE[piece_type]
        endgame_squares = config.ENDGAME_TABLE[piece_type]
        weight = config.PHASE_WEIGHT[piece_type]
        for color in (chess.WHITE, chess.BLACK):
            middlegame_side = middlegame_squares[color]
            endgame_side = endgame_squares[color]
            for square in chess.scan_forward(board.pieces_mask(piece_type, color)):
                middlegame += middlegame_side[square] + handler
                endgame +=  endgame_side[square]
                phase += weight

    # A promotion can put more material on the board than the opening had, so cap it.
    blend = min(phase, config.TOTAL_PHASE) / config.TOTAL_PHASE
    score = middlegame * blend + endgame * (1.0 - blend)

    return score if board.turn == chess.WHITE else -score
