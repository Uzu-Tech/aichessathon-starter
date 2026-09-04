"""Static evaluation.

Material on its own cannot tell a knight on the rim from a knight in the centre, so
every piece is scored as its value plus a piece-square bonus for where it stands. The
tables come in a middle_game and an endgame set and the two are blended by how much
material is left, which keeps the king's score from jumping the moment a queen trades.
"""

import chess
from config import Config

# [piece_type][color][square] -> value + placement bonus, signed for White's point of
# view. Built once at import so the search only ever does a lookup and an add.

def evaluate(board: chess.Board, config: Config) -> float:
    """Score the position in centipawns from the point of view of the side to move."""
    middle_game = 0.0
    endgame = 0.0
    phase = 0
    mover = board.turn
    
    for piece_type in chess.PIECE_TYPES:
        middle_game_squares = config.MIDDLE_GAME_TABLE[piece_type]
        endgame_squares = config.ENDGAME_TABLE[piece_type]
        weight = config.PHASE_WEIGHT[piece_type]
        for color in (chess.WHITE, chess.BLACK):
            middle_game_side = middle_game_squares[color]
            endgame_side = endgame_squares[color]
            for square in chess.scan_forward(board.pieces_mask(piece_type, color)):
                middle_game += middle_game_side[square]
                endgame +=  endgame_side[square]
                phase += weight

    # A promotion can put more material on the board than the opening had, so cap it.
    blend = min(phase, config.TOTAL_PHASE) / config.TOTAL_PHASE
    score = middle_game * blend + endgame * (1.0 - blend)

    return score if board.turn == chess.WHITE else -score