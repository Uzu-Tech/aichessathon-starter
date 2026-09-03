"""Static evaluation.

Material on its own cannot tell a knight on the rim from a knight in the centre, so
every piece is scored as its value plus a piece-square bonus for where it stands. The
tables come in a middlegame and an endgame set and the two are blended by how much
material is left, which keeps the king's score from jumping the moment a queen trades.
"""

import chess

from config import PHASE_WEIGHT, PIECE_VALUE, PST_ENDGAME, PST_MIDDLEGAME, TOTAL_PHASE

# [piece_type][color][square] -> value + placement bonus, signed for White's point of
# view. Built once at import so the search only ever does a lookup and an add.
Table = list[list[list[float]]]


def build_table(tables: dict[chess.Color, dict[chess.PieceType, list[int]]]) -> Table:
    table: Table = [[[0.0] * 64 for _ in range(2)] for _ in range(len(chess.PIECE_TYPES) + 1)]
    for color, squares_by_piece in tables.items():
        sign = 1.0 if color == chess.WHITE else -1.0
        for piece_type, squares in squares_by_piece.items():
            value = PIECE_VALUE[piece_type]
            for square in chess.SQUARES:
                # The tables are written a8 first, so a square's index into them is its
                # own index with the ranks flipped. Both colours read theirs the same
                # way; it is the tables themselves that are mirrored.
                table[piece_type][color][square] = sign * (value + squares[square ^ 56])
    return table


MIDDLEGAME_TABLE = build_table(PST_MIDDLEGAME)
ENDGAME_TABLE = build_table(PST_ENDGAME)


def evaluate(board: chess.Board) -> float:
    """Score the position in centipawns from the point of view of the side to move."""
    middlegame = 0.0
    endgame = 0.0
    phase = 0

    for piece_type in chess.PIECE_TYPES:
        middlegame_squares = MIDDLEGAME_TABLE[piece_type]
        endgame_squares = ENDGAME_TABLE[piece_type]
        weight = PHASE_WEIGHT[piece_type]
        for color in (chess.WHITE, chess.BLACK):
            middlegame_side = middlegame_squares[color]
            endgame_side = endgame_squares[color]
            for square in chess.scan_forward(board.pieces_mask(piece_type, color)):
                middlegame += middlegame_side[square]
                endgame += endgame_side[square]
                phase += weight

    # A promotion can put more material on the board than the opening had, so cap it.
    blend = min(phase, TOTAL_PHASE) / TOTAL_PHASE
    score = middlegame * blend + endgame * (1.0 - blend)

    return score if board.turn == chess.WHITE else -score
