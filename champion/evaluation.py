"""Static evaluation used by the challenger search.

Scores are assembled from White's perspective and flipped only at the public
boundary.  This keeps every positional term symmetric for negamax.
"""

import chess

from config import Config


# Centipawn weights.  Positional terms intentionally remain small vs material.
DOUBLED_PAWN = 12.0
ISOLATED_PAWN = 14.0
BACKWARD_PAWN = 10.0
PAWN_ISLAND = 6.0
PAWN_CHAIN = 5.0
PASSED_PAWN_BY_RANK = (0.0, 0.0, 8.0, 16.0, 30.0, 55.0, 90.0, 0.0)
CONNECTED_PASSER = 14.0

MOBILITY_WEIGHT: dict[chess.PieceType, float] = {
    chess.PAWN: 1.0, chess.KNIGHT: 4.0, chess.BISHOP: 3.5,
    chess.ROOK: 2.0, chess.QUEEN: 1.0, chess.KING: 0.5,
}


def _squares(
    board: chess.Board, piece_type: chess.PieceType, color: chess.Color
) -> list[chess.Square]:
    return list(chess.scan_forward(board.pieces_mask(piece_type, color)))


def _advance(color: chess.Color) -> int:
    return 1 if color == chess.WHITE else -1


def _relative_rank(square: chess.Square, color: chess.Color) -> int:
    rank = chess.square_rank(square)
    return rank if color == chess.WHITE else 7 - rank


def _pawn_is_passed(
    square: chess.Square, color: chess.Color, enemy_pawns: list[chess.Square]
) -> bool:
    file, rank = chess.square_file(square), chess.square_rank(square)
    for enemy in enemy_pawns:
        enemy_rank = chess.square_rank(enemy)
        if abs(chess.square_file(enemy) - file) <= 1 and (
            (color == chess.WHITE and enemy_rank > rank)
            or (color == chess.BLACK and enemy_rank < rank)
        ):
            return False
    return True


def evaluate_pawn(board: chess.Board, color: chess.Color) -> float:
    """Pawn shape, passers, chains, and islands for one side."""
    pawns = _squares(board, chess.PAWN, color)
    enemy_pawns = _squares(board, chess.PAWN, not color)
    by_file: list[list[chess.Square]] = [[] for _ in range(8)]
    for square in pawns:
        by_file[chess.square_file(square)].append(square)

    score = 0.0
    islands = sum(bool(by_file[file]) and (file == 0 or not by_file[file - 1]) for file in range(8))
    score -= PAWN_ISLAND * max(0, islands - 1)
    passed: set[chess.Square] = set()
    direction = _advance(color)
    enemy_pawn_attacks = chess.BB_EMPTY
    for enemy in enemy_pawns:
        enemy_pawn_attacks |= int(board.attacks(enemy))

    for file, file_pawns in enumerate(by_file):
        score -= DOUBLED_PAWN * max(0, len(file_pawns) - 1)
        adjacent = (by_file[file - 1] if file else []) + (by_file[file + 1] if file < 7 else [])
        for square in file_pawns:
            rank = chess.square_rank(square)
            if not adjacent:
                score -= ISOLATED_PAWN
            supports_advance = any(
                (chess.square_rank(other) - rank) * direction >= 0 for other in adjacent
            )
            front_rank = rank + direction
            if 0 <= front_rank < 8 and not supports_advance:
                if enemy_pawn_attacks & chess.BB_SQUARES[chess.square(file, front_rank)]:
                    score -= BACKWARD_PAWN
            if board.attackers(color, square) & board.pieces_mask(chess.PAWN, color):
                score += PAWN_CHAIN
            if _pawn_is_passed(square, color, enemy_pawns):
                passed.add(square)
                score += PASSED_PAWN_BY_RANK[_relative_rank(square, color)]

    for square in passed:
        file = chess.square_file(square)
        if (file and any(other in passed for other in by_file[file - 1])) or (
            file < 7 and any(other in passed for other in by_file[file + 1])
        ):
            score += CONNECTED_PASSER
    return score


def _legal_mobility(board: chess.Board, color: chess.Color) -> dict[chess.Square, int]:
    """Legal moves by origin, so pinned/trapped pieces are counted correctly."""
    original_turn = board.turn
    board.turn = color
    try:
        result: dict[chess.Square, int] = {}
        for move in board.generate_legal_moves():
            result[move.from_square] = result.get(move.from_square, 0) + 1
        return result
    finally:
        board.turn = original_turn


def evaluate_mobility(
    board: chess.Board, color: chess.Color, moves_by_square: dict[chess.Square, int] | None = None
) -> float:
    if moves_by_square is None:
        moves_by_square = _legal_mobility(board, color)
    return sum(
        MOBILITY_WEIGHT[piece.piece_type] * count
        for square, count in moves_by_square.items()
        if (piece := board.piece_at(square)) is not None and piece.color == color
    )


def evaluate_king(board: chess.Board, color: chess.Color) -> float:
    king = board.king(color)
    if king is None:
        return 0.0
    score = 0.0
    king_file, king_rank = chess.square_file(king), chess.square_rank(king)
    direction = _advance(color)

    # A pawn shield retains some value after a king moves from its castled square.
    for file in range(max(0, king_file - 1), min(7, king_file + 1) + 1):
        for distance, bonus in ((1, 12.0), (2, 5.0)):
            rank = king_rank + direction * distance
            if 0 <= rank < 8 and board.piece_at(chess.square(file, rank)) == chess.Piece(
                chess.PAWN, color
            ):
                score += bonus
        own_pawns = bool(board.pieces_mask(chess.PAWN, color) & chess.BB_FILES[file])
        enemy_pawns = bool(board.pieces_mask(chess.PAWN, not color) & chess.BB_FILES[file])
        score -= 16.0 if not own_pawns and not enemy_pawns else 8.0 if not own_pawns else 0.0

    zone = chess.BB_KING_ATTACKS[king] | chess.BB_SQUARES[king]
    attackers = 0
    tropism = {chess.KNIGHT: 1.5, chess.BISHOP: 1.0, chess.ROOK: 1.0, chess.QUEEN: 2.0}
    for piece_type, weight in tropism.items():
        for square in _squares(board, piece_type, not color):
            if board.attacks(square) & zone:
                attackers += 1
            distance = max(
                abs(chess.square_file(square) - king_file),
                abs(chess.square_rank(square) - king_rank),
            )
            score -= max(0, 6 - distance) * weight
    score -= attackers * attackers * 5.0
    score += 8.0 if board.has_kingside_castling_rights(color) else 0.0
    score += 5.0 if board.has_queenside_castling_rights(color) else 0.0
    return score


def evaluate_knight(board: chess.Board, color: chess.Color) -> float:
    score = 0.0
    own_pawns = board.pieces_mask(chess.PAWN, color)
    enemy_pawns = board.pieces_mask(chess.PAWN, not color)
    for square in _squares(board, chess.KNIGHT, color):
        protected = bool(board.attackers(color, square) & own_pawns)
        challenged = bool(board.attackers(not color, square) & enemy_pawns)
        if 3 <= _relative_rank(square, color) <= 5 and protected and not challenged:
            score += 22.0
    return score


def evaluate_bishop(board: chess.Board, color: chess.Color) -> float:
    score = 0.0
    pawns = _squares(board, chess.PAWN, color)
    for square in _squares(board, chess.BISHOP, color):
        square_color = (chess.square_file(square) + chess.square_rank(square)) & 1
        score -= 3.0 * sum(
            ((chess.square_file(pawn) + chess.square_rank(pawn)) & 1) == square_color
            for pawn in pawns
        )
    return score


def evaluate_rook(board: chess.Board, color: chess.Color) -> float:
    score = 0.0
    own_pawns = board.pieces_mask(chess.PAWN, color)
    enemy_pawns = board.pieces_mask(chess.PAWN, not color)
    for square in _squares(board, chess.ROOK, color):
        file = chess.square_file(square)
        own_on_file = bool(own_pawns & chess.BB_FILES[file])
        enemy_on_file = bool(enemy_pawns & chess.BB_FILES[file])
        score += 18.0 if not own_on_file and not enemy_on_file else 9.0 if not own_on_file else 0.0
        score += 20.0 if _relative_rank(square, color) == 6 else 0.0
    return score


def evaluate_queen(board: chess.Board, color: chess.Color) -> float:
    del board, color
    return 0.0  # Queen activity is covered by mobility and the piece-square table.


def evaluate_space(board: chess.Board, color: chess.Color) -> float:
    controlled = 0
    for piece_type in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
        for square in _squares(board, piece_type, color):
            for target in chess.scan_forward(int(board.attacks(square))):
                rank = chess.square_rank(target)
                in_enemy_half = (color == chess.WHITE and rank >= 3) or (
                    color == chess.BLACK and rank <= 4
                )
                if 2 <= chess.square_file(target) <= 5 and in_enemy_half:
                    controlled += 1
    return controlled * 1.25


def evaluate_trapped_pieces(
    board: chess.Board, color: chess.Color, mobility: dict[chess.Square, int] | None = None
) -> float:
    if mobility is None:
        mobility = _legal_mobility(board, color)
    penalties = {
        chess.KNIGHT: (2, 12.0),
        chess.BISHOP: (3, 10.0),
        chess.ROOK: (3, 8.0),
        chess.QUEEN: (4, 6.0),
    }
    score = 0.0
    for square, count in mobility.items():
        piece = board.piece_at(square)
        if piece is not None and piece.piece_type in penalties:
            minimum, penalty = penalties[piece.piece_type]
            score -= penalty * max(0, minimum - count)
    return score


PIECE_HANDLER = {
    chess.PAWN: evaluate_pawn, chess.KNIGHT: evaluate_knight, chess.BISHOP: evaluate_bishop,
    chess.ROOK: evaluate_rook, chess.QUEEN: evaluate_queen, chess.KING: evaluate_king,
}


def evaluate(board: chess.Board, config: Config) -> float:
    """Score the position in centipawns from the point of view of the side to move."""
    middle_game = endgame = positional = 0.0
    phase = 0
    for piece_type in chess.PIECE_TYPES:
        middle_squares = config.MIDDLE_GAME_TABLE[piece_type]
        end_squares = config.ENDGAME_TABLE[piece_type]
        for color in (chess.WHITE, chess.BLACK):
            for square in _squares(board, piece_type, color):
                middle_game += middle_squares[color][square]
                endgame += end_squares[color][square]
                phase += config.PHASE_WEIGHT[piece_type]
            side_score = PIECE_HANDLER[piece_type](board, color)
            positional += side_score if color == chess.WHITE else -side_score

    for color in (chess.WHITE, chess.BLACK):
        mobility = _legal_mobility(board, color)
        side_score = (
            evaluate_mobility(board, color, mobility)
            + evaluate_space(board, color)
            + evaluate_trapped_pieces(board, color, mobility)
        )
        positional += side_score if color == chess.WHITE else -side_score

    blend = min(phase, config.TOTAL_PHASE) / config.TOTAL_PHASE
    score = middle_game * blend + endgame * (1.0 - blend) + positional
    return score if board.turn == chess.WHITE else -score
