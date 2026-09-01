import math
import random

import chess

PIECE_VALUE = {
    chess.PAWN: 100.0,
    chess.KNIGHT: 320.0,
    chess.BISHOP: 330.0,
    chess.ROOK: 500.0,
    chess.QUEEN: 900.0,
}
MOBILITY_WEIGHT = 4.0
MATE = 1e6
INF = float('inf')


def evaluate(board: chess.Board, mobility: int) -> float:
    mover = board.turn
    material = sum(
        value * (len(board.pieces(piece, mover)) - len(board.pieces(piece, not mover)))
        for piece, value in PIECE_VALUE.items()
    )
    return material + MOBILITY_WEIGHT * mobility


def negamax(board: chess.Board, alpha: float, beta: float, depth: int) -> float:
    moves = list(board.legal_moves)
    if not moves:
        return -MATE if board.is_check() else 0.0

    if board.is_insufficient_material() or board.can_claim_draw():
        return 0.0
    
    if depth == 0:
        return evaluate(board, len(moves))
    
    best_score = -INF
    
    for move in moves:
        board.push(move)
        score = -negamax(board, -beta, -alpha, depth - 1)
        board.pop()
        best_score = max(best_score, score)
        alpha = max(alpha, score)
        
        if alpha >= beta: 
            break
        
    return best_score


def get_move(fen: str, time_left_ms: int) -> str:
    board = chess.Board(fen)
    best_score = -math.inf
    best: list[chess.Move] = []
    for move in board.legal_moves:
        board.push(move)
        score = -negamax(board, -INF, INF, 1)
        board.pop()
        if score > best_score:
            best_score = score
            best = [move]
        elif score == best_score:
            best.append(move)
    return random.choice(best).uci()
