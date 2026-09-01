import math
import random

import chess
from stats import SearchStats
from time import perf_counter

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
SEARCH_DEPTH = 5


def evaluate(board: chess.Board, mobility: int) -> float:
    mover = board.turn
    material = sum(
        value * (len(board.pieces(piece, mover)) - len(board.pieces(piece, not mover)))
        for piece, value in PIECE_VALUE.items()
    )
    return material + MOBILITY_WEIGHT * mobility


def negamax(board: chess.Board, alpha: float, beta: float, depth: int, stats: SearchStats,) -> float:
    stats.node(depth)
    
    if board.is_insufficient_material():
        stats.leaves += 1
        return 0.0

    if depth == 0:
        # We need to detect mate at the horizon.
        if board.is_checkmate():
            stats.leaves += 1
            return -MATE

        stats.leaves += 1
        return evaluate(board, 0)

    
    best_score = -INF

    found_move = False
    for move in board.legal_moves:
        found_move = True
        board.push(move)
        score = -negamax(board, -beta, -alpha, depth - 1, stats)
        board.pop()
        best_score = max(best_score, score)
        alpha = max(alpha, score)
        
        if alpha >= beta: 
            stats.cutoffs += 1
            break
    
    if not found_move:
        stats.leaves += 1
        return -MATE if board.is_check() else 0.0
        
    return best_score


def get_move(fen: str, time_left_ms: int) -> str:
    board = chess.Board(fen)
    
    stats = SearchStats()
    start = perf_counter()
    
    best_score = -math.inf
    best: list[chess.Move] = []
    
    for move in board.legal_moves:
        board.push(move)
        score = -negamax(board, -INF, INF, SEARCH_DEPTH - 1, stats)
        board.pop()
        if score > best_score:
            best_score = score
            best = [move]
        elif score == best_score:
            best.append(move)
            
    elapsed = perf_counter() - start
    stats.report(elapsed)
    
    return random.choice(best).uci()

def get_move_depth(fen: str, depth: int) -> str:
    board = chess.Board(fen)
    
    stats = SearchStats()
    start = perf_counter()
    
    best_score = -math.inf
    best: list[chess.Move] = []
    
    for move in board.legal_moves:
        board.push(move)
        score = -negamax(board, -INF, INF, depth - 1, stats)
        board.pop()
        if score > best_score:
            best_score = score
            best = [move]
        elif score == best_score:
            best.append(move)
            
    elapsed = perf_counter() - start
    stats.report(elapsed)
    
    return random.choice(best).uci()
