import math
import random
import json
import sys

import chess
import chess.polyglot
from user.stats import SearchResult
from time import perf_counter
from collections import Counter

from config import PIECE_VALUE
from evaluate import evaluate

MATE = 1e6
INF = float('inf')
NODES_PER_TIME_CHECK = 1 << 6 # Power of 2
MAX_DEPTH = 10

board_state_counts = Counter()

class SearchTimeout(Exception):
    pass

# Example time management for now will improve later
def get_budget_ms(board: chess.Board, time_left_ms):
    ply_count = board.ply()

    if ply_count < 20:
        moves_left_estimate = 40
    elif ply_count < 60:
        moves_left_estimate = 30
    elif ply_count < 120:
        moves_left_estimate = 20
    else:
        moves_left_estimate = 15

    budget_ms = time_left_ms / max(20, moves_left_estimate)

    budget_ms = min(budget_ms, time_left_ms * 0.33)

    if time_left_ms < 5000:
        budget_ms = min(budget_ms, time_left_ms * 0.05)

    return max(budget_ms, 50)  # never budget below 20ms

def quiescence_search(board: chess.Board, alpha: float, beta: float, ply: int, deadline: float, result: SearchResult):
    if result.nodes & (NODES_PER_TIME_CHECK - 1) == 0:
        if perf_counter() >= deadline:
            raise SearchTimeout
    
    result.nodes += 1
    stand_pat = evaluate(board)
    
    if stand_pat >= beta: # alpha - beta cutoff
        result.cutoffs += 1
        return stand_pat
    
    best_score = stand_pat
    alpha = max(alpha, stand_pat)

    for move in board.legal_moves:
        if not board.is_capture(move):
            continue
        board.push(move)
        score = -quiescence_search(board, -beta, -alpha, ply + 1, deadline, result)
        board.pop()
        
        alpha = max(alpha, score)
        best_score = max(score, best_score)
        
        if alpha >= beta: 
                result.cutoffs += 1
                break
        
    return best_score


def get_capture_pieces(board: chess.Board, move: chess.Move):
    attacker = board.piece_at(move.from_square).piece_type # type: ignore
    if board.is_en_passant(move):
        victim = chess.PAWN
    else:
        victim = board.piece_at(move.to_square).piece_type # type: ignore
    return attacker, victim

def order_score(board: chess.Board, move: chess.Move):
    if board.is_capture(move):
        attacker, victim = get_capture_pieces(board, move)
        return 100_000 + (PIECE_VALUE[victim] * 10 - PIECE_VALUE[attacker]) # type: ignore

    return 50_000

def negamax(
    board: chess.Board, alpha: float, beta: float, depth: int, ply: int, deadline: float, result: SearchResult
) -> float:
    if result.nodes & (NODES_PER_TIME_CHECK - 1) == 0:
        if perf_counter() >= deadline:
            raise SearchTimeout
        
    current_hash = chess.polyglot.zobrist_hash(board)
    result.nodes += 1
    if board.is_insufficient_material() or board_state_counts[current_hash] >= 2:
        result.leaves += 1
        return 0.0

    if depth == 0:
        return quiescence_search(board, alpha, beta, ply, deadline, result)

    moves = list(board.legal_moves)

    if not moves:
        result.leaves += 1
        return -(MATE - ply) if board.is_check() else 0.0

    best_score = -INF
    for move in sorted(moves, key=lambda m: order_score(board, m), reverse=True):
        board.push(move)
        current_hash = chess.polyglot.zobrist_hash(board)
        board_state_counts[current_hash] += 1
        score = -negamax(
            board, 
            alpha=-beta, 
            beta=-alpha, 
            depth=depth - 1, 
            ply=ply + 1, 
            deadline=deadline,
            result=result
        )
        board_state_counts[current_hash] -= 1
        board.pop()
        best_score = max(best_score, score)
        alpha = max(alpha, score)
        
        if alpha >= beta: 
            result.cutoffs += 1
            break
        
    return best_score


def search_root(board: chess.Board, depth: int, deadline: float, result: SearchResult):
    result.nodes += 1
    
    moves = list(board.legal_moves)
    best = moves[0] # Assuming we'll always have moves to play at the start
    best_score = -INF
    
    alpha = -INF
    beta = INF
    for move in sorted(moves, key=lambda m: order_score(board, m), reverse=True):
        board.push(move)
        current_hash = chess.polyglot.zobrist_hash(board)
        board_state_counts[current_hash] += 1
        score = -negamax(
            board, 
            alpha=-beta, 
            beta=-alpha, 
            depth=depth - 1, 
            ply=1, 
            deadline=deadline,
            result=result
        )
        board_state_counts[current_hash] -= 1
        board.pop()
                
        alpha = max(score, alpha)
        if score > best_score:
            best_score = score
            best = move
    
    result.best_score = best_score
    result.best_move = best


def get_move(fen: str, time_left_ms: int) -> str:
    board = chess.Board(fen)
    current_hash = chess.polyglot.zobrist_hash(board)
    board_state_counts[current_hash] += 1
    
    budget = min(get_budget_ms(board, time_left_ms), time_left_ms * 0.95)
    deadline = perf_counter() + budget / 1000
    results = []
    
    best_move = next(iter(board.legal_moves))
    
    for depth in range(1, MAX_DEPTH + 1):
        try:
            result = SearchResult(depth=depth, budget_ms=budget)
            search_root(board, depth, deadline, result)
            best_move = result.best_move
            results.append(result)
            
        except SearchTimeout:
            break
            
    total_nodes = sum(r.nodes for r in results)
    total_cutoffs = sum(r.cutoffs for r in results)
    final_depth = results[-1].depth if results else 0

    print(json.dumps({
        "stats": True,
        "ply": board.ply(),
        "depth": final_depth,
        "nodes": total_nodes,
        "cutoffs": total_cutoffs,
    }), file=sys.stderr, flush=True)
    return best_move.uci() # type: ignore