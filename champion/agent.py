import math
import random
import json
import sys

import chess
import chess.polyglot
from collections import Counter
from ordering import sorted_moves
from timing import SearchTimeout, get_budget_ms, check_time, get_deadline
from evaluation import evaluate
from results import SearchResult

MATE = 1e6
INF = float('inf')
MAX_DEPTH = 10

board_state_counts = Counter()

def quiescence_search(board: chess.Board, alpha: float, beta: float, ply: int, deadline: float, result: SearchResult):
    check_time(result.nodes, deadline)
    
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


def negamax(
    board: chess.Board, alpha: float, beta: float, depth: int, ply: int, deadline: float, result: SearchResult
) -> float:
    check_time(result.nodes, deadline)
        
    current_hash = chess.polyglot.zobrist_hash(board)
    result.nodes += 1
    if board.is_insufficient_material() or board_state_counts[current_hash] >= 2:
        return 0.0

    if depth == 0:
        return quiescence_search(board, alpha, beta, ply, deadline, result)

    moves = sorted_moves(board)

    if not moves:
        return -(MATE - ply) if board.is_check() else 0.0

    best_score = -INF
    for move in moves:
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
    
    moves = sorted_moves(board)
    best = moves[0] # Assuming we'll always have moves to play at the start
    best_score = -INF
    
    alpha = -INF
    beta = INF
    
    for move in moves:
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
    
    budget_ms = min(get_budget_ms(board, time_left_ms), time_left_ms * 0.95)
    deadline = get_deadline(budget_ms)
    results = []
    
    best_move = next(iter(board.legal_moves))
    
    for depth in range(1, MAX_DEPTH + 1):
        try:
            result = SearchResult(depth=depth, budget_ms=budget_ms)
            search_root(board, depth, deadline, result)
            best_move = result.best_move
            results.append(result)
            
        except SearchTimeout:
            break
    return best_move.uci() # type: ignore