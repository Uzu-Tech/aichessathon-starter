import math
import random
import json
import sys

import chess
from chess.polyglot import zobrist_hash
from collections import Counter
from ordering import sorted_moves, order_score
from timing import SearchTimeout, get_budget_ms, check_time, get_deadline
from evaluation import evaluate
from results import SearchResult
from utils import MATE
from transposition import TranspositionTable, EXACT, LOWER_BOUND, UPPER_BOUND

INF = float('inf')
MAX_DEPTH = 10

board_state_counts = Counter()
tt = TranspositionTable()

def quiescence_search(board: chess.Board, alpha: float, beta: float, ply: int, deadline: float, result: SearchResult):
    check_time(result.nodes, deadline)
    
    result.nodes += 1
    stand_pat = evaluate(board)
    
    if stand_pat >= beta: # alpha - beta cutoff
        result.cutoffs += 1
        return beta
    
    best_score = stand_pat
    alpha = max(alpha, stand_pat)
    
    captures = sorted(
        board.generate_legal_captures(),
        key=lambda m: order_score(board, m, tt_move=None),
        reverse=True
    )

    for move in captures:
        board.push(move)
        try:
            score = -quiescence_search(board, -beta, -alpha, ply + 1, deadline, result)
        finally:
            board.pop()

        if score >= beta:
            result.cutoffs += 1
            return beta

        alpha = max(alpha, score)

    return alpha


def negamax(
    board: chess.Board, alpha: float, beta: float, depth: int, ply: int, deadline: float, result: SearchResult
) -> float:
    check_time(result.nodes, deadline)
    result.nodes += 1

    current_hash = zobrist_hash(board)
    if board.is_insufficient_material() or board_state_counts[current_hash] >= 2:
        return 0.0

    # TT Probe, find the prev best move and score bound at this board
    tt_score, tt_move = tt.probe(board, alpha, beta, depth, ply)
    if tt_score is not None and ply > 0:
        result.cutoffs += 1
        return tt_score

    if depth == 0:
        return quiescence_search(board, alpha, beta, ply, deadline, result)

    moves = sorted_moves(board, tt_move)
    
    # If no moves must be checkmate or stalemate, adding in ply count so we look for faster mates
    if not moves:
        return -(MATE - ply) if board.is_check() else 0.0

    orig_alpha = alpha
    best_score = -INF
    best_move = moves[0]

    for move in moves:
        board.push(move)
        next_hash = zobrist_hash(board)
        board_state_counts[next_hash] += 1

        try:
            score = -negamax(
                board, alpha=-beta, beta=-alpha, depth=depth - 1, ply=ply + 1,
                deadline=deadline, result=result
            )
        finally:
            board_state_counts[next_hash] -= 1
            board.pop()

        if score > best_score:
            best_score = score
            best_move = move

            # Update root-level results
            if ply == 0:
                result.best_score = best_score
                result.best_move = best_move

        alpha = max(alpha, score)

        # Beta Cutoff
        if alpha >= beta:
            tt.store(board, depth, ply, move, best_score, bound=LOWER_BOUND)
            result.cutoffs += 1
            return best_score

    # Store TT Entry
    bound = EXACT if best_score > orig_alpha else UPPER_BOUND
    tt.store(board, depth, ply, best_move, best_score, bound)

    return best_score


def get_move(fen: str, time_left_ms: int) -> str:
    board = chess.Board(fen)
    current_hash = zobrist_hash(board)
    board_state_counts[current_hash] += 1  
    
    budget_ms = min(get_budget_ms(board, time_left_ms), time_left_ms * 0.95)
    deadline = get_deadline(budget_ms)
    results = []
    
    best_move = next(iter(board.legal_moves))
    
    for depth in range(1, MAX_DEPTH + 1):
        try:
            result = SearchResult(depth=depth, budget_ms=budget_ms)
            negamax(
                board, alpha=-INF, beta=INF, depth=depth,
                ply=0, deadline=deadline, result=result
            )
            if result.best_move is not None:   # <-- only trust it if it was actually set
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