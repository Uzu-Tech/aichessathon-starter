import chess
from time import perf_counter

NODES_PER_TIME_CHECK = 1 << 6 # Power of 2

class SearchTimeout(Exception):
    pass

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

    return max(budget_ms, 50)  # never budget below 50ms


def get_deadline(budget_ms):
    return perf_counter() + budget_ms / 1000

def check_time(nodes: int, deadline: float):
    if nodes & (NODES_PER_TIME_CHECK - 1) == 0:
            if perf_counter() >= deadline:
                raise SearchTimeout