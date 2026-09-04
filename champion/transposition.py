import chess
from chess.polyglot import zobrist_hash
from utils import MATE

EXACT = 0
UPPER_BOUND = 1
LOWER_BOUND = -1

class TranspositionTable:
    def __init__(self):
        self.tt = {}
        
    def store(self, board: chess.Board, depth: int, ply: int, move: chess.Move, score: float, bound: int):
        board_hash = zobrist_hash(board)
        if board_hash in self.tt and depth < self.tt[board_hash]['depth']:
            return
        
        if score > MATE - 1000:
            score += ply
        elif score < -MATE + 1000:
            score -= ply
        
        self.tt[board_hash] = {
            'depth': depth,
            'move': move,
            'score': score,
            'bound': bound
        }
        
    def probe(self, board: chess.Board, alpha: float, beta: float, depth: int, ply: int):
        board_hash = zobrist_hash(board)
        
        if board_hash not in self.tt:
            return None, None
        
        entry = self.tt[board_hash]
        score = entry['score']
        move = entry['move']
        bound = entry['bound']
        
        if score > MATE - 1000:
            score -= ply
        elif score < -MATE + 1000:
            score += ply
        
        if entry['depth'] < depth:
            return None, move
            
        if (
            bound == EXACT
            or (bound == UPPER_BOUND and score <= alpha)
            or (bound == LOWER_BOUND and score >= beta)
        ):
            return score, move
        
        return None, move