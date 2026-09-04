import chess
from utils import PIECE_VALUE
from typing import Optional

def get_capture_pieces(board: chess.Board, move: chess.Move):
    attacker = board.piece_at(move.from_square).piece_type # type: ignore
    if board.is_en_passant(move):
        victim = chess.PAWN
    else:
        victim = board.piece_at(move.to_square).piece_type # type: ignore
    return attacker, victim

def order_score(board: chess.Board, move: chess.Move, tt_move: Optional[chess.Move]):
    if move == tt_move:
        return 1_000_000
    
    if board.is_capture(move):
        attacker, victim = get_capture_pieces(board, move)
        return 100_000 + (PIECE_VALUE[victim] * 10 - PIECE_VALUE[attacker]) # type: ignore

    return 50_000

def sorted_moves(board: chess.Board, tt_move: Optional[chess.Move]):
    return list(
        sorted(
            board.legal_moves, 
            key=lambda m: order_score(board, m, tt_move), 
            reverse=True
        )
    )
