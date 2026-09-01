"""
board_viewer.py — visualize positions and play against your agent locally.

Two ways to use this:

1. In a Jupyter/IPython notebook (recommended — boards render inline):

    from board_viewer import play_interactive_visual
    from agent import get_move   # your actual agent.py

    play_interactive_visual(get_move)

2. From a plain terminal (no images, ASCII board instead):

    python board_viewer.py

Requires only python-chess, which is already in the competition's base image
(and standard in any dev environment where you have python-chess installed).
"""

import chess

try:
    import chess.svg
    _HAS_SVG = True
except ImportError:
    _HAS_SVG = False

try:
    from IPython.display import display, clear_output
    _IN_NOTEBOOK = True
except ImportError:
    _IN_NOTEBOOK = False


# ---------------------------------------------------------------------------
# Static rendering helpers
# ---------------------------------------------------------------------------

def render(board: chess.Board, size: int = 400, arrows=None, squares=None):
    """
    Return an SVG string of the given position.

    arrows:  list of (from_square, to_square) tuples, e.g. [(chess.E2, chess.E4)]
             — useful for visualizing candidate moves / move ordering.
    squares: a chess.SquareSet or iterable of squares to highlight.
    """
    if not _HAS_SVG:
        raise RuntimeError("chess.svg not available — check your python-chess install")

    kwargs = {"size": size}
    if arrows:
        kwargs["arrows"] = [chess.svg.Arrow(f, t, color="#15781B") for f, t in arrows] # type: ignore

    if squares is not None:
        kwargs["squares"] = squares

    if board.move_stack:
        kwargs["lastmove"] = board.peek() # type: ignore

    return chess.svg.board(board, **kwargs) # type: ignore


def save_position(board: chess.Board, path: str, size: int = 400, **kwargs):
    """Save a single position to an .svg file you can open in a browser."""
    svg = render(board, size=size, **kwargs)
    with open(path, "w") as f:
        f.write(svg)
    print(f"saved {path}")


def render_game(moves_uci, fen_start=None, size: int = 350):
    """
    Given a list of UCI move strings (e.g. from a logged game), return a list
    of SVG frames, one per position, so you can step through a completed game.
    """
    board = chess.Board(fen_start) if fen_start else chess.Board()
    frames = [render(board, size=size)]
    for uci in moves_uci:
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            raise ValueError(f"illegal move in game log: {uci} at ply {len(frames)}")
        board.push(move)
        frames.append(render(board, size=size))
    return frames


# ---------------------------------------------------------------------------
# Interactive play loops
# ---------------------------------------------------------------------------

def play_interactive_visual(get_move_fn, fen_start=None, you_play=chess.WHITE,
                             time_left_ms: int = 120_000, size: int = 400):
    """
    Play against your own agent inside a notebook, with the board rendered
    as an image after every move.

    get_move_fn: your agent's get_move(fen, time_left_ms) -> uci_str function
    fen_start:   optional starting FEN (default: standard start)
    you_play:    chess.WHITE or chess.BLACK
    time_left_ms: fake clock value passed to your agent each call
                  (does not actually decrement — for real clock testing use
                  the harness's `make play` / `make arena` instead)
    """
    if not _IN_NOTEBOOK:
        print("No IPython display available — falling back to ASCII loop.")
        return play_interactive_ascii(get_move_fn, fen_start, you_play, time_left_ms)

    board = chess.Board(fen_start) if fen_start else chess.Board()

    while not board.is_game_over():
        clear_output(wait=True) # type: ignore
        display(render(board, size=size)) # type: ignore
        print(f"Turn: {'White' if board.turn == chess.WHITE else 'Black'}   "
              f"({board.fullmove_number} moves played)")

        if board.turn == you_play:
            move_str = input("Your move (UCI, e.g. e2e4): ").strip()
            try:
                move = chess.Move.from_uci(move_str)
            except ValueError:
                print("Malformed UCI, try again.")
                continue
            if move not in board.legal_moves:
                print("Illegal move, try again.")
                continue
        else:
            move_str = get_move_fn(board.fen(), time_left_ms)
            try:
                move = chess.Move.from_uci(move_str)
            except ValueError:
                print(f"!! agent returned malformed move: {move_str!r}")
                return
            if move not in board.legal_moves:
                print(f"!! agent returned illegal move: {move_str!r}")
                return
            print(f"Engine plays: {move_str}")

        board.push(move)

    clear_output(wait=True) # type: ignore
    display(render(board, size=size)) # type: ignore
    print("Result:", board.result())
    print("Reason:", _game_over_reason(board))


def play_interactive_ascii(get_move_fn, fen_start=None, you_play=chess.WHITE,
                            time_left_ms: int = 120_000):
    """Same as play_interactive_visual but for a plain terminal, no images."""
    board = chess.Board(fen_start) if fen_start else chess.Board()

    while not board.is_game_over():
        print()
        print(board)
        print()
        print(f"Turn: {'White' if board.turn == chess.WHITE else 'Black'}")

        if board.turn == you_play:
            move_str = input("Your move (UCI, e.g. e2e4): ").strip()
            try:
                move = chess.Move.from_uci(move_str)
            except ValueError:
                print("Malformed UCI, try again.")
                continue
            if move not in board.legal_moves:
                print("Illegal move, try again.")
                continue
        else:
            move_str = get_move_fn(board.fen(), time_left_ms)
            try:
                move = chess.Move.from_uci(move_str)
            except ValueError:
                print(f"!! agent returned malformed move: {move_str!r}")
                return
            if move not in board.legal_moves:
                print(f"!! agent returned illegal move: {move_str!r}")
                return
            print(f"Engine plays: {move_str}")

        board.push(move)

    print()
    print(board)
    print()
    print("Result:", board.result())
    print("Reason:", _game_over_reason(board))


def watch_agent_vs_agent(get_move_white, get_move_black, fen_start=None,
                          time_left_ms: int = 120_000, size: int = 350,
                          pause_seconds: float = 0.0):
    """
    Watch two agents (or your agent vs. a baseline you've imported) play out
    a full game automatically in a notebook, one frame per move.
    Set pause_seconds > 0 to slow it down for easier viewing.
    """
    import time as _time

    if not _IN_NOTEBOOK:
        raise RuntimeError("watch_agent_vs_agent needs a notebook (IPython display)")

    board = chess.Board(fen_start) if fen_start else chess.Board()

    while not board.is_game_over():
        clear_output(wait=True) # type: ignore
        display(render(board, size=size)) # type: ignore

        move_fn = get_move_white if board.turn == chess.WHITE else get_move_black
        move_str = move_fn(board.fen(), time_left_ms)
        move = chess.Move.from_uci(move_str)

        if move not in board.legal_moves:
            print(f"!! {'White' if board.turn else 'Black'} returned illegal move: {move_str!r}")
            return board

        board.push(move)
        if pause_seconds:
            _time.sleep(pause_seconds)

    clear_output(wait=True) # type: ignore
    display(render(board, size=size)) # type: ignore
    print("Result:", board.result())
    print("Reason:", _game_over_reason(board))
    return board


def _game_over_reason(board: chess.Board) -> str:
    if board.is_checkmate():
        return "checkmate"
    if board.is_stalemate():
        return "stalemate"
    if board.is_insufficient_material():
        return "insufficient material"
    if board.can_claim_threefold_repetition():
        return "threefold repetition"
    if board.can_claim_fifty_moves():
        return "fifty-move rule"
    return "game over"


# ---------------------------------------------------------------------------
# Plain terminal entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import random

    def _random_agent(fen: str, time_left_ms: int) -> str:
        b = chess.Board(fen)
        return random.choice(list(b.legal_moves)).uci()

    print("No agent.py found in this quick demo — playing against a random mover.")
    print("Edit this file's __main__ block to import your real agent instead.\n")
    play_interactive_ascii(_random_agent)