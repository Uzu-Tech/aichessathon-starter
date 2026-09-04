"""player.py — play a game against your own agent from the terminal.

    python player.py                     # you are White
    python player.py --black             # you are Black, the engine opens
    python player.py --fen "8/5k2/..."   # start from a position

Enter moves in SAN (e4, Nf3, O-O, exd5, e8=Q) or UCI (e2e4, e7e8q). Type `help`
for the other commands.

The engine's clock really counts down and is handed to get_move, so this drives
the same time management the platform will. Its per-move stats line goes to
stderr, which this captures and prints alongside the move rather than letting it
scribble over the board.
"""

import argparse
import json
from contextlib import redirect_stderr
from io import StringIO
from time import perf_counter

import chess

import agent
from config import PIECE_VALUE
from evaluate import evaluate
from user.stats import SearchResult

HELP = """
  <move>    play a move, in SAN (Nf3, e4, O-O, exd5) or UCI (g1f3, e7e8q)
  moves     list every legal move in this position
  eval      static evaluation of the position, before any search
  score     search the position and print what the engine thinks it is worth
  score N   the same, given N milliseconds to think (default 1000)
  fen       print the current FEN
  undo      take back your last move and the engine's reply
  flip      view the board from the other side
  help      this list
  quit      resign and exit
"""


COMMANDS = {"moves", "eval", "score", "fen", "undo", "flip", "help", "quit", "exit"}


def material_balance(board: chess.Board) -> float:
    """Centipawns of material, White minus Black. Kings excluded; they never trade."""
    total = 0.0
    for piece_type, value in PIECE_VALUE.items():
        if piece_type == chess.KING:
            continue
        total += value * (
            len(board.pieces(piece_type, chess.WHITE)) - len(board.pieces(piece_type, chess.BLACK))
        )
    return total


def format_score(score: float, board: chess.Board) -> str:
    """Render a search score from White's point of view, or as a mate distance."""
    # negamax returns the score for the side to move; flip it so + always means White.
    white_pov = score if board.turn == chess.WHITE else -score

    if abs(white_pov) > agent.MATE - 10_000:
        plies = agent.MATE - abs(white_pov)
        moves = int((plies + 1) // 2)
        winner = "White" if white_pov > 0 else "Black"
        return f"mate in {moves} for {winner}"

    leader = "White" if white_pov > 0 else "Black" if white_pov < 0 else "level"
    if leader == "level":
        return "0.00 (level)"
    return f"{white_pov / 100:+.2f} ({leader} better)"


def search_score(board: chess.Board, budget_ms: float) -> tuple[chess.Move | None, float, int, int]:
    """Run the agent's search on a copy and report (best move, score, depth, nodes).

    The search is given its own board and its own copy of the repetition counter:
    a SearchTimeout unwinds out of negamax without popping the moves it pushed or
    decrementing the counts it made, so letting it loose on the live game would
    corrupt the position you are playing.
    """
    probe = board.copy()
    saved = agent.board_state_counts.copy()

    deadline = perf_counter() + budget_ms / 1000
    best_move: chess.Move | None = None
    best_score = 0.0
    completed = 0
    nodes = 0

    for depth in range(1, agent.MAX_DEPTH + 1):
        result = SearchResult(depth=depth, budget_ms=budget_ms)
        try:
            agent.search_root(probe, depth, deadline, result)
        except agent.SearchTimeout:
            nodes += result.nodes
            break
        # Only trust a depth that finished; a cut-off one has seen too few moves.
        best_move, best_score, completed = result.best_move, result.best_score, depth
        nodes += result.nodes

    agent.board_state_counts.clear()
    agent.board_state_counts.update(saved)
    return best_move, best_score, completed, nodes


def render(board: chess.Board, orientation: chess.Color) -> str:
    """An ASCII board with rank/file labels, seen from `orientation`'s side."""
    ranks = range(7, -1, -1) if orientation == chess.WHITE else range(8)
    files = range(8) if orientation == chess.WHITE else range(7, -1, -1)

    last = board.peek() if board.move_stack else None
    lines = []
    for rank in ranks:
        cells = []
        for file in files:
            square = chess.square(file, rank)
            piece = board.piece_at(square)
            symbol = piece.symbol() if piece else "."
            # Bracket the square the last move landed on so it is easy to spot.
            if last is not None and square == last.to_square:
                cells.append(f"[{symbol}]")
            else:
                cells.append(f" {symbol} ")
        lines.append(f" {rank + 1} " + "".join(cells))
    lines.append("    " + "".join(f" {chess.FILE_NAMES[f]} " for f in files))
    return "\n".join(lines)


def parse_stats(stderr_text: str) -> dict[str, int]:
    """Pull the last {"stats": true, ...} line the agent wrote, if any."""
    found: dict[str, int] = {}
    for line in stderr_text.splitlines():
        try:
            obj = json.loads(line.strip())
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and obj.get("stats") is True:
            found = obj
    return found


def outcome_text(board: chess.Board) -> str:
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return "game still in progress"
    reason = outcome.termination.name.replace("_", " ").lower()
    result = board.result(claim_draw=True)
    if outcome.winner is None:
        return f"draw by {reason} ({result})"
    return f"{'White' if outcome.winner == chess.WHITE else 'Black'} wins by {reason} ({result})"


def read_move(board: chess.Board, prompt: str) -> chess.Move | str:
    """Return a legal Move, or a lowercase command word."""
    while True:
        try:
            text = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            return "quit"
        if not text:
            continue
        if text.split()[0].lower() in COMMANDS:
            return text.lower()
        for parse in (board.parse_san, board.parse_uci):
            try:
                return parse(text)
            except ValueError:
                continue
        print(f"  '{text}' is not a legal move here. Type `moves` to see what is.")


def engine_move(board: chess.Board, clock_ms: float) -> tuple[chess.Move | None, float, str]:
    """Ask the agent for a move. Returns (move, elapsed_ms, note)."""
    captured = StringIO()
    start = perf_counter()
    try:
        with redirect_stderr(captured):
            uci = agent.get_move(board.fen(), int(clock_ms))
    except Exception as error:  # a crash here would forfeit the game on the platform
        note = f"agent raised {type(error).__name__}: {error}"
        return None, (perf_counter() - start) * 1000, note
    elapsed = (perf_counter() - start) * 1000

    try:
        move = chess.Move.from_uci(uci)
    except ValueError:
        return None, elapsed, f"agent returned malformed UCI: {uci!r}"
    if move not in board.legal_moves:
        return None, elapsed, f"agent returned an illegal move: {uci!r}"

    stats = parse_stats(captured.getvalue())
    note = ""
    if stats:
        note = (
            f"depth {stats.get('depth', 0)}  "
            f"nodes {stats.get('nodes', 0):,}  "
            f"cutoffs {stats.get('cutoffs', 0):,}"
        )
    return move, elapsed, note


def play(you: chess.Color, fen: str | None, clock_ms: float, increment_ms: float) -> None:
    board = chess.Board(fen) if fen else chess.Board()
    orientation = you

    # board_state_counts is module-global and persists for the life of the process,
    # so clear it or a previous game's positions count as repetitions in this one.
    agent.board_state_counts.clear()

    print(f"\nYou are {'White' if you == chess.WHITE else 'Black'}.  "
          f"Engine clock {clock_ms / 1000:.0f}s +{increment_ms / 1000:g}s.")
    print("Type `help` for commands.")

    while board.outcome(claim_draw=True) is None:
        print()
        print(render(board, orientation))
        turn = "White" if board.turn == chess.WHITE else "Black"
        check = "  — CHECK" if board.is_check() else ""
        print(f"\n  move {board.fullmove_number}, {turn} to play{check}")

        if board.turn == you:
            choice = read_move(board, "  your move > ")

            if isinstance(choice, str):
                name, _, argument = choice.partition(" ")
                if name in {"quit", "exit"}:
                    print("\n  you resigned.")
                    return
                if name == "help":
                    print(HELP)
                elif name == "fen":
                    print(f"  {board.fen()}")
                elif name == "flip":
                    orientation = not orientation
                elif name == "eval":
                    static = evaluate(board)
                    white_pov = static if board.turn == chess.WHITE else -static
                    print(f"  static {white_pov / 100:+.2f} from White's side "
                          f"({white_pov:+.1f} cp, no search)")
                elif name == "score":
                    budget = float(argument) if argument.strip().isdigit() else 1000.0
                    print(f"  thinking for {budget:.0f}ms...")
                    move, raw, depth, nodes = search_score(board, budget)
                    if move is None:
                        print("  no depth completed in that time — give it more ms.")
                    else:
                        print(f"  {format_score(raw, board)}   best {board.san(move)}   "
                              f"depth {depth}   nodes {nodes:,}")
                    print(f"  material {material_balance(board) / 100:+.2f} from White's side")
                elif name == "moves":
                    print("  " + " ".join(sorted(board.san(m) for m in board.legal_moves)))
                elif name == "undo":
                    if not board.move_stack:
                        print("  nothing to take back.")
                    else:
                        board.pop()                      # the engine's reply
                        if board.move_stack and board.turn != you:
                            board.pop()                  # and your move before it
                        print("  taken back.")
                continue

            print(f"  you play {board.san(choice)}")
            board.push(choice)
            continue

        move, elapsed, note = engine_move(board, clock_ms)
        if move is None:
            print(f"\n  !! {note}")
            print("  on the platform this would lose the game.")
            return

        clock_ms = clock_ms - elapsed + increment_ms
        san = board.san(move)
        print(f"  engine plays {san} ({move.uci()})  in {elapsed:.0f}ms, "
              f"{clock_ms / 1000:.1f}s left")
        if note:
            print(f"    {note}")
        board.push(move)

        if clock_ms <= 0:
            print("\n  !! the engine flagged — on the platform this would lose the game.")
            return

    print()
    print(render(board, orientation))
    print(f"\n  {outcome_text(board)}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Play a game against agent.py in the terminal.")
    parser.add_argument("--black", action="store_true", help="play Black; the engine opens")
    parser.add_argument("--fen", default=None, help="start from this FEN")
    parser.add_argument("--time-ms", type=int, default=120_000, help="engine's starting clock")
    parser.add_argument("--increment-ms", type=int, default=500, help="engine's per-move increment")
    arguments = parser.parse_args()

    play(
        you=chess.BLACK if arguments.black else chess.WHITE,
        fen=arguments.fen,
        clock_ms=arguments.time_ms,
        increment_ms=arguments.increment_ms,
    )


if __name__ == "__main__":
    main()
