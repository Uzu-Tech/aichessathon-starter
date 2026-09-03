from time import perf_counter
from user.stats import SearchResult
from agent import evaluate, MAX_DEPTH, search_root, SearchTimeout
from user.test_suite import TEST_POSITIONS
import chess

def benchmark_search(
    fen: str,
    budgets_ms: list[int] = [250, 500, 1000, 2000, 5000],
):
    print(f"FEN: {fen}\n")

    for budget_ms in budgets_ms:
        board = chess.Board(fen)

        # Always have a legal fallback move
        best_move = next(iter(board.legal_moves))
        deadline = perf_counter() + budget_ms / 1000

        result = SearchResult(
            depth=0,
            budget_ms=budget_ms,
        )

        start = perf_counter()
        completed_depth = 0
        best_score = -float('inf')

        for depth in range(1, MAX_DEPTH + 1):
            result.depth = depth

            try:
                search_root(
                    board,
                    depth,
                    deadline,
                    result,
                )

                # Only update the move after the entire depth
                # successfully completed.
                best_move = result.best_move
                best_score = result.best_score
                completed_depth = depth

            except SearchTimeout:
                break

        elapsed_ms = (perf_counter() - start) * 1000

        nps = (
            result.nodes / (elapsed_ms / 1000)
            if elapsed_ms > 0
            else 0
        )
        
        print(
            f"{budget_ms:>5} ms | "
            f"depth {completed_depth:<2} | "
            f"move {best_move.uci():<5} | "  # type: ignore
            f"score {best_score:>7.1f} | "
            f"nodes {result.nodes:>9,} | "
            f"leaves {result.leaves:>9,} | "
            f"cutoffs {result.cutoffs:>8,} | "
            f"NPS {nps:>9,.0f} | "
            f"actual {elapsed_ms:>7.1f} ms"
        )
                
    
if __name__ == "__main__":
    fen = "r2q1rk1/ppp1bppp/2n1pn2/8/2B1P3/2N1BN2/PPP2PPP/2RQR1K1 b - - 0 1"
    benchmark_search(fen)