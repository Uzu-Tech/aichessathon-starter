import time
from agent import get_move_depth, evaluate
from test_suite import TEST_POSITIONS
import chess


def benchmark_operations(fen, iterations=100_000):
    board = chess.Board(fen)

    moves = list(board.legal_moves)
    mobility = len(moves)

    # Pure legal move generation
    start = time.perf_counter()

    for _ in range(iterations):
        list(board.legal_moves)

    move_time = time.perf_counter() - start

    # Push/pop
    move = moves[0]

    start = time.perf_counter()

    for _ in range(iterations):
        board.push(move)
        board.pop()

    push_pop_time = time.perf_counter() - start

    # Pure evaluation, using precomputed mobility
    start = time.perf_counter()

    for _ in range(iterations):
        evaluate(board, mobility)

    eval_time = time.perf_counter() - start

    print(f"\nMicrobenchmark: {iterations:,} iterations")
    print(f"legal_moves : {move_time * 1000:.2f} ms")
    print(f"push/pop    : {push_pop_time * 1000:.2f} ms")
    print(f"evaluate    : {eval_time * 1000:.2f} ms")

def run_test_suite(depths=(4, )):

    print("=" * 100)
    print("CHESS ENGINE TEST SUITE")
    print("=" * 100)

    for depth in depths:
        print(f"\n{'=' * 100}")
        print(f"DEPTH {depth}")
        print(f"{'=' * 100}")

        total_time = 0.0
        total_nodes = 0

        for name, fen in TEST_POSITIONS.items():
            start = time.perf_counter()

            move = get_move_depth(
                fen,
                depth
            )

            elapsed = time.perf_counter() - start
            total_time += elapsed

            print(
                f"{name:<22} "
                f"{move:<6} "
            )

        print("-" * 100)
        print(
            f"TOTAL: {total_time:.3f} sec | "
            f"AVERAGE: {total_time / len(TEST_POSITIONS) * 1000:.2f} ms"
        )

    
if __name__ == "__main__":
    run_test_suite()