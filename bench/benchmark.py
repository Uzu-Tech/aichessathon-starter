from __future__ import annotations

import argparse
import io
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import chess
import chess.pgn
import yaml

from harness.referee import FAILED_TERMINATIONS, play_match
from harness.rules import PLY_CAP
from harness.sandbox import local
from harness.stats import game_averages


def count_plies(pgn_text: str) -> int:
    """Number of half-moves actually played, read back out of the game's PGN
    (Outcome carries pgn but not a ply count directly)."""
    game = chess.pgn.read_game(io.StringIO(pgn_text))
    if game is None:
        return 0
    node = game
    plies = 0
    while node.variations:
        node = node.variations[0]
        plies += 1
    return plies


@dataclass(frozen=True)
class GameSpec:
    game_id: int
    position_id: int
    fen: str
    challenger_white: bool


@dataclass
class GameResult:
    game_id: int
    position_id: int
    challenger_white: bool
    result: str
    termination: str
    challenger_result: str
    pgn: str
    plies: int
    avg_depth: float
    avg_nodes: float
    avg_cutoffs: float
    moves_with_stats: int


@dataclass
class BenchmarkResult:
    games: int
    challenger_wins: int
    draws: int
    champion_wins: int
    challenger_score: float
    failures: int
    avg_plies: float
    avg_depth: float
    avg_nodes: float
    avg_cutoffs: float
    games_with_stats: int


def load_positions(path: Path) -> list[str]:
    positions = []

    for line in path.read_text().splitlines():
        line = line.strip()

        if not line or line.startswith("#"):
            continue

        chess.Board(line)
        positions.append(line)

    if not positions:
        raise ValueError(f"No positions found in {path}")

    return positions


def make_games(positions: list[str]) -> list[GameSpec]:
    games = []
    game_id = 0

    for position_id, fen in enumerate(positions):
        games.append(
            GameSpec(
                game_id=game_id,
                position_id=position_id,
                fen=fen,
                challenger_white=True,
            )
        )
        game_id += 1

        games.append(
            GameSpec(
                game_id=game_id,
                position_id=position_id,
                fen=fen,
                challenger_white=False,
            )
        )
        game_id += 1

    return games


def play_game(
    champion_path: str,
    challenger_path: str,
    spec: GameSpec,
    base_ms: int,
    increment_ms: int,
    ply_cap: int,
) -> GameResult:

    champion = Path(champion_path)
    challenger = Path(challenger_path)

    if spec.challenger_white:
        white_path = challenger
        black_path = champion
    else:
        white_path = champion
        black_path = challenger

    # Hold onto the Agent objects (not just play_match's return value) so we
    # can read .stderr_tail after the game — play_match's finally block
    # already calls .stop() on both, which is what populates stderr_tail.
    white_agent = local(white_path)
    black_agent = local(black_path)

    outcome = play_match(
        white_agent,
        black_agent,
        base_ms=base_ms,
        increment_ms=increment_ms,
        ply_cap=ply_cap,
        start_fen=spec.fen,
    )

    challenger_agent = white_agent if spec.challenger_white else black_agent
    stats = game_averages(challenger_agent.stderr_tail)
    plies = count_plies(outcome.pgn)

    if outcome.result == "draw" or outcome.result == "void":
        challenger_result = "draw"

    elif outcome.result == "white":
        challenger_result = (
            "win" if spec.challenger_white else "loss"
        )

    else:
        challenger_result = (
            "loss" if spec.challenger_white else "win"
        )

    return GameResult(
        game_id=spec.game_id,
        position_id=spec.position_id,
        challenger_white=spec.challenger_white,
        result=outcome.result,
        termination=outcome.termination,
        challenger_result=challenger_result,
        pgn=outcome.pgn,
        plies=plies,
        avg_depth=stats["avg_depth"],
        avg_nodes=stats["avg_nodes"],
        avg_cutoffs=stats["avg_cutoffs"],
        moves_with_stats=stats["moves_with_stats"], # type: ignore
    )


def summarize(results: list[GameResult]) -> BenchmarkResult:
    challenger_wins = sum(
        r.challenger_result == "win"
        for r in results
    )

    draws = sum(
        r.challenger_result == "draw"
        for r in results
    )

    champion_wins = sum(
        r.challenger_result == "loss"
        for r in results
    )

    failures = sum(
        r.termination in FAILED_TERMINATIONS
        for r in results
    )

    games = len(results)

    score = (
        challenger_wins + 0.5 * draws
    ) / games

    # only average search stats over games that actually emitted them — a
    # baseline opponent or a game that failed before move 1 won't have any,
    # and mixing in zeros would silently drag the average down for no reason
    with_stats = [r for r in results if r.moves_with_stats > 0]
    if with_stats:
        avg_depth = sum(r.avg_depth for r in with_stats) / len(with_stats)
        avg_nodes = sum(r.avg_nodes for r in with_stats) / len(with_stats)
        avg_cutoffs = sum(r.avg_cutoffs for r in with_stats) / len(with_stats)
    else:
        avg_depth = avg_nodes = avg_cutoffs = 0.0

    # plies are known for every completed game regardless of whether the
    # challenger emitted stats, so this averages over all results
    avg_plies = sum(r.plies for r in results) / games if games else 0.0

    return BenchmarkResult(
        games=games,
        challenger_wins=challenger_wins,
        draws=draws,
        champion_wins=champion_wins,
        challenger_score=score,
        failures=failures,
        avg_plies=avg_plies,
        avg_depth=avg_depth,
        avg_nodes=avg_nodes,
        avg_cutoffs=avg_cutoffs,
        games_with_stats=len(with_stats),
    )

def load_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Champion vs challenger chess benchmark."
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("bench/benchmark.yml"),
    )

    args = parser.parse_args()

    config = load_config(args.config)

    champion = Path(config["champion"]).resolve()
    challenger = Path(config["challenger"]).resolve()

    positions = load_positions(
        Path(config["positions"])
    )

    games = make_games(positions)

    if config.get("games") is not None:
        games = games[:config["games"]]

    base_ms = config.get("base_ms", 10_000)
    increment_ms = config.get("increment_ms", 0)
    ply_cap = config.get("ply_cap", PLY_CAP)
    workers = config.get("workers", 8)
    output = config.get("output")

    print(f"Champion:   {champion}")
    print(f"Challenger: {challenger}")
    print(f"Positions:  {len(positions)}")
    print(f"Games:      {len(games)}")
    print(f"Clock:      {base_ms} ms + {increment_ms} ms")
    print(f"Workers:    {workers}")
    print()

    # Every worker creates fresh local() agents.
    # Nothing stateful is shared between games.
    with ProcessPoolExecutor(
        max_workers=workers
    ) as executor:

        futures = [
            executor.submit(
                play_game,
                str(champion),
                str(challenger),
                game,
                base_ms,
                increment_ms,
                ply_cap,
            )
            for game in games
        ]

        results = []

        for i, future in enumerate(futures, start=1):
            result = future.result()
            results.append(result)

            stat_note = (
                f"d={result.avg_depth:4.1f} n={result.avg_nodes:6.0f} c={result.avg_cutoffs:5.0f}"
                if result.moves_with_stats
                else f"{'(no stats)':21}"
            )
            print(
                f"[{i:3}/{len(games)}] "
                f"game {result.game_id:3} | "
                f"{'C+' if result.challenger_white else 'C-'} | "
                f"{result.challenger_result:<4} | "
                f"{result.termination:<20} | "
                f"plies={result.plies:3} | "
                f"{stat_note}"
            )

    results.sort(key=lambda r: r.game_id)

    summary = summarize(results)

    print()
    print("=" * 50)
    print("BENCHMARK RESULT")
    print("=" * 50)

    print(
        f"Challenger: "
        f"+{summary.challenger_wins} "
        f"={summary.draws} "
        f"-{summary.champion_wins}"
    )

    print(f"Score:      {summary.challenger_score:.1%}")

    print(f"Failures:   {summary.failures}")
    print(f"Avg plies:  {summary.avg_plies:.1f}")

    if summary.games_with_stats:
        print()
        print(f"Challenger search stats (avg over {summary.games_with_stats} games with stats):")
        print(f"  avg depth:   {summary.avg_depth:.2f}")
        print(f"  avg nodes:   {summary.avg_nodes:.0f}")
        print(f"  avg cutoffs: {summary.avg_cutoffs:.0f}")

    if summary.challenger_score > 0.50:
        print("VERDICT:    CHALLENGER WINS")
    elif summary.challenger_score < 0.50:
        print("VERDICT:    CHAMPION WINS")
    else:
        print("VERDICT:    DRAW")

    if summary.failures:
        print()
        print("WARNING: benchmark contained failed games.")

    if output:
        data = {
            "champion": str(champion),
            "challenger": str(challenger),
            "base_ms": base_ms,
            "increment_ms": increment_ms,
            "results": [asdict(r) for r in results],
            "summary": asdict(summary),
        }
        
        output = Path(output)

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output.write_text(
            json.dumps(data, indent=2)
        )

        print(f"\nResults written to {output}")


if __name__ == "__main__":
    main()