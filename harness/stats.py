"""
Parse and aggregate per-move search stats an agent writes to stderr.

Your agent.py can print one JSON line per move to stderr, e.g.:

    print(json.dumps({
        "stats": True, "ply": board.ply(),
        "depth": depth, "nodes": nodes, "cutoffs": cutoffs,
    }), file=sys.stderr)

stderr is never seen by the referee or the wire protocol — runner.py routes
fd 1 to fd 2 before importing agent.py, so anything your agent writes via
print() or sys.stderr never reaches the {"move": ...} channel. sandbox.py's
Agent already captures the full stderr stream into `agent.stderr_tail`,
populated once play_match() returns (its finally block calls agent.stop()).
This module just reads that text back out for local benchmarking — nothing
here ships in agent.zip, and nothing here affects what the referee sees.
"""

import json


def parse_stat_lines(stderr_text: str) -> list[dict]:
    """Pull out every well-formed {"stats": true, ...} line, ignore the rest."""
    stats = []
    for line in stderr_text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and obj.get("stats") is True:
            stats.append(obj)
    return stats


def average(stats: list[dict], key: str) -> float:
    if not stats:
        return 0.0
    return sum(s.get(key, 0) for s in stats) / len(stats)


def game_averages(stderr_text: str) -> dict[str, float | int]:
    """Convenience wrapper: parse + compute all three averages for one game."""
    stats = parse_stat_lines(stderr_text)
    return {
        "avg_depth": average(stats, "depth"),
        "avg_nodes": average(stats, "nodes"),
        "avg_cutoffs": average(stats, "cutoffs"),
        "moves_with_stats": len(stats),
    }