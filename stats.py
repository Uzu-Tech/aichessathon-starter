from dataclasses import dataclass, field
from time import perf_counter


@dataclass
class SearchStats:
    nodes: int = 0
    leaves: int = 0
    cutoffs: int = 0

    nodes_by_depth: dict[int, int] = field(default_factory=dict)

    def node(self, depth: int):
        self.nodes += 1
        self.nodes_by_depth[depth] = (
            self.nodes_by_depth.get(depth, 0) + 1
        )

    def report(self, elapsed: float):
        print("\n========== SEARCH ==========")
        print(f"Time:       {elapsed * 1000:.2f} ms")
        print(f"Nodes:      {self.nodes:,}")
        print(f"Nodes/sec:  {self.nodes / elapsed:,.0f}")
        print(f"Leaves:     {self.leaves:,}")
        print(f"Cutoffs:    {self.cutoffs:,}")

        print("\nNodes by depth:")
        for depth in sorted(self.nodes_by_depth, reverse=True):
            print(
                f"  depth {depth}: "
                f"{self.nodes_by_depth[depth]:,}"
            )