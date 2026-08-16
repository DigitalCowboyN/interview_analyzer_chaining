from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from tools.graph.reader import Edge, harvest


@dataclass
class Node:
    address: str        # "<slug>:<id>"
    type: str           # node type name ("Capability", ...); filled by the context pass
    context: str = ""   # claim + context body (Task 2)


@dataclass
class Subgraph:
    nodes: Dict[str, Node] = field(default_factory=dict)
    edges: List[Edge] = field(default_factory=list)


def _adjacency(edges: List[Edge]):
    out = defaultdict(list)   # addr -> list[(neighbor, edge)] following edge direction
    inc = defaultdict(list)   # addr -> list[(neighbor, edge)] against edge direction
    for e in edges:
        out[e.src].append((e.dst, e))
        inc[e.dst].append((e.src, e))
    return out, inc


def walk(entry, direction: str = "both", depth: Optional[int] = None, root: str = ".") -> Subgraph:
    """Materialize the subgraph reachable from `entry` — a node address (selectors: Task 3) —
    following edges `out` | `in` | `both`, to `depth` hops (None = to exhaustion). Rebuilt from
    source each call (harvest())."""
    edges = harvest(root)
    out, inc = _adjacency(edges)
    starts = [entry] if isinstance(entry, str) else list(entry)

    visited = set(starts)
    frontier = deque((s, 0) for s in starts)
    used_edges: List[Edge] = []
    seen_edge = set()

    def _neighbors(addr):
        pairs = []
        if direction in ("out", "both"):
            pairs += out.get(addr, [])
        if direction in ("in", "both"):
            pairs += inc.get(addr, [])
        return pairs

    while frontier:
        addr, d = frontier.popleft()
        if depth is not None and d >= depth:
            continue
        for nbr, e in _neighbors(addr):
            key = (e.src, e.dst, e.type)
            if key not in seen_edge:
                seen_edge.add(key)
                used_edges.append(e)
            if nbr not in visited:
                visited.add(nbr)
                frontier.append((nbr, d + 1))

    # induced edges: only those whose BOTH endpoints are in the visited set
    induced = [e for e in used_edges if e.src in visited and e.dst in visited]
    return Subgraph(nodes={a: Node(address=a, type="") for a in visited}, edges=induced)
