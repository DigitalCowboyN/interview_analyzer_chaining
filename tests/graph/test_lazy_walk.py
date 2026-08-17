import pytest

from tools.graph.reader import harvest
from tools.graph.traverse import walk, Subgraph


def _harvest_walk(entry, direction, depth, root="."):
    """The OLD algorithm, inlined, to compare against: harvest -> adjacency -> BFS."""
    from collections import defaultdict, deque
    edges = harvest(root)
    out, inc = defaultdict(list), defaultdict(list)
    for e in edges:
        out[e.src].append((e.dst, e))
        inc[e.dst].append((e.src, e))
    starts = [entry] if isinstance(entry, str) else list(entry)
    visited, frontier, seen = set(starts), deque((s, 0) for s in starts), set()
    used = []
    while frontier:
        addr, d = frontier.popleft()
        if depth is not None and d >= depth:
            continue
        nbrs = (out.get(addr, []) if direction in ("out", "both") else []) + \
               (inc.get(addr, []) if direction in ("in", "both") else [])
        for nbr, e in nbrs:
            k = (e.src, e.dst, e.type)
            if k not in seen:
                seen.add(k)
                used.append(e)
            if nbr not in visited:
                visited.add(nbr)
                frontier.append((nbr, d + 1))
    return visited, {(e.src, e.dst, e.type) for e in used if e.src in visited and e.dst in visited}


CASES = [
    ("code:tools.graph.reader", "both", 2),
    ("code:tools.graph", "out", 1),
    ("capabilities:link-the-domains", "out", None),
    ("code:tools.graph.classify", "in", 2),
]


@pytest.mark.parametrize("entry,direction,depth", CASES)
def test_lazy_walk_matches_harvest(entry, direction, depth):
    got = walk(entry, direction=depth and direction or direction, depth=depth)  # level defaults to module
    want_nodes, want_edges = _harvest_walk(entry, direction, depth)
    assert set(got.nodes) == want_nodes
    assert {(e.src, e.dst, e.type) for e in got.edges} == want_edges
