"""Ephemeral graph traversal. `walk(entry, direction, depth)` rebuilds the reachable
subgraph from source on each call — the LLM working-context substrate (ADR-0025) —
resolving each node's claim/context via the per-type `_CONTEXT` table. Separate from the
transcript Neo4j graph."""
from __future__ import annotations

import os
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from tools.graph.reader import Edge
from tools.graph.reader import nodes as _all_nodes, _unit_dir
from tools.graph.registry import NODE_DOMAINS
from tools.graph.neighbors import WalkContext, neighbors
from tools.capability.reader import load_capabilities
from tools.usecase.reader import load_use_cases
from tools.code.reader import load_units
from tools.adr.index import load_bundle
from tools.testmap.reader import load_tests
from tools.glossary.model import load_glossary
from tools.graphq.reader import load_queries
from tools.prompts.reader import load_prompt_entries


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


# slug -> node type name (inverse of NODE_DOMAINS)
_SLUG_TYPE = {slug: t for t, slug in NODE_DOMAINS.items()}

# node type -> (loader(root)->objs, id attribute, body function(obj)->str)
_CONTEXT = {
    "Capability": (load_capabilities, "slug", lambda o: o.statement),
    "UseCase": (load_use_cases, "slug", lambda o: o.statement),
    "CodeUnit": (load_units, "unit", lambda o: o.description),
    "ADR": (lambda root: load_bundle(os.path.join(root, "docs/adr")), "id",
            lambda o: f"{o.title}\n{o.body}"),
    "Test": (load_tests, "slug",
             lambda o: f"{o.slug} ({o.test_type}) verifies {o.target or o.verifies}"),
    "GlossaryTerm": (lambda root: load_glossary(os.path.join(root, "docs/glossary")), "term",
                     lambda o: o.definition),
    "GraphQuery": (load_queries, "graph_id",
                   lambda o: f"{o.purpose or ''} returns={o.returns} labels={o.labels}".strip()),
    "Prompt": (load_prompt_entries, "graph_id",
               lambda o: f"used_for={o.used_for} audience={o.audience}"),
}


def resolve_context(addresses, root: str = "."):
    """address -> (type, context body). Loads each needed type's objects once."""
    want = defaultdict(set)                     # type -> {id}
    for a in addresses:
        slug, _, nid = a.partition(":")
        t = _SLUG_TYPE.get(slug)
        if t:
            want[t].add(nid)
    out = {}
    for t, ids in want.items():
        spec = _CONTEXT.get(t)
        if not spec:
            continue
        load, idattr, body = spec
        by_id = {str(getattr(o, idattr)): o for o in load(root)}
        for nid in ids:
            o = by_id.get(nid)
            out[f"{NODE_DOMAINS[t]}:{nid}"] = (t, (body(o) or "").strip() if o else "")
    return out


def _entry_addresses(entry: str, root: str = ".") -> List[str]:
    if entry.startswith("type:"):
        t = entry[len("type:"):]
        ids = _all_nodes(root).get(t, set())
        slug = NODE_DOMAINS.get(t)
        return sorted(f"{slug}:{i}" for i in ids) if slug else []
    if entry.startswith("under:"):
        path = entry[len("under:"):]
        p = path if path.endswith("/") else path + "/"
        return sorted(f"code:{u}" for u in _all_nodes(root).get("CodeUnit", set())
                      if _unit_dir(u).startswith(p))
    return [entry]


def walk(entry, direction: str = "both", depth: Optional[int] = None,
         root: str = ".", level: str = "module") -> Subgraph:
    """Materialize the subgraph reachable from `entry` by expanding neighbors on demand
    (lazy, frontier-driven). `level` gates disclosure: 'module' (default) never descends into
    symbols; 'symbol' expands symbol nodes along the frontier."""
    ctx = WalkContext(root=root, level=level)
    starts = _entry_addresses(entry, root) if isinstance(entry, str) else list(entry)

    visited = set(starts)
    frontier = deque((s, 0) for s in starts)
    used_edges: List[Edge] = []
    seen_edge = set()

    while frontier:
        addr, d = frontier.popleft()
        if depth is not None and d >= depth:
            continue
        for nbr, e in neighbors(addr, direction, ctx):
            key = (e.src, e.dst, e.type)
            if key not in seen_edge:
                seen_edge.add(key)
                used_edges.append(e)
            if nbr not in visited:
                visited.add(nbr)
                frontier.append((nbr, d + 1))

    induced = [e for e in used_edges if e.src in visited and e.dst in visited]
    ctx_map = resolve_context(visited, root)
    nodes = {}
    for a in visited:
        rec = ctx.symbol_record(a.partition(":")[2]) if (
            level == "symbol" and a.startswith("code:")) else None
        if rec is not None:                                  # symbol node: signature + docstring
            body = rec.signature + ("\n" + rec.docstring if rec.docstring else "")
            nodes[a] = Node(address=a, type="CodeUnit", context=body.strip())
        else:
            t, body = ctx_map.get(a, ("", ""))
            nodes[a] = Node(address=a, type=t, context=body)
    return Subgraph(nodes=nodes, edges=induced)
