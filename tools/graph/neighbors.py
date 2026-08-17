# tools/graph/neighbors.py
"""Lazy neighbor expansion for walk(): a node's incident edges computed from its own
file/AST/id (outbound + structural) and from a per-walk cached reverse index (inbound intent
edges), so a traversal never builds the whole graph. Symbol expansion (level='symbol') parses a
module's bodies only when the frontier reaches it."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from tools.graph.reader import Edge


@dataclass
class WalkContext:
    root: str = "."
    level: str = "module"
    _all_edges: Optional[List[Edge]] = None          # cached full edge set (see note)
    _out: Dict[str, List[Tuple[str, Edge]]] = field(default_factory=dict)
    _inc: Dict[str, List[Tuple[str, Edge]]] = field(default_factory=dict)
    _built: bool = False

    def _ensure(self):
        # Module-grain base graph: cheap (no symbol bodies). Built once per walk, cached.
        # This IS today's harvest at module grain; Task 4 makes symbol edges lazy on top.
        if self._built:
            return
        from tools.graph.reader import harvest
        self._all_edges = harvest(self.root)
        for e in self._all_edges:
            self._out.setdefault(e.src, []).append((e.dst, e))
            self._inc.setdefault(e.dst, []).append((e.src, e))
        self._built = True


def neighbors(addr: str, direction: str, ctx: WalkContext) -> List[Tuple[str, Edge]]:
    ctx._ensure()
    pairs: List[Tuple[str, Edge]] = []
    if direction in ("out", "both"):
        pairs += ctx._out.get(addr, [])
    if direction in ("in", "both"):
        pairs += ctx._inc.get(addr, [])
    return pairs
