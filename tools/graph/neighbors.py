# tools/graph/neighbors.py
"""Lazy neighbor expansion for walk(): a node's incident edges computed from its own
file/AST/id (outbound + structural) and from a per-walk cached base index (module/doc grain),
so a traversal never eagerly builds symbols. Symbol expansion (level='symbol') parses a module's
bodies only when the frontier reaches it — memoized per module on the WalkContext."""
# governed-by: ADR-0025 ADR-0027
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from tools.graph.reader import Edge


@dataclass
class WalkContext:
    root: str = "."
    level: str = "module"
    _out: Dict[str, List[Tuple[str, Edge]]] = field(default_factory=dict)
    _inc: Dict[str, List[Tuple[str, Edge]]] = field(default_factory=dict)
    _built: bool = False
    _code_ids: Set[str] = field(default_factory=set)     # module/package node ids (base grain)
    _sym_cache: Dict[str, list] = field(default_factory=dict)   # module_id -> [Symbol]

    def _ensure(self):
        # Module-grain base graph: cheap (no symbol bodies). Built once per walk, cached.
        if self._built:
            return
        from tools.graph.reader import harvest, nodes
        for e in harvest(self.root):
            self._out.setdefault(e.src, []).append((e.dst, e))
            self._inc.setdefault(e.dst, []).append((e.src, e))
        self._code_ids = set(nodes(self.root).get("CodeUnit", ()))
        self._built = True

    def symbols_for(self, module_id: str) -> list:
        """The symbols of a module, parsed and memoized on first access (frontier-lazy)."""
        if module_id not in self._sym_cache:
            from tools.code.reader import symbols_of
            self._sym_cache[module_id] = symbols_of(module_id, self.root)
        return self._sym_cache[module_id]

    def owning_module(self, cid: str) -> Optional[str]:
        """The longest prefix of a code id that is a real module/package node."""
        parts = cid.split(".")
        for i in range(len(parts), 0, -1):
            pref = ".".join(parts[:i])
            if pref in self._code_ids:
                return pref
        return None

    def symbol_record(self, cid: str):
        """The Symbol for a symbol-grain code id, or None if cid is a module/package/unknown."""
        mod = self.owning_module(cid)
        if mod is None or mod == cid:        # cid is itself a module/package, not a symbol
            return None
        for s in self.symbols_for(mod):
            if s.id == cid:
                return s
        return None


def _symbol_edges(addr: str, direction: str, ctx: WalkContext) -> List[Tuple[str, Edge]]:
    """Symbol-grain edges spliced in at level='symbol' — a module's `contains` to its top-level
    symbols, a class's `contains` to its methods, a symbol's `contained_by` (parent) and `calls`."""
    dom, _, cid = addr.partition(":")
    if dom != "code":
        return []
    out: List[Tuple[str, Edge]] = []
    if cid in ctx._code_ids:                 # addr is a module/package
        if direction in ("out", "both"):
            for s in ctx.symbols_for(cid):
                if s.parent == cid:          # top-level symbols (fns/classes), not methods
                    out.append((f"code:{s.id}", Edge("contains", addr, f"code:{s.id}")))
        return out
    rec = ctx.symbol_record(cid)             # addr is a symbol
    if rec is None:
        return out
    if direction in ("in", "both"):
        out.append((f"code:{rec.parent}", Edge("contains", f"code:{rec.parent}", addr)))
    if direction in ("out", "both"):
        for s in ctx.symbols_for(ctx.owning_module(cid)):
            if s.parent == cid:              # a class contains its methods
                out.append((f"code:{s.id}", Edge("contains", addr, f"code:{s.id}")))
        for callee in rec.calls:
            out.append((f"code:{callee}", Edge("calls", addr, f"code:{callee}")))
        for ev in getattr(rec, "emits", []):
            out.append((f"code:{ev}", Edge("emits", addr, f"code:{ev}")))
    return out


def neighbors(addr: str, direction: str, ctx: WalkContext) -> List[Tuple[str, Edge]]:
    ctx._ensure()
    pairs: List[Tuple[str, Edge]] = []
    if direction in ("out", "both"):
        pairs += ctx._out.get(addr, [])
    if direction in ("in", "both"):
        pairs += ctx._inc.get(addr, [])
    if ctx.level == "symbol":
        pairs += _symbol_edges(addr, direction, ctx)
    return pairs
