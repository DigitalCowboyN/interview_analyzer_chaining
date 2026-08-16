# tools/graph/reader.py
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Set

from tools.graph.registry import EDGES, NODE_DOMAINS, EdgeType
from tools.capability.reader import load_capabilities
from tools.code.reader import dep_edges, load_units
from tools.adr.index import load_bundle
from tools.usecase.reader import load_use_cases
from tools.testmap.reader import load_tests, verifies_edges
from tools.glossary.model import load_glossary
from tools.graphq.reader import load_queries
from tools.prompts.reader import load_prompt_entries


@dataclass
class Edge:
    type: str
    src: str                 # "<domain>:<id>"
    dst: str                 # "<domain>:<id>"
    props: dict = field(default_factory=dict)


def _addr(node_type: str, node_id) -> str:
    return f"{NODE_DOMAINS[node_type]}:{node_id}"


# --- per-node-type adapter: (load callable, id attribute). Add one for a new node type. ---
_ADAPTERS = {
    "Capability": (load_capabilities, "slug"),
    "CodeUnit": (load_units, "unit"),
    "ADR": (lambda root: load_bundle(os.path.join(root, "docs/adr")), "id"),
    "UseCase": (load_use_cases, "slug"),
    "Test": (load_tests, "slug"),
    "GlossaryTerm": (lambda root: load_glossary(os.path.join(root, "docs/glossary")), "term"),
    "GraphQuery": (load_queries, "name"),
    "Prompt": (load_prompt_entries, "key"),
}


def nodes(root: str = ".") -> Dict[str, Set[str]]:
    out: Dict[str, Set[str]] = {}
    for t, (load, idattr) in _ADAPTERS.items():
        out[t] = {str(getattr(n, idattr)) for n in load(root)}
    return out


def _unit_dir(unit: str) -> str:
    if unit.startswith("tools."):
        return f"tools/{unit.split('.', 1)[1]}/"
    if "." in unit:                                   # src key module a.b -> its package dir
        return "src/" + "/".join(unit.split(".")[:-1]) + "/"
    return f"src/{unit}/"


def _units_under(path: str, code_ids: Set[str]) -> Set[str]:
    p = path if path.endswith("/") else path + "/"
    return {u for u in code_ids if _unit_dir(u).startswith(p)}


def _unit_of_file(path: str, code_ids: Set[str]) -> List[str]:
    """The top-level code unit that owns a src/tools file path (src/events/x.py -> 'events')."""
    p = (path or "").replace("\\", "/")
    parts = p.split("/")
    if len(parts) >= 2 and parts[0] in ("src", "tools"):
        unit = parts[1] if parts[0] == "src" else f"tools.{parts[1]}"
        return [unit] if unit in code_ids else []
    return []


def _authored(edge: EdgeType, root: str, node_ids: Dict[str, Set[str]]) -> List[Edge]:
    load, idattr = _ADAPTERS[edge.from_type]
    out: List[Edge] = []
    for n in load(root):
        src = _addr(edge.from_type, getattr(n, idattr))
        targets = getattr(n, edge.field, None) or []
        if isinstance(targets, (str, int)):
            targets = [targets]
        for t in targets:
            if edge.resolve == "path":
                dsts = _units_under(str(t), node_ids[edge.to_type])
            elif edge.resolve == "file":
                dsts = _unit_of_file(str(t), node_ids[edge.to_type])
            else:
                dsts = [str(t)]                        # kept even if unresolved — the guard flags it
            for d in dsts:
                out.append(Edge(edge.name, src, _addr(edge.to_type, d)))
    return out


def _derived_deps(edge: EdgeType, root: str) -> List[Edge]:
    return [Edge(edge.name, _addr("CodeUnit", u), _addr("CodeUnit", d))
            for u, deps in dep_edges(root).items() for d in deps]


def _derived_verifies(edge: EdgeType, root: str) -> List[Edge]:
    return [Edge(edge.name, src, dst, {"test_type": tt})
            for src, dst, tt in verifies_edges(root)]


def _derived_consumers(from_type, id_attr, load):
    def build(edge: EdgeType, root: str) -> List[Edge]:
        out: List[Edge] = []
        for o in load(root):
            src = _addr(from_type, getattr(o, id_attr))
            for c in getattr(o, "consumers", []):
                out.append(Edge(edge.name, src, _addr("CodeUnit", c)))
        return out
    return build


_DERIVED = {
    "dep_edges": _derived_deps,
    "verifies_edges": _derived_verifies,
    "gq_consumed_by": _derived_consumers("GraphQuery", "name", load_queries),
    "prompt_consumed_by": _derived_consumers("Prompt", "key", load_prompt_entries),
}


def harvest(root: str = ".", edges: List[EdgeType] = EDGES) -> List[Edge]:
    node_ids = nodes(root)
    out: List[Edge] = []
    for e in edges:
        if e.source == "authored":
            out += _authored(e, root, node_ids)
        else:
            out += _DERIVED[e.field](e, root)
    return out
