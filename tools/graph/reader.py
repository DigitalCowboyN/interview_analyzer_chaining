# tools/graph/reader.py
"""Assembles the cross-domain graph's `Edge` list and node-id sets by calling each domain's
reader (`tools.capability`, `tools.code`, `tools.adr`, ...) through the registry-driven
adapters in `tools.graph.registry`: `nodes()` collects every domain's ids, and `harvest()`
resolves each registered `EdgeType` — authored edges from a frontmatter field, derived edges
from a domain-specific builder — into concrete `Edge`s addressed as `<domain>:<id>`."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Set

from tools.graph.registry import EDGES, NODE_DOMAINS, EdgeType
from tools.capability.reader import load_capabilities
from tools.code.reader import contains_edges, dep_edges, load_units
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
    "GraphQuery": (load_queries, "graph_id"),
    "Prompt": (load_prompt_entries, "graph_id"),
}


def nodes(root: str = ".") -> Dict[str, Set[str]]:
    out: Dict[str, Set[str]] = {}
    for t, (load, idattr) in _ADAPTERS.items():
        out[t] = {str(getattr(n, idattr)) for n in load(root)}
    return out


def _unit_dir(unit: str) -> str:
    if unit.startswith("tools."):
        return "tools/" + "/".join(unit.split(".")[1:]) + "/"
    return "src/" + "/".join(unit.split(".")) + "/"


def _units_under(path: str, code_ids: Set[str]) -> Set[str]:
    if path.endswith(".py"):                            # a specific file -> its module node
        return set(_unit_of_file(path, code_ids))
    p = path if path.endswith("/") else path + "/"
    return {u for u in code_ids if _unit_dir(u).startswith(p)}


def _unit_of_file(path: str, code_ids: Set[str]) -> List[str]:
    """The code node that owns a src/tools file path (src/events/store.py -> 'events.store')."""
    p = (path or "").replace("\\", "/")
    parts = p.split("/")
    if len(parts) < 2 or parts[0] not in ("src", "tools") or not parts[-1].endswith(".py"):
        return []
    prefix = "tools." if parts[0] == "tools" else ""
    stem = parts[-1][:-3]
    mid_parts = parts[1:-1] + ([stem] if stem != "__init__" else [])
    module_id = prefix + ".".join(mid_parts)
    if module_id in code_ids:
        return [module_id]
    pkg_id = prefix + ".".join(parts[1:-1])
    return [pkg_id] if pkg_id in code_ids else []


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


def _derived_contains(edge: EdgeType, root: str) -> List[Edge]:
    return [Edge(edge.name, _addr("CodeUnit", p), _addr("CodeUnit", c))
            for p, c in contains_edges(root)]


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


def _derived_writes(edge: EdgeType, root: str) -> List[Edge]:
    from tools.graph.flow import writes_edges
    return [Edge(edge.name, _addr("CodeUnit", mod), _addr("GlossaryTerm", label))
            for mod, labels in writes_edges(root).items() for label in labels]


def _derived_reads(edge: EdgeType, root: str) -> List[Edge]:
    from tools.glossary.model import load_glossary
    terms = {t.term for t in load_glossary(os.path.join(root, "docs/glossary"))}
    out: List[Edge] = []
    for q in load_queries(root):
        for label in getattr(q, "labels", []) or []:
            if label in terms:                                   # only real glossary labels
                out.append(Edge(edge.name, _addr("GraphQuery", q.graph_id),
                                _addr("GlossaryTerm", label)))
    return out


_DERIVED = {
    "dep_edges": _derived_deps,
    "contains_edges": _derived_contains,
    "verifies_edges": _derived_verifies,
    "gq_consumed_by": _derived_consumers("GraphQuery", "graph_id", load_queries),
    "prompt_consumed_by": _derived_consumers("Prompt", "graph_id", load_prompt_entries),
    "reads_edges": _derived_reads,
    "writes_edges": _derived_writes,
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
