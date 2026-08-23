# tools/graph/check.py
"""Non-blocking drift checks for the cross-domain graph: edge endpoints that fail to
resolve to a known node, an edge-type registry entry pointing at an unknown node type, the
generated `index.md`/`graph.md` going stale against a fresh render, and reachability —
code units unreached by any capability, use-case, or ADR walked outward from `run_all`."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Set

from tools.graph.reader import Edge, harvest, nodes
from tools.graph.registry import EDGES, NODE_DOMAINS, EdgeType
from tools.graph.render import render_catalog, render_graph
from tools.graph.traverse import walk


@dataclass
class Finding:
    message: str


def _domain_to_type(node_domains: Dict[str, str]) -> Dict[str, str]:
    return {domain: node_type for node_type, domain in node_domains.items()}


def check_endpoints(edges: List[Edge], node_ids: Dict[str, Set[str]]) -> List[Finding]:
    domain_to_type = _domain_to_type(NODE_DOMAINS)
    findings: List[Finding] = []
    for e in edges:
        for addr in (e.src, e.dst):
            domain, _, id_ = addr.partition(":")
            node_type = domain_to_type.get(domain)
            if node_type is None or id_ not in node_ids.get(node_type, set()):
                findings.append(
                    Finding(f"graph: edge {e.type} endpoint {addr} does not resolve")
                )
    return findings


def check_registry(
    edges_registry: List[EdgeType], node_domains: Dict[str, str]
) -> List[Finding]:
    findings: List[Finding] = []
    for et in edges_registry:
        if et.from_type not in node_domains:
            findings.append(
                Finding(f"graph: edge type {et.name} from_type {et.from_type} is not a known node type")
            )
        for tt in et.to_type.split("|"):
            if tt not in node_domains:
                findings.append(
                    Finding(f"graph: edge type {et.name} to_type {tt} is not a known node type")
                )
        if et.source == "authored" and not et.field:
            findings.append(
                Finding(f"graph: authored edge type {et.name} has no field")
            )
    return findings


def check_index_sync(
    index_path: str,
    graph_path: str,
    edges: List[Edge],
    node_ids: Dict[str, Set[str]],
) -> List[Finding]:
    findings: List[Finding] = []

    def _read(path: str) -> str:
        if not os.path.exists(path):
            return ""
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    fresh_catalog = render_catalog(edges, node_ids)
    fresh_graph = render_graph(edges)

    if _read(index_path) != fresh_catalog:
        findings.append(Finding(f"graph: {index_path} is out of sync with a fresh render"))
    if _read(graph_path) != fresh_graph:
        findings.append(Finding(f"graph: {graph_path} is out of sync with a fresh render"))

    return findings


def check_reachability(root: str = ".") -> List[Finding]:
    """Code the graph cannot explain: a CodeUnit reached by no Capability / UseCase / ADR.

    One multi-start walk outward from every "why" node; anything not in the reached set has no
    path from an intent, a use-case, or a decision (nor is a dependency of anything that does)."""
    ns = nodes(root)
    intents = ([f"capabilities:{i}" for i in ns.get("Capability", ())]
               + [f"use-cases:{i}" for i in ns.get("UseCase", ())]
               + [f"adr:{i}" for i in ns.get("ADR", ())])
    reached = set(walk(intents, direction="out", depth=None, root=root).nodes)
    code = {f"code:{u}" for u in ns.get("CodeUnit", ())}
    return [Finding(f"graph: code unit {a} is reached by no capability / use-case / ADR (unexplained)")
            for a in sorted(code - reached)]


def check_flow_registrations(root: str = ".") -> List[Finding]:
    """KG-2 flow-overlay drift: a `registry.register("Type", Handler)` whose `TypeData` event class
    or handler class can't be resolved (so `handled_by` silently drops it); and a registered handler
    module that writes Neo4j (Cypher MERGE) but resolves no glossary label (a candidate missing term)."""
    from tools.graph.flow import register_map, writes_edges, _class_index, _REGISTER, _MERGE_LABEL
    findings: List[Finding] = []
    events = _class_index(root, "events")
    handlers = _class_index(root, "projections.handlers")
    path = os.path.join(root, "src", "projections", "bootstrap.py")
    try:
        text = open(path, encoding="utf-8", errors="ignore").read()
    except OSError:
        return findings
    for m in _REGISTER.finditer(text):
        etype, handler = m.group(1), m.group(2)
        if (etype + "Data") not in events or handler not in handlers:
            findings.append(Finding(
                f"graph: register(\"{etype}\", {handler}) has no resolvable event/handler class "
                f"— handled_by will drop it"))
    wl = writes_edges(root)
    for mod, labels in wl.items():
        if not labels:
            from tools.graph.flow import _module_file
            try:
                src = open(_module_file(mod, root), encoding="utf-8", errors="ignore").read()
            except OSError:
                continue
            if _MERGE_LABEL.search(src):
                findings.append(Finding(
                    f"graph: handler module {mod} writes Neo4j (MERGE) but no label maps to a "
                    f"glossary term — add the term, or writes is blind"))
    return findings


def run_all(root: str = ".") -> List[Finding]:
    try:
        edges = harvest(root)
        node_ids = nodes(root)
    except Exception as exc:  # non-blocking: harvesting itself must never raise out
        return [Finding(f"graph: harvest failed: {exc}")]

    findings: List[Finding] = []
    findings += check_endpoints(edges, node_ids)
    findings += check_registry(EDGES, NODE_DOMAINS)

    index_path = os.path.join(root, "docs/graph/index.md")
    graph_path = os.path.join(root, "docs/graph/graph.md")
    findings += check_index_sync(index_path, graph_path, edges, node_ids)
    try:
        findings += check_reachability(root)   # re-harvests via walk() — guard it too
    except Exception as exc:  # non-blocking: reachability must never raise out
        findings.append(Finding(f"graph: reachability check failed: {exc}"))
    try:
        findings += check_flow_registrations(root)
    except Exception as exc:  # non-blocking
        findings.append(Finding(f"graph: flow-registration check failed: {exc}"))

    return findings
