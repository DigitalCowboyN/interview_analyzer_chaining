# tools/graph/check.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Set

from tools.graph.reader import Edge, harvest, nodes
from tools.graph.registry import EDGES, NODE_DOMAINS, EdgeType
from tools.graph.render import render_catalog, render_graph


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
        if et.to_type not in node_domains:
            findings.append(
                Finding(f"graph: edge type {et.name} to_type {et.to_type} is not a known node type")
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

    return findings
