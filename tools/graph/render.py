# tools/graph/render.py
from __future__ import annotations

import re
from typing import Dict, List, Set

from tools.graph.reader import Edge
from tools.graph.registry import EDGES, EdgeType

_UNSAFE = re.compile(r"[^A-Za-z0-9_]")


def _safe_id(addr: str) -> str:
    return "n_" + _UNSAFE.sub("_", addr)


def render_catalog(edges: List[Edge], node_ids: Dict[str, Set[str]]) -> str:
    counts: Dict[str, int] = {}
    for e in edges:
        counts[e.type] = counts.get(e.type, 0) + 1

    lines = ["# Graph", "", "| edge | inverse | from → to | source | count |",
             "| --- | --- | --- | --- | --- |"]
    for et in EDGES:
        lines.append(
            f"| {et.name} | {et.inverse} | {et.from_type} → {et.to_type} | "
            f"{et.source} | {counts.get(et.name, 0)} |"
        )
    lines.append("")

    lines.append("## Nodes")
    lines.append("")
    for t in sorted(node_ids):
        lines.append(f"- {t}: {len(node_ids[t])}")
    lines.append("")

    lines.append("## Meta-schema")
    lines.append("")
    lines.append("```mermaid")
    lines.append("graph LR")
    for et in EDGES:
        lines.append(f"    {et.from_type} -->|{et.name}| {et.to_type}")
    lines.append("```")

    return "\n".join(lines).rstrip() + "\n"


def render_graph(edges: List[Edge]) -> str:
    by_type: Dict[str, List[Edge]] = {}
    for e in edges:
        by_type.setdefault(e.type, []).append(e)

    lines: List[str] = []
    for et in EDGES:
        instances = by_type.get(et.name)
        if not instances:
            continue
        lines.append(f"## {et.name}")
        lines.append("")
        lines.append("```mermaid")
        lines.append("graph LR")
        for e in sorted(instances, key=lambda e: (e.src, e.dst)):
            src_id, dst_id = _safe_id(e.src), _safe_id(e.dst)
            lines.append(f'    {src_id}["{e.src}"] --> {dst_id}["{e.dst}"]')
        lines.append("```")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"
