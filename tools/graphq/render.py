"""Renders loaded `QueryEntry` records into `docs/graph-queries/index.md`, one table per
source bundle listing each query's purpose, scope, audience, consumers, and schema
footprint (labels/rels/returns)."""

from __future__ import annotations

from typing import List

from tools.graphq.reader import QueryEntry


def render_catalog(entries: List[QueryEntry]) -> str:
    by_bundle: dict = {}
    for e in entries:
        by_bundle.setdefault(e.bundle, []).append(e)
    lines = ["# Graph-query registry", ""]
    for bundle in sorted(by_bundle):
        lines.append(f"## {bundle}")
        lines.append("")
        lines.append("| query | purpose | scope | audience | consumers | labels | rels | returns |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for e in sorted(by_bundle[bundle], key=lambda e: e.name):
            lines.append(f"| {e.name} | {e.purpose} | {e.scope} | {', '.join(e.audience)} | "
                         f"{', '.join(e.consumers)} | {', '.join(e.labels)} | {', '.join(e.rels)} | "
                         f"{', '.join(e.returns)} |")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
