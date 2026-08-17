"""Renders the `tools.api` `Endpoint` list into the `docs/api/index.md` catalog, grouped by
router module."""

from __future__ import annotations

from typing import List

from tools.api.reader import Endpoint


def render_catalog(endpoints: List[Endpoint]) -> str:
    by_router: dict = {}
    for e in endpoints:
        by_router.setdefault(e.router, []).append(e)
    lines = ["# API surface", ""]
    for router in sorted(by_router):
        lines.append(f"## {router}")
        lines.append("")
        for e in sorted(by_router[router], key=lambda e: (e.path, e.method)):
            summ = f" — {e.summary}" if e.summary else ""
            lines.append(f"- `{e.method} {e.path}`{summ}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
