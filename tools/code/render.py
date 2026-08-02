from __future__ import annotations

from typing import List

from tools.code.reader import CodeUnit


def render_index(units: List[CodeUnit]) -> str:
    by_role: dict = {}
    for u in units:
        by_role.setdefault(u.role or "(unclassified)", []).append(u)
    lines = ["# Code map", "", "See `pipeline.md` for the dependency graph.", ""]
    for role in sorted(by_role):
        lines.append(f"## {role}")
        lines.append("")
        lines.append("| unit | io | depends_on |")
        lines.append("| --- | --- | --- |")
        for u in sorted(by_role[role], key=lambda u: u.unit):
            lines.append(f"| {u.unit} | {', '.join(u.io)} | {', '.join(u.depends_on)} |")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_pipeline(units: List[CodeUnit]) -> str:
    lines = ["# Dependency / pipeline map", "", "```mermaid", "graph LR"]
    for u in sorted(units, key=lambda u: u.unit):
        if not u.depends_on:
            lines.append(f"    {u.unit}")
        for dep in u.depends_on:
            lines.append(f"    {u.unit} --> {dep}")
    lines.append("```")
    return "\n".join(lines) + "\n"
