from __future__ import annotations

from typing import List

from tools.code.reader import CodeUnit


def render_index(units: List[CodeUnit], axes=None) -> str:
    axes = axes or {}
    lines = ["# Code map", "",
             "Derived from `src/` and `tools/`. See `pipeline.md` for the dependency graph.", ""]
    for level in ("package", "module"):
        rows = sorted((u for u in units if u.level == level), key=lambda u: u.unit)
        if not rows:
            continue
        lines += [f"## {level.capitalize()}s", "",
                  "| unit | category | determinism | depends_on |", "| --- | --- | --- | --- |"]
        for u in rows:
            cat, det = axes.get(u.unit, ("", ""))
            lines.append(f"| {u.unit} | {cat} | {det} | {', '.join(u.depends_on)} |")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_pipeline(units: List[CodeUnit]) -> str:
    lines = ["# Dependency / pipeline map", "", "```mermaid", "graph LR"]
    for u in sorted(units, key=lambda u: u.unit):
        for dep in u.depends_on:                         # modules only carry deps
            lines.append(f"    {u.unit} --> {dep}")
    lines.append("```")
    return "\n".join(lines) + "\n"
