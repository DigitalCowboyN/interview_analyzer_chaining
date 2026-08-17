"""Renders `tools.code` `CodeUnit`s into the `docs/code/index.md` catalog, the
`pipeline.md` mermaid dependency graph, and the docstring backlog worklist."""

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


def render_docstring_backlog(units: List[CodeUnit]) -> str:
    """A burn-down worklist of modules missing a docstring (no derivable context), grouped by
    package. Add a docstring, regenerate, and the module drops off the list; when empty, the
    `check_missing_docstring` signal goes silent."""
    missing = sorted((u for u in units if u.level == "module" and not u.description),
                     key=lambda u: u.unit)
    lines = ["# Docstring backlog", "",
             "Modules with no module-level docstring — no derivable context. Burn this down: add a",
             "docstring, run `make code-index`, and the module drops off this list.", "",
             f"**{len(missing)} module(s)** remaining.", ""]
    groups: dict = {}
    for u in missing:
        pkg = u.unit.rsplit(".", 1)[0] if "." in u.unit else "(top-level)"
        groups.setdefault(pkg, []).append(u.unit)
    for pkg in sorted(groups):
        lines.append(f"## {pkg}")
        lines.append("")
        for uid in groups[pkg]:
            lines.append(f"- [ ] {uid}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
