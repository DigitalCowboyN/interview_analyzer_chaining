# tools/code/check.py
"""Non-blocking findings for the `tools.code` domain: modules with no docstring (no
derivable context) and a stale `docs/code/index.md`/`pipeline.md`. `run_all` is the entry
point `tools.code.__main__` calls."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List

from tools.code.reader import CodeUnit, load_units
from tools.code.render import render_index, render_pipeline
from tools.graph.classify import derive_axes


@dataclass
class Finding:
    message: str


def check_missing_docstring(units: List[CodeUnit]) -> List[Finding]:
    """A module with no docstring has no derivable context — the completeness signal that
    replaces the retired authored `role`/description overlay."""
    return [Finding(f"code: module {u.unit} has no docstring (no derivable context)")
            for u in units if u.level == "module" and not u.description]


def check_missing_symbol_docstring(root: str = ".") -> List[Finding]:
    """Opt-in symbol-grain completeness: functions/classes/methods with no docstring. A symbol is
    'thin, not empty' (its signature is always derivable), so this is a lower-priority backlog than
    the module one — deliberately NOT wired into `run_all`. Parses every module (eager), so callers
    invoke it explicitly when they want a symbol burn-down worklist."""
    from tools.code.reader import symbols_of
    findings: List[Finding] = []
    for u in load_units(root):
        if u.level != "module":
            continue
        for s in symbols_of(u.unit, root):
            if not s.docstring:
                findings.append(Finding(f"code: symbol {s.id} has no docstring (thin — signature only)"))
    return findings


def check_map_in_sync(index_path: str, pipeline_path: str, units: List[CodeUnit], axes=None) -> List[Finding]:
    findings: List[Finding] = []
    renders = ((index_path, lambda u: render_index(u, axes)), (pipeline_path, render_pipeline))
    for path, render in renders:
        want = render(units)
        have = open(path, encoding="utf-8", errors="ignore").read() if os.path.exists(path) else ""
        if want != have:
            findings.append(Finding(
                f"code: {os.path.basename(path)} out of sync — run make code-index"))
    return findings


def run_all(root: str = ".") -> List[Finding]:
    units = load_units(root)
    axes = derive_axes(root)
    findings: List[Finding] = []
    findings += check_missing_docstring(units)
    findings += check_map_in_sync(os.path.join(root, "docs/code/index.md"),
                                  os.path.join(root, "docs/code/pipeline.md"), units, axes)
    return findings
