# tools/code/check.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List

from tools.code.reader import CodeUnit, load_units
from tools.code.render import render_index, render_pipeline


@dataclass
class Finding:
    message: str


def check_missing_docstring(units: List[CodeUnit]) -> List[Finding]:
    """A module with no docstring has no derivable context — the completeness signal that
    replaces the retired authored `role`/description overlay."""
    return [Finding(f"code: module {u.unit} has no docstring (no derivable context)")
            for u in units if u.level == "module" and not u.description]


def check_map_in_sync(index_path: str, pipeline_path: str, units: List[CodeUnit]) -> List[Finding]:
    findings: List[Finding] = []
    for path, render in ((index_path, render_index), (pipeline_path, render_pipeline)):
        want = render(units)
        have = open(path, encoding="utf-8", errors="ignore").read() if os.path.exists(path) else ""
        if want != have:
            findings.append(Finding(
                f"code: {os.path.basename(path)} out of sync — run make code-index"))
    return findings


def run_all(root: str = ".") -> List[Finding]:
    units = load_units(root)
    findings: List[Finding] = []
    findings += check_missing_docstring(units)
    findings += check_map_in_sync(os.path.join(root, "docs/code/index.md"),
                                  os.path.join(root, "docs/code/pipeline.md"), units)
    return findings
