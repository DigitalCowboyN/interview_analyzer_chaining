# tools/code/check.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List

from tools.code.reader import CodeUnit, KEY_MODULES, load_units, packages
from tools.code.render import render_index, render_pipeline


@dataclass
class Finding:
    message: str


def check_coverage(pkgs: List[str], units: List[CodeUnit]) -> List[Finding]:
    have = {u.unit for u in units}
    return [Finding(f"code: package src/{p} has no doc node") for p in pkgs if p not in have]


def check_classification(units: List[CodeUnit]) -> List[Finding]:
    return [Finding(f"code: unit {u.unit} has no role classification") for u in units if not u.role]


def check_map_in_sync(index_path: str, pipeline_path: str, units: List[CodeUnit]) -> List[Finding]:
    findings: List[Finding] = []
    for path, render in ((index_path, render_index), (pipeline_path, render_pipeline)):
        want = render(units)
        have = open(path, encoding="utf-8", errors="ignore").read() if os.path.exists(path) else ""
        if want != have:
            findings.append(Finding(f"code: {os.path.basename(path)} out of sync — run make code-index (new dependency?)"))
    return findings


def check_stale(units: List[CodeUnit], real_units: List[str]) -> List[Finding]:
    real = set(real_units)
    return [Finding(f"code: doc node {u.unit} no longer exists in src") for u in units if u.unit not in real]


def run_all(root: str = ".") -> List[Finding]:
    pkgs = packages(root)
    units = load_units(root)
    real = pkgs + KEY_MODULES
    findings: List[Finding] = []
    findings += check_coverage(pkgs, units)
    findings += check_classification(units)
    findings += check_map_in_sync(os.path.join(root, "docs/code/index.md"),
                                  os.path.join(root, "docs/code/pipeline.md"), units)
    findings += check_stale(units, real)
    return findings
