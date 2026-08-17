# tools/capability/check.py
"""Non-blocking findings for the `tools.capability` domain: `implemented_by` links to
unknown code units, source packages claimed by no capability, primaries/children with
invalid kind/tier/category, and a stale `docs/capabilities/index.md`. `run_all` is the
entry point `tools.capability.__main__` calls."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List

from tools.capability.reader import CATEGORIES, load_capabilities, real_code_units
from tools.code.reader import load_units
from tools.capability.render import render_index

# Top-level src packages that are infrastructure / model / agent — not expected to trace to a
# capability. The source-derived replacement for the retired per-unit `role` exclusion; add a
# package here (or give it a capability) when the coverage check flags a new infra area.
_INFRA_PACKAGES = frozenset({
    "agents", "models", "events", "persistence", "utils", "io", "commands",
})
_VALID_KINDS = ("primary", "child", "variant")


@dataclass
class Finding:
    message: str


def check_links(caps, valid_units) -> List[Finding]:
    findings: List[Finding] = []
    for c in caps:
        for u in c.implemented_by:
            if u not in valid_units:
                findings.append(Finding(
                    f"capability: {c.slug} implemented_by unknown code unit '{u}'"))
    return findings


def check_coverage(caps, units) -> List[Finding]:
    """A product/tooling package that no capability claims. Mandatory scope = every top-level
    tools.* package and every top-level src package except infrastructure (_INFRA_PACKAGES).
    A package is covered if it, or any module/sub-package under it, is implemented_by a capability."""
    claimed = set()
    for c in caps:
        claimed.update(c.implemented_by)
    findings: List[Finding] = []
    for u in units:
        if u.level != "package":
            continue
        is_tool = u.unit.startswith("tools.")
        segs = u.unit.count(".")
        if is_tool and segs != 1:
            continue                              # only top-level tools.<name> packages
        if not is_tool and segs != 0:
            continue                              # only top-level src packages
        if not is_tool and u.unit in _INFRA_PACKAGES:
            continue                              # infrastructure — not expected to trace to a capability
        covered = u.unit in claimed or any(t.startswith(u.unit + ".") for t in claimed)
        if not covered:
            findings.append(Finding(f"capability: package {u.unit} is claimed by no capability"))
    return findings


def check_classification(caps) -> List[Finding]:
    slugs = {c.slug for c in caps}
    findings: List[Finding] = []
    for c in caps:
        if c.kind not in _VALID_KINDS:
            findings.append(Finding(f"capability: {c.slug} has no/invalid kind"))
        if c.kind == "primary":
            if c.tier not in ("core", "enabling"):
                findings.append(Finding(f"capability: primary {c.slug} has no tier"))
            if c.category not in CATEGORIES:
                findings.append(Finding(f"capability: primary {c.slug} has no/invalid category"))
        if c.kind in ("child", "variant"):
            if not c.parent:
                findings.append(Finding(f"capability: {c.kind} {c.slug} has no parent"))
            elif c.parent not in slugs:
                findings.append(Finding(
                    f"capability: {c.slug} parent '{c.parent}' does not resolve"))
    return findings


def check_index_sync(index_path: str, caps) -> List[Finding]:
    want = render_index(caps)
    have = open(index_path, encoding="utf-8", errors="ignore").read() if os.path.exists(index_path) else ""
    if want != have:
        return [Finding("capability: docs/capabilities/index.md out of sync — run make capability-index")]
    return []


def run_all(root: str = ".") -> List[Finding]:
    caps = load_capabilities(root)
    findings: List[Finding] = []
    findings += check_links(caps, real_code_units(root))
    findings += check_coverage(caps, load_units(root))
    findings += check_classification(caps)
    findings += check_index_sync(os.path.join(root, "docs/capabilities/index.md"), caps)
    return findings
