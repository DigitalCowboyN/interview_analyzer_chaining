# tools/capability/check.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List

from tools.capability.reader import CATEGORIES, code_nodes, load_capabilities, real_code_units
from tools.capability.render import render_index

_MANDATORY_ROLES = ("pipeline-layer", "surface", "tooling")
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


def check_coverage(caps, nodes) -> List[Finding]:
    claimed = set()
    for c in caps:
        claimed.update(c.implemented_by)
    findings: List[Finding] = []
    for n in nodes:
        if n.role not in _MANDATORY_ROLES:
            continue  # infrastructure/model/agent — advisory, never flagged
        parent_pkg = n.unit.split(".")[0]
        if n.unit not in claimed and parent_pkg not in claimed:
            findings.append(Finding(
                f"capability: code unit {n.unit} ({n.role}) is claimed by no capability"))
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
    findings += check_coverage(caps, code_nodes(root))
    findings += check_classification(caps)
    findings += check_index_sync(os.path.join(root, "docs/capabilities/index.md"), caps)
    return findings
