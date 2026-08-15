from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass
from typing import Dict, List

from tools.capability.reader import CATEGORIES, category_defined, load_capabilities
from tools.usecase.reader import load_use_cases


@dataclass
class Domain:
    slug: str            # docs/<slug>/  (cascade row + graph addressing)
    make: str            # runnable module/check name: `python -m tools.<make> check`
    surfaces: list        # path prefixes whose change can cause this check to find drift


# Single source of truth for the knowledge-graph domains. `surfaces` drives the
# changed-domain pre-commit (tools.knowledge.surfaces). Add a row here (+ a docs/index.md
# row) when a new domain ships.
DOMAINS = [
    Domain("adr", "adr", ["docs/adr/", "src/"]),
    Domain("api", "api", ["src/api/", "frontend/openapi.json"]),
    Domain("cli", "cli", ["Makefile", "tools/"]),
    Domain("code", "code", ["src/", "tools/"]),
    Domain("capabilities", "capability", ["docs/capabilities/", "src/", "tools/"]),
    Domain("glossary", "glossary", ["src/", "docs/glossary/"]),
    Domain("graph", "graph", []),  # cross-domain: always appended by the hook/CI, never path-resolved
    Domain("graph-queries", "graphq", ["src/projections/", "docs/graph-queries/"]),
    Domain("prompts", "prompts", ["src/", "docs/prompts/"]),
    Domain("use-cases", "usecase", ["docs/use-cases/"]),
    Domain("tests", "testmap", ["tests/"]),
]

ADOPTION_DATE = "2026-08-05"          # specs/plans dated >= this must carry the addendum
ADDENDUM_HEADING = "## Knowledge-graph check"
_DATE = re.compile(r"^(\d{4}-\d{2}-\d{2})")


@dataclass
class Finding:
    message: str


def _leading_date(path: str) -> str:
    m = _DATE.match(os.path.basename(path))
    return m.group(1) if m else ""  # "" (no date prefix) => not grandfathered


def check_addendum_present(specs_dir: str, plans_dir: str,
                           adoption_date: str = ADOPTION_DATE) -> List[Finding]:
    findings: List[Finding] = []
    for directory, kind in ((specs_dir, "spec"), (plans_dir, "plan")):
        for path in sorted(glob.glob(os.path.join(directory, "*.md"))):
            date = _leading_date(path)
            if date and date < adoption_date:
                continue  # grandfathered — predates the honesty-check process
            try:
                text = open(path, encoding="utf-8", errors="ignore").read()
            except OSError:
                continue
            if ADDENDUM_HEADING not in text:
                findings.append(Finding(
                    f"knowledge: {kind} {os.path.basename(path)} has no "
                    f"'{ADDENDUM_HEADING}' addendum — was the knowledge-graph check run?"))
    return findings


def check_cascade_covers_domains(root: str = ".", domains=DOMAINS) -> List[Finding]:
    index_path = os.path.join(root, "docs", "index.md")
    try:
        text = open(index_path, encoding="utf-8", errors="ignore").read()
    except OSError:
        return [Finding("knowledge: cascade root docs/index.md is missing — author it")]
    findings: List[Finding] = []
    for d in domains:
        if f"{d.slug}/" not in text:
            findings.append(Finding(
                f"knowledge: cascade root docs/index.md has no row for '{d.slug}/'"))
    return findings


def check_category_axis(root: str = ".") -> List[Finding]:
    """Cross-domain: every category USED by a capability or use-case must be DEFINED,
    not a reserved placeholder. Complements the per-domain 'unknown category' checks
    (which flag values not in the axis at all)."""
    try:
        used: Dict[str, int] = {}
        for node in (*load_capabilities(root), *load_use_cases(root)):
            if node.category:
                used[node.category] = used.get(node.category, 0) + 1
    except Exception as exc:  # non-blocking: a guard must never raise out
        return [Finding(f"knowledge: category-axis check failed: {exc}")]
    findings: List[Finding] = []
    for cat, n in sorted(used.items()):
        if cat in CATEGORIES and not category_defined(cat):
            findings.append(Finding(
                f"knowledge: category '{cat}' is in use ({n} node(s)) but has no "
                f"definition in tools/capability/reader.py — define it before use"))
    return findings


def run_all(root: str = ".") -> List[Finding]:
    specs = os.path.join(root, "docs/superpowers/specs")
    plans = os.path.join(root, "docs/superpowers/plans")
    findings: List[Finding] = []
    findings += check_cascade_covers_domains(root)
    findings += check_category_axis(root)
    findings += check_addendum_present(specs, plans)
    return findings
