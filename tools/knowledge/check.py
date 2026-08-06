from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass
from typing import List, Tuple

# Single source of truth for the knowledge-graph domains: (docs slug, make-name).
# Each has docs/<slug>/index.md and a `make <make-name>-check`. Add a row here (and
# to docs/index.md) when a new domain ships (e.g. capabilities).
DOMAINS: List[Tuple[str, str]] = [
    ("adr", "adr"),
    ("api", "api"),
    ("cli", "cli"),
    ("code", "code"),
    ("capabilities", "capability"),
    ("glossary", "glossary"),
    ("graph", "graph"),
    ("graph-queries", "graphq"),
    ("prompts", "prompt"),
    ("use-cases", "usecase"),
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
    for slug, _make in domains:
        if f"{slug}/" not in text:
            findings.append(Finding(
                f"knowledge: cascade root docs/index.md has no row for '{slug}/'"))
    return findings


def run_all(root: str = ".") -> List[Finding]:
    specs = os.path.join(root, "docs/superpowers/specs")
    plans = os.path.join(root, "docs/superpowers/plans")
    findings: List[Finding] = []
    findings += check_cascade_covers_domains(root)
    findings += check_addendum_present(specs, plans)
    return findings
