from __future__ import annotations

import glob
import os
import re
import subprocess
from dataclasses import dataclass
from typing import Callable, List, Optional

from tools.adr.index import RESERVED, load_bundle, render_index, render_log
from tools.adr.model import VALID_STATUS, Adr

DECISION_MARKERS = ("decisions locked", "rejected alternative")
ADR_REF = re.compile(r"\bADR[-\s]?\d{1,4}\b|docs/adr/\d{4}", re.IGNORECASE)


@dataclass
class Finding:
    message: str


def check_structural(adrs: List[Adr]) -> List[Finding]:
    findings: List[Finding] = []
    seen: dict = {}
    for a in adrs:
        if a.id in seen:
            findings.append(Finding(f"duplicate id {a.id:04d}: {a.path} and {seen[a.id]}"))
        seen[a.id] = a.path
        if a.status not in VALID_STATUS:
            findings.append(Finding(f"{a.id:04d}: invalid status {a.status!r}"))
    by_id = {a.id: a for a in adrs}
    for a in adrs:
        for target in a.supersedes:
            other = by_id.get(target)
            if other is None:
                findings.append(Finding(f"{a.id:04d} supersedes unknown {target:04d}"))
            elif a.id not in other.superseded_by:
                findings.append(
                    Finding(f"{a.id:04d} supersedes {target:04d} but {target:04d}.superseded_by lacks it")
                )
    return findings


def check_generated_in_sync(adr_dir: str, adrs: List[Adr]) -> List[Finding]:
    findings: List[Finding] = []
    for name, render in (("index.md", render_index), ("log.md", render_log)):
        path = os.path.join(adr_dir, name)
        want = render(adrs)
        have = open(path, encoding="utf-8").read() if os.path.exists(path) else ""
        if want != have:
            findings.append(Finding(f"{name} out of sync — run `make adr-index`"))
    return findings


def check_specs_reference_adr(specs_dir: str) -> List[Finding]:
    findings: List[Finding] = []
    for path in sorted(glob.glob(os.path.join(specs_dir, "*.md"))):
        text = open(path, encoding="utf-8").read()
        low = text.lower()
        if any(m in low for m in DECISION_MARKERS) and not ADR_REF.search(text):
            findings.append(Finding(f"{os.path.basename(path)} locks decisions but references no ADR"))
    return findings


def git_committer_ts(path: str) -> Optional[int]:
    try:
        out = subprocess.run(
            ["git", "log", "-1", "--format=%ct", "--", path],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        return int(out) if out else None
    except Exception:
        return None


def check_staleness(adrs: List[Adr],
                    ts_fn: Callable[[str], Optional[int]] = git_committer_ts) -> List[Finding]:
    findings: List[Finding] = []
    for a in adrs:
        if not a.source or not a.path:
            continue
        src_ts, adr_ts = ts_fn(a.source), ts_fn(a.path)
        if src_ts is not None and adr_ts is not None and src_ts > adr_ts:
            findings.append(Finding(f"{a.id:04d}: source {a.source} changed after the ADR"))
    return findings


def run_all(adr_dir: str, specs_dir: str) -> List[Finding]:
    adrs = load_bundle(adr_dir)
    findings: List[Finding] = []
    findings += check_structural(adrs)
    findings += check_generated_in_sync(adr_dir, adrs)
    findings += check_specs_reference_adr(specs_dir)
    findings += check_staleness(adrs)
    return findings
