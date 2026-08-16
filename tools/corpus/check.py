from __future__ import annotations

import collections
from dataclasses import dataclass
from typing import List

from src.ingestion.front_matter import parse_front_matter
from tools.corpus.model import OKF_HOMES
from tools.corpus.reader import _IGNORE_DIRS, _iter_markdown, okf_records


@dataclass
class Finding:
    message: str


def check_misfiled(records) -> List[Finding]:
    """A record whose path is outside its type's home directory is misfiled — the blind spot
    type-primary intake exists to catch (found by type, judged by home)."""
    out: List[Finding] = []
    for r in records:
        home = OKF_HOMES[r.type]
        p = r.path.replace("\\", "/")
        if not p.startswith(home.rstrip("/") + "/"):
            out.append(Finding(
                f"corpus: {r.type} '{r.id}' is at {p} — outside its home {home}/ (misfiled)"))
    return out


def check_unregistered_types(root: str = ".") -> List[Finding]:
    """A `.md` whose OWN top frontmatter declares a `type:` that is not a registered document
    type. okf_records silently skips these, so a new *kind* of record is invisible until wired
    in. This surfaces it. (Declared-type detection; undeclared new domains stay the hard case.)"""
    unknown = collections.Counter()
    for path in _iter_markdown(root, _IGNORE_DIRS):
        try:
            fm, _ = parse_front_matter(open(path, encoding="utf-8", errors="ignore").read())
        except OSError:
            continue
        t = fm.get("type") if fm else None
        if t and t not in OKF_HOMES:
            unknown[t] += 1
    return [Finding(f"corpus: '{t}' is declared as a type on {n} file(s) but is not a registered "
                    f"node type — wire it in, or it stays invisible to the graph")
            for t, n in sorted(unknown.items())]


def run_all(root: str = ".") -> List[Finding]:
    return check_misfiled(okf_records(root)) + check_unregistered_types(root)
