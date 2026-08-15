from __future__ import annotations

from dataclasses import dataclass
from typing import List

from tools.corpus.model import OKF_HOMES
from tools.corpus.reader import okf_records


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


def run_all(root: str = ".") -> List[Finding]:
    return check_misfiled(okf_records(root))
