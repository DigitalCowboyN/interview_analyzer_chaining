"""Knowledge domain surfaces: maps a set of changed file paths to the
`make <domain>-check` targets whose surface they touch, driving the
changed-domain pre-commit nudge.
"""
from __future__ import annotations

from typing import Iterable, List

from tools.knowledge.check import DOMAINS


def changed_domains(files: Iterable[str], domains=DOMAINS) -> List[str]:
    """The `make`-names of domains whose surface any of `files` touches (sorted, deduped)."""
    hit = set()
    for f in files:
        f = f.replace("\\", "/")
        for d in domains:
            if any(f.startswith(p) for p in d.surfaces):
                hit.add(d.make)
    return sorted(hit)
