"""Use-case domain coverage: derives each use-case's NOT_COVERED /
PARTIALLY_COVERED / FULLY_COVERED state, transitively through the capabilities
that fulfill it and whether those capabilities have resolving code.
"""
from __future__ import annotations

from typing import Dict, List, Set

from tools.capability.reader import Capability
from tools.usecase.reader import UseCase

NOT_COVERED = "NOT_COVERED"
PARTIALLY_COVERED = "PARTIALLY_COVERED"
FULLY_COVERED = "FULLY_COVERED"


def _implemented(cap: Capability, valid_units: Set[str]) -> bool:
    return any(u in valid_units for u in cap.implemented_by)


def coverage(
    use_cases: List[UseCase], capabilities: List[Capability], valid_units: Set[str]
) -> Dict[str, str]:
    """Derived coverage state per use-case, transitive through capabilities.

    NOT_COVERED   — no capability fulfills the intent.
    FULLY_COVERED — every fulfilling capability is implemented (has resolving code).
    PARTIALLY_COVERED — fulfilled, but at least one fulfilling capability is unbuilt.
    """
    by_slug = {c.slug: c for c in capabilities}
    out: Dict[str, str] = {}
    for uc in use_cases:
        fulfilling = [by_slug[s] for s in uc.fulfilled_by if s in by_slug]
        if not fulfilling:
            out[uc.slug] = NOT_COVERED
        elif all(_implemented(c, valid_units) for c in fulfilling):
            out[uc.slug] = FULLY_COVERED
        else:
            out[uc.slug] = PARTIALLY_COVERED
    return out
