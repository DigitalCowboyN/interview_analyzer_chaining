"""Derived verification axis — is a node PROVEN by tests, orthogonal to whether it is implemented.

States `UNVERIFIED | PARTIALLY_VERIFIED | VERIFIED`, computed from the `verifies` edges
(test→code by convention, test→intent by authored marker) and rolled up transitively
through capabilities. Pure — callers supply the loaded reader objects.
"""

from __future__ import annotations

from typing import Dict, List, Set

from tools.capability.reader import Capability
from tools.usecase.reader import UseCase
from tools.testmap.reader import Test

UNVERIFIED = "UNVERIFIED"
PARTIALLY_VERIFIED = "PARTIALLY_VERIFIED"
VERIFIED = "VERIFIED"


def verified_units(tests: List[Test]) -> Set[str]:
    """Code-unit slugs that at least one test verifies (the derived test→code targets)."""
    return {t.target for t in tests if t.target}


def _direct(tests: List[Test]) -> Set[str]:
    out: Set[str] = set()
    for t in tests:
        out.update(t.verifies)
    return out


def _capability_state(cap: Capability, vunits: Set[str], direct: Set[str]) -> str:
    if f"capabilities:{cap.slug}" in direct:
        return VERIFIED
    units = cap.implemented_by
    if not units:
        return UNVERIFIED
    hit = [u for u in units if u in vunits]
    if len(hit) == len(units):
        return VERIFIED
    return PARTIALLY_VERIFIED if hit else UNVERIFIED


def verify_capabilities(caps: List[Capability], tests: List[Test]) -> Dict[str, str]:
    """Derived verification state per capability: VERIFIED when every `implemented_by`
    unit is tested (or a direct marker names it), PARTIALLY when some are, else UNVERIFIED.
    """
    vunits, direct = verified_units(tests), _direct(tests)
    return {c.slug: _capability_state(c, vunits, direct) for c in caps}


def verify_use_cases(
    use_cases: List[UseCase], caps: List[Capability], tests: List[Test]
) -> Dict[str, str]:
    """Derived verification state per use-case: VERIFIED via a direct `use-cases:<slug>`
    marker or when every fulfilling capability is VERIFIED; PARTIALLY when some are; else
    UNVERIFIED. Rolls up the capability states, so a direct marker on a fulfilling
    capability propagates automatically."""
    direct = _direct(tests)
    cap_state = verify_capabilities(caps, tests)
    known = {c.slug for c in caps}
    out: Dict[str, str] = {}
    for uc in use_cases:
        if f"use-cases:{uc.slug}" in direct:
            out[uc.slug] = VERIFIED
            continue
        states = [cap_state[s] for s in uc.fulfilled_by if s in known]
        if states and all(s == VERIFIED for s in states):
            out[uc.slug] = VERIFIED
        elif any(s != UNVERIFIED for s in states):
            out[uc.slug] = PARTIALLY_VERIFIED
        else:
            out[uc.slug] = UNVERIFIED
    return out
