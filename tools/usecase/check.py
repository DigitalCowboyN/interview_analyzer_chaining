"""Use-case domain guards: validate form/category axis membership, flag use-cases
missing acceptance criteria or with no fulfilling capability, and confirm
docs/use-cases/index.md is in sync. Non-blocking — findings are warnings, not build
failures.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List

from tools.capability.reader import CATEGORIES, load_capabilities, real_code_units
from tools.usecase.coverage import NOT_COVERED, coverage
from tools.usecase.reader import FORMS, UseCase, load_use_cases
from tools.usecase.render import render_index


@dataclass
class Finding:
    message: str


def check_forms(ucs: List[UseCase]) -> List[Finding]:
    return [
        Finding(f"use-case: {u.slug} has unknown form '{u.form}'")
        for u in ucs
        if u.form not in FORMS
    ]


def check_categories(ucs: List[UseCase]) -> List[Finding]:
    return [
        Finding(f"use-case: {u.slug} has unknown category '{u.category}'")
        for u in ucs
        if u.category not in CATEGORIES
    ]


def check_acceptance_criteria(ucs: List[UseCase]) -> List[Finding]:
    return [
        Finding(f"use-case: {u.slug} has no acceptance_criteria yet")
        for u in ucs
        if not u.acceptance_criteria
    ]


def check_uncovered(ucs: List[UseCase], cov: Dict[str, str]) -> List[Finding]:
    return [
        Finding(f"use-case: {u.slug} is NOT_COVERED — no capability fulfills it")
        for u in ucs
        if cov.get(u.slug) == NOT_COVERED
    ]


def check_index_sync(
    index_path: str, ucs: List[UseCase], cov: Dict[str, str]
) -> List[Finding]:
    want = render_index(ucs, cov)
    have = (
        open(index_path, encoding="utf-8", errors="ignore").read()
        if os.path.exists(index_path)
        else ""
    )
    if want != have:
        return [
            Finding(
                "use-case: docs/use-cases/index.md out of sync — run make usecase-index"
            )
        ]
    return []


def run_all(root: str = ".") -> List[Finding]:
    ucs = load_use_cases(root)
    caps = load_capabilities(root)
    cov = coverage(ucs, caps, real_code_units(root))
    findings: List[Finding] = []
    findings += check_forms(ucs)
    findings += check_categories(ucs)
    findings += check_acceptance_criteria(ucs)
    findings += check_uncovered(ucs, cov)
    findings += check_index_sync(
        os.path.join(root, "docs/use-cases/index.md"), ucs, cov
    )
    return findings
