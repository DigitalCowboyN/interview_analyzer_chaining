from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List

from tools.capability.reader import load_capabilities
from tools.usecase.reader import load_use_cases
from tools.testmap.reader import TEST_TYPES, Test, load_tests
from tools.testmap.render import render_index
from tools.testmap.verification import UNVERIFIED, verify_capabilities, verify_use_cases


@dataclass
class Finding:
    message: str


def check_test_type(tests: List[Test]) -> List[Finding]:
    return [
        Finding(f"test: {t.slug} has unknown test_type '{t.test_type}'")
        for t in tests
        if t.test_type not in TEST_TYPES
    ]


def check_unmapped(tests: List[Test]) -> List[Finding]:
    return [
        Finding(
            f"test: {t.slug} verifies nothing the graph can see "
            f"(no target, no marker)"
        )
        for t in tests
        if not t.target and not t.verifies
    ]


def check_unverified(uc_ver: Dict[str, str]) -> List[Finding]:
    return [
        Finding(f"test: use-case {slug} is UNVERIFIED — no test proves it")
        for slug, state in uc_ver.items()
        if state == UNVERIFIED
    ]


def check_index_sync(index_path: str, tests, cap_ver, uc_ver) -> List[Finding]:
    want = render_index(tests, cap_ver, uc_ver)
    have = (
        open(index_path, encoding="utf-8", errors="ignore").read()
        if os.path.exists(index_path)
        else ""
    )
    if want != have:
        return [
            Finding("test: docs/tests/index.md out of sync — run make testmap-index")
        ]
    return []


def run_all(root: str = ".") -> List[Finding]:
    tests = load_tests(root)
    caps = load_capabilities(root)
    ucs = load_use_cases(root)
    cap_ver = verify_capabilities(caps, tests)
    uc_ver = verify_use_cases(ucs, caps, tests)
    findings: List[Finding] = []
    findings += check_test_type(tests)
    findings += check_unmapped(tests)
    findings += check_unverified(uc_ver)
    findings += check_index_sync(
        os.path.join(root, "docs/tests/index.md"), tests, cap_ver, uc_ver
    )
    return findings
