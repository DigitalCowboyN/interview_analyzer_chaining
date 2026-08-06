from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass, field
from typing import List, Set, Tuple

from tools.capability.reader import real_code_units

# test_type is an open ordered set (reserved: acceptance, contract). Add a value here.
TEST_TYPES = ["unit", "integration", "e2e"]

_VERIFIES = re.compile(r"^#\s*verifies:\s*(\S+)", re.MULTILINE)
_TESTFN = re.compile(r"^\s*def test_", re.MULTILINE)
_E2E = re.compile(r"(test_e2e_|test_end_to_end_|_smoke)")

_TESTS_ADDR = "tests"  # this domain's node-address prefix
_CODE_ADDR = (
    "code"  # the code domain's prefix (hardcoded to avoid importing tools.graph)
)


@dataclass
class Test:
    slug: str
    path: str
    test_type: str
    target: str  # derived code-unit slug ("" if unresolved)
    verifies: List[str] = field(
        default_factory=list
    )  # authored "<domain>:<id>" markers
    n_tests: int = 0


def _test_type(rel: str) -> str:
    seg = rel.split(os.sep, 1)[0]
    if seg == "integration":
        return "e2e" if _E2E.search(os.path.basename(rel)) else "integration"
    return "unit"


def _target(rel: str, units: Set[str]) -> str:
    seg = rel.split(os.sep, 1)[0]
    if seg in units:
        return seg
    if f"tools.{seg}" in units:
        return f"tools.{seg}"
    return ""


def load_tests(root: str = ".", tests_dir: str = "tests") -> List[Test]:
    units = real_code_units(root)
    base = os.path.join(root, tests_dir)
    tests: List[Test] = []
    for path in sorted(
        glob.glob(os.path.join(base, "**", "test_*.py"), recursive=True)
    ):
        rel = os.path.relpath(path, base)
        if rel.startswith("fixtures" + os.sep) or "__pycache__" in path:
            continue
        try:
            text = open(path, encoding="utf-8", errors="ignore").read()
        except OSError:
            continue
        tests.append(
            Test(
                slug=os.path.splitext(rel)[0].replace(os.sep, "."),
                path=os.path.join(tests_dir, rel),
                test_type=_test_type(rel),
                target=_target(rel, units),
                verifies=_VERIFIES.findall(text),
                n_tests=len(_TESTFN.findall(text)),
            )
        )
    return tests


def verifies_edges(root: str = ".") -> List[Tuple[str, str, str]]:
    """(src_addr, dst_addr, test_type): derived→code by convention + authored→intent by marker."""
    out: List[Tuple[str, str, str]] = []
    for t in load_tests(root):
        src = f"{_TESTS_ADDR}:{t.slug}"
        if t.target:
            out.append((src, f"{_CODE_ADDR}:{t.target}", t.test_type))
        for marker in t.verifies:
            out.append((src, marker, t.test_type))
    return out
