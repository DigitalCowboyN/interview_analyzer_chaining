# Tests Domain Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the tests domain — derived `Test` nodes per file, a polymorphic `verifies` edge (derived→code, authored→intent), and an orthogonal verification axis — closing the Requirements Traceability Matrix.

**Architecture:** A new `tools/testmap/` package (`reader → verification → render → check → __main__`) that derives `Test` nodes from the `tests/` suite, plus activation of the graph's reserved `Test` node and `verifies` edge. Verification (`UNVERIFIED/PARTIALLY_VERIFIED/VERIFIED`) is derived and orthogonal to implementation coverage; it lives in the tests domain and touches no other domain's files.

**Tech Stack:** Python 3 (stdlib + `yaml` via existing readers), pytest, Makefile. No new deps.

**Spec:** `docs/superpowers/specs/2026-08-06-tests-domain-design.md`.

## Global Constraints

- **Non-blocking:** every `check_*`/`run_all` returns `list[Finding]` and never raises; every CLI `cmd_check` returns `0`.
- **Derived, not authored, nodes:** `Test` nodes come from scanning `tests/`; no Test markdown files. The only authored surface is the `# verifies: <domain>:<id>` marker inside real test files.
- **`test_type` is an open ordered set** `TEST_TYPES = ["unit", "integration", "e2e"]` in `tools/testmap/reader.py` (reserving `acceptance`/`contract`); an out-of-set value is flagged, never rejected.
- **Verification is derived and orthogonal.** Implementation coverage (`tools/usecase/coverage.py`) is NOT modified. No stored verification field anywhere.
- **Do not modify the use-cases or capabilities domains' logic or generated files.** The only capability write is the additive `map-the-tests` self-registration node (Task 7). The verification view lives only in `docs/tests/`.
- **`verifies` is a derived graph edge** whose endpoints are fully `<domain>:<id>`-addressed; reuse the graph's existing `check_endpoints`. `tools/testmap` must NOT import `tools.graph` (avoid a cycle) — it hardcodes the `tests:` and `code:` address prefixes.
- **Naming (verbatim):** package `tools/testmap/`; docs `docs/tests/`; slug `tests`; node type `Test`; addressing `tests:<slug>` (path under `tests/`, `/`→`.`, no `.py`); make targets `testmap-index`/`testmap-check`; `DOMAINS` entry `("tests", "testmap")`.
- **Names used across tasks:** `Test`, `TEST_TYPES`, `load_tests`, `verifies_edges`, `verified_units`, `verify_capabilities`, `verify_use_cases`, `UNVERIFIED`, `PARTIALLY_VERIFIED`, `VERIFIED`, `render_index(tests, cap_ver, uc_ver)`.

---

### Task 1: Test reader + verifies_edges

**Files:**
- Create: `tools/testmap/__init__.py` (empty), `tools/testmap/reader.py`
- Test: `tests/testmap/__init__.py` (empty), `tests/testmap/test_reader.py`

**Interfaces:**
- Consumes: `tools.capability.reader.real_code_units(root) -> set`.
- Produces: `TEST_TYPES`; `Test` dataclass; `load_tests(root=".", tests_dir="tests") -> list[Test]`; `verifies_edges(root=".") -> list[tuple[str,str,str]]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/testmap/test_reader.py
from tools.testmap.reader import TEST_TYPES, Test, load_tests, verifies_edges


def _seed(tmp_path):
    t = tmp_path / "tests"
    (t / "capability").mkdir(parents=True)
    (t / "capability" / "test_check.py").write_text(
        "def test_a():\n    pass\n\ndef test_b():\n    pass\n", encoding="utf-8")
    (t / "integration").mkdir(parents=True)
    (t / "integration" / "test_e2e_user_edits.py").write_text(
        '"""e2e."""\n# verifies: use-cases:correct-what-the-system-got-wrong\n'
        "def test_flow():\n    pass\n", encoding="utf-8")
    (t / "integration" / "test_api_calls.py").write_text(
        "def test_call():\n    pass\n", encoding="utf-8")
    # a real tools package DIR so real_code_units()/packages() resolves the target
    # (resolution scans src/ + tools/ dirs, NOT docs/code nodes):
    (tmp_path / "tools" / "capability").mkdir(parents=True)


def test_type_and_target_derivation(tmp_path):
    _seed(tmp_path)
    tests = {t.slug: t for t in load_tests(str(tmp_path))}
    assert tests["capability.test_check"].test_type == "unit"
    assert tests["capability.test_check"].target == "tools.capability"
    assert tests["capability.test_check"].n_tests == 2
    assert tests["integration.test_e2e_user_edits"].test_type == "e2e"
    assert tests["integration.test_api_calls"].test_type == "integration"
    assert tests["integration.test_e2e_user_edits"].verifies == [
        "use-cases:correct-what-the-system-got-wrong"]
    assert "unit" in TEST_TYPES


def test_verifies_edges(tmp_path):
    _seed(tmp_path)
    edges = set(verifies_edges(str(tmp_path)))
    assert ("tests:capability.test_check", "code:tools.capability", "unit") in edges
    assert ("tests:integration.test_e2e_user_edits",
            "use-cases:correct-what-the-system-got-wrong", "e2e") in edges
    # unresolved target (no code unit for 'integration') emits no derived→code edge
    assert not any(s == "tests:integration.test_api_calls" and d.startswith("code:")
                   for s, d, _ in edges)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/testmap/test_reader.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tools.testmap'`.

- [ ] **Step 3: Write minimal implementation**

```python
# tools/testmap/reader.py
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

_TESTS_ADDR = "tests"   # this domain's node-address prefix
_CODE_ADDR = "code"     # the code domain's prefix (hardcoded to avoid importing tools.graph)


@dataclass
class Test:
    slug: str
    path: str
    test_type: str
    target: str                                 # derived code-unit slug ("" if unresolved)
    verifies: List[str] = field(default_factory=list)   # authored "<domain>:<id>" markers
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
    for path in sorted(glob.glob(os.path.join(base, "**", "test_*.py"), recursive=True)):
        rel = os.path.relpath(path, base)
        if rel.startswith("fixtures" + os.sep) or "__pycache__" in path:
            continue
        try:
            text = open(path, encoding="utf-8", errors="ignore").read()
        except OSError:
            continue
        tests.append(Test(
            slug=os.path.splitext(rel)[0].replace(os.sep, "."),
            path=os.path.join(tests_dir, rel),
            test_type=_test_type(rel),
            target=_target(rel, units),
            verifies=_VERIFIES.findall(text),
            n_tests=len(_TESTFN.findall(text)),
        ))
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
```

Also create empty `tools/testmap/__init__.py` and `tests/testmap/__init__.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/testmap/test_reader.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add tools/testmap/__init__.py tools/testmap/reader.py tests/testmap/
git commit -m "feat(testmap): Test reader — derived per file, verifies_edges (derived+authored)"
```

---

### Task 2: Verification axis

**Files:**
- Create: `tools/testmap/verification.py`
- Test: `tests/testmap/test_verification.py`

**Interfaces:**
- Consumes: `tools.capability.reader.Capability` (`.slug`, `.implemented_by`); `tools.usecase.reader.UseCase` (`.slug`, `.fulfilled_by`); `tools.testmap.reader.Test`.
- Produces: `UNVERIFIED`, `PARTIALLY_VERIFIED`, `VERIFIED`; `verified_units(tests) -> set`; `verify_capabilities(caps, tests) -> dict`; `verify_use_cases(use_cases, caps, tests) -> dict`.

- [ ] **Step 1: Write the failing test**

```python
# tests/testmap/test_verification.py
from tools.capability.reader import Capability
from tools.usecase.reader import UseCase
from tools.testmap.reader import Test
from tools.testmap.verification import (
    UNVERIFIED, PARTIALLY_VERIFIED, VERIFIED,
    verified_units, verify_capabilities, verify_use_cases,
)


def _test(slug, target="", verifies=None, tt="unit"):
    return Test(slug=slug, path="p", test_type=tt, target=target, verifies=verifies or [])


def _cap(slug, impl):
    return Capability(slug=slug, kind="primary", tier="core", parent="",
                      implemented_by=impl, statement="s", path="p", category="product")


def _uc(slug, fulfilled_by):
    return UseCase(slug=slug, form="use-case", category="product", actor="a",
                   statement="s", path="p", fulfilled_by=fulfilled_by)


def test_verified_units_and_capability():
    tests = [_test("t1", target="api"), _test("t2", target="lens")]
    assert verified_units(tests) == {"api", "lens"}
    caps = [_cap("built", ["api"]), _cap("half", ["api", "unt"]), _cap("none", ["x"]),
            _cap("bare", [])]
    cv = verify_capabilities(caps, tests)
    assert cv["built"] == VERIFIED
    assert cv["half"] == PARTIALLY_VERIFIED
    assert cv["none"] == UNVERIFIED
    assert cv["bare"] == UNVERIFIED


def test_direct_marker_on_capability_verifies_it():
    tests = [_test("t", verifies=["capabilities:c"])]
    cv = verify_capabilities([_cap("c", [])], tests)
    assert cv["c"] == VERIFIED   # direct marker beats empty implemented_by


def test_use_case_rollup_and_direct():
    tests = [_test("u", target="api"),
             _test("acc", verifies=["use-cases:direct"], tt="e2e")]
    caps = [_cap("built", ["api"]), _cap("unbuilt", ["x"])]
    ucs = [_uc("proven", ["built"]), _uc("partial", ["built", "unbuilt"]),
           _uc("none", ["unbuilt"]), _uc("bare", []), _uc("direct", [])]
    uv = verify_use_cases(ucs, caps, tests)
    assert uv["proven"] == VERIFIED
    assert uv["partial"] == PARTIALLY_VERIFIED
    assert uv["none"] == UNVERIFIED
    assert uv["bare"] == UNVERIFIED
    assert uv["direct"] == VERIFIED     # direct acceptance marker, no fulfilling cap needed
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/testmap/test_verification.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tools.testmap.verification'`.

- [ ] **Step 3: Write minimal implementation**

```python
# tools/testmap/verification.py
from __future__ import annotations

from typing import Dict, List, Set

from tools.capability.reader import Capability
from tools.usecase.reader import UseCase
from tools.testmap.reader import Test

UNVERIFIED = "UNVERIFIED"
PARTIALLY_VERIFIED = "PARTIALLY_VERIFIED"
VERIFIED = "VERIFIED"


def verified_units(tests: List[Test]) -> Set[str]:
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
    vunits, direct = verified_units(tests), _direct(tests)
    return {c.slug: _capability_state(c, vunits, direct) for c in caps}


def verify_use_cases(use_cases: List[UseCase], caps: List[Capability],
                     tests: List[Test]) -> Dict[str, str]:
    vunits, direct = verified_units(tests), _direct(tests)
    cap_state = {c.slug: _capability_state(c, vunits, direct) for c in caps}
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/testmap/test_verification.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add tools/testmap/verification.py tests/testmap/test_verification.py
git commit -m "feat(testmap): orthogonal verification axis — UNVERIFIED/PARTIALLY/VERIFIED, transitive"
```

---

### Task 3: Renderer

**Files:**
- Create: `tools/testmap/render.py`
- Test: `tests/testmap/test_render.py`

**Interfaces:**
- Consumes: `tools.testmap.reader.TEST_TYPES`, `Test`.
- Produces: `render_index(tests, cap_ver, uc_ver) -> str` (pure, deterministic, single trailing newline).

- [ ] **Step 1: Write the failing test**

```python
# tests/testmap/test_render.py
from tools.testmap.reader import Test
from tools.testmap.render import render_index
from tools.testmap.verification import VERIFIED, UNVERIFIED


def _t(slug, tt, target="", verifies=None, n=1):
    return Test(slug=slug, path="tests/" + slug.replace(".", "/") + ".py",
                test_type=tt, target=target, verifies=verifies or [], n_tests=n)


def test_groups_by_type_and_shows_rollup():
    tests = [_t("cap.test_x", "unit", target="tools.capability"),
             _t("integration.test_e2e", "e2e", verifies=["use-cases:uc1"])]
    out = render_index(tests, {"capA": VERIFIED}, {"uc1": VERIFIED, "uc2": UNVERIFIED})
    assert out.startswith("# Tests")
    assert "## unit" in out and "## e2e" in out
    assert "cap.test_x" in out and "tools.capability" in out
    assert "use-cases:uc1" in out                 # authored marker shown
    assert "## Verification rollup" in out
    assert "uc1" in out and "VERIFIED" in out and "uc2" in out and "UNVERIFIED" in out
    assert out.endswith("\n") and not out.endswith("\n\n")


def test_empty_type_omitted():
    out = render_index([_t("a.test_a", "unit")], {}, {})
    assert "## integration" not in out and "## e2e" not in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/testmap/test_render.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tools.testmap.render'`.

- [ ] **Step 3: Write minimal implementation**

```python
# tools/testmap/render.py
from __future__ import annotations

from typing import Dict, List

from tools.testmap.reader import TEST_TYPES, Test


def render_index(tests: List[Test], cap_ver: Dict[str, str], uc_ver: Dict[str, str]) -> str:
    lines = [
        "# Tests", "",
        "The test suite as a graph node set, and what it verifies (`../code/`, "
        "`../capabilities/`, `../use-cases/`). Verification is derived, orthogonal to "
        "implementation coverage.", "",
    ]
    for tt in TEST_TYPES:
        group = sorted((t for t in tests if t.test_type == tt), key=lambda t: t.slug)
        if not group:
            continue
        lines.append(f"## {tt}")
        lines.append("")
        for t in group:
            target = t.target or "—"
            verifies = ", ".join(t.verifies) or "—"
            lines.append(f"- `{t.slug}` ({t.n_tests}) → {target}  ·  verifies: {verifies}")
        lines.append("")

    lines.append("## Verification rollup")
    lines.append("")
    lines.append("Use-cases:")
    for slug in sorted(uc_ver):
        lines.append(f"- {slug}: {uc_ver[slug]}")
    lines.append("")
    lines.append("Capabilities:")
    for slug in sorted(cap_ver):
        lines.append(f"- {slug}: {cap_ver[slug]}")

    return "\n".join(lines).rstrip() + "\n"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/testmap/test_render.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add tools/testmap/render.py tests/testmap/test_render.py
git commit -m "feat(testmap): render_index — suite grouped by test_type + verification rollup"
```

---

### Task 4: Guard (check)

**Files:**
- Create: `tools/testmap/check.py`
- Test: `tests/testmap/test_check.py`

**Interfaces:**
- Consumes: `tools.capability.reader.load_capabilities`; `tools.usecase.reader.load_use_cases`; `tools.testmap.reader.TEST_TYPES`, `load_tests`; `tools.testmap.verification.UNVERIFIED`, `verify_capabilities`, `verify_use_cases`; `tools.testmap.render.render_index`.
- Produces: `Finding`; `check_test_type`, `check_unmapped`, `check_unverified`, `check_index_sync`, `run_all(root=".")`.

- [ ] **Step 1: Write the failing test**

```python
# tests/testmap/test_check.py
from tools.testmap.reader import Test
from tools.testmap.verification import UNVERIFIED, VERIFIED
from tools.testmap.check import (
    check_test_type, check_unmapped, check_unverified, run_all,
)


def _t(slug, tt="unit", target="", verifies=None):
    return Test(slug=slug, path="p", test_type=tt, target=target, verifies=verifies or [])


def test_check_test_type_flags_unknown():
    assert check_test_type([_t("a", tt="fuzz")])
    assert check_test_type([_t("a", tt="unit")]) == []


def test_check_unmapped_flags_targetless_markerless():
    assert check_unmapped([_t("orphan")])                       # no target, no marker
    assert check_unmapped([_t("ok", target="api")]) == []
    assert check_unmapped([_t("ok2", verifies=["use-cases:x"])]) == []


def test_check_unverified_flags_unverified_use_cases():
    flagged = check_unverified({"a": UNVERIFIED, "b": VERIFIED})
    assert len(flagged) == 1 and "a" in flagged[0].message


def test_run_all_never_raises_on_real_repo():
    assert isinstance(run_all("."), list)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/testmap/test_check.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tools.testmap.check'`.

- [ ] **Step 3: Write minimal implementation**

```python
# tools/testmap/check.py
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
    return [Finding(f"test: {t.slug} has unknown test_type '{t.test_type}'")
            for t in tests if t.test_type not in TEST_TYPES]


def check_unmapped(tests: List[Test]) -> List[Finding]:
    return [Finding(f"test: {t.slug} verifies nothing the graph can see "
                    f"(no target, no marker)")
            for t in tests if not t.target and not t.verifies]


def check_unverified(uc_ver: Dict[str, str]) -> List[Finding]:
    return [Finding(f"test: use-case {slug} is UNVERIFIED — no test proves it")
            for slug, state in uc_ver.items() if state == UNVERIFIED]


def check_index_sync(index_path: str, tests, cap_ver, uc_ver) -> List[Finding]:
    want = render_index(tests, cap_ver, uc_ver)
    have = open(index_path, encoding="utf-8", errors="ignore").read() if os.path.exists(index_path) else ""
    if want != have:
        return [Finding("test: docs/tests/index.md out of sync — run make testmap-index")]
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
    findings += check_index_sync(os.path.join(root, "docs/tests/index.md"), tests, cap_ver, uc_ver)
    return findings
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/testmap/test_check.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add tools/testmap/check.py tests/testmap/test_check.py
git commit -m "feat(testmap): non-blocking guard — test_type/unmapped/unverified/index-sync"
```

---

### Task 5: CLI

**Files:**
- Create: `tools/testmap/__main__.py`
- Test: `tests/testmap/test_cli.py`

**Interfaces:**
- Consumes: all `tools.testmap.*`; `tools.capability.reader.load_capabilities`; `tools.usecase.reader.load_use_cases`.
- Produces: `python -m tools.testmap {index | check | verification}`; `main(argv=None) -> int`; `INDEX = "docs/tests/index.md"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/testmap/test_cli.py
from tools.testmap.__main__ import main


def test_check_returns_zero(capsys):
    assert main(["check"]) == 0
    assert "testmap-check:" in capsys.readouterr().out


def test_verification_runs(capsys):
    assert main(["verification"]) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/testmap/test_cli.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tools.testmap.__main__'`.

- [ ] **Step 3: Write minimal implementation**

```python
# tools/testmap/__main__.py
from __future__ import annotations

import argparse
import os
import sys

from tools.capability.reader import load_capabilities
from tools.usecase.reader import load_use_cases
from tools.testmap.check import run_all
from tools.testmap.reader import load_tests
from tools.testmap.render import render_index
from tools.testmap.verification import verify_capabilities, verify_use_cases

INDEX = "docs/tests/index.md"


def _rollups(root: str = "."):
    tests = load_tests(root)
    caps = load_capabilities(root)
    ucs = load_use_cases(root)
    return tests, verify_capabilities(caps, tests), verify_use_cases(ucs, caps, tests)


def cmd_index(args) -> int:
    os.makedirs(os.path.dirname(INDEX), exist_ok=True)
    tests, cap_ver, uc_ver = _rollups()
    with open(INDEX, "w", encoding="utf-8") as fh:
        fh.write(render_index(tests, cap_ver, uc_ver))
    print(f"wrote {INDEX}")
    return 0


def cmd_check(args) -> int:
    findings = run_all()
    if findings:
        print(f"testmap-check: {len(findings)} warning(s):")
        for f in findings:
            print(f"  - {f.message}")
    else:
        print("testmap-check: clean")
    return 0  # NON-BLOCKING


def cmd_verification(args) -> int:
    _, cap_ver, uc_ver = _rollups()
    for slug in sorted(uc_ver):
        print(f"use-case      {uc_ver[slug]:18} {slug}")
    for slug in sorted(cap_ver):
        print(f"capability    {cap_ver[slug]:18} {slug}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="tools.testmap")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("index")
    sub.add_parser("check")
    sub.add_parser("verification")
    args = parser.parse_args(argv)
    return {"index": cmd_index, "check": cmd_check, "verification": cmd_verification}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/testmap/test_cli.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add tools/testmap/__main__.py tests/testmap/test_cli.py
git commit -m "feat(testmap): CLI — index | check | verification (non-blocking)"
```

---

### Task 6: Activate the `Test` node + `verifies` edge in the graph

**Files:**
- Modify: `tools/graph/registry.py` (node domain + edge type), `tools/graph/reader.py` (adapter + derived handler)
- Test: `tests/graph/test_verifies_edge.py`

**Interfaces:**
- Consumes: `tools.testmap.reader.load_tests`, `verifies_edges`.
- Produces: `NODE_DOMAINS["Test"]="tests"`; a derived `verifies` `EdgeType` with a `test_type` property; harvest emits `Edge("verifies", "tests:<slug>", "<domain>:<id>", {"test_type": ...})`.

- [ ] **Step 1: Write the failing test**

```python
# tests/graph/test_verifies_edge.py
from tools.graph.reader import harvest, nodes
from tools.graph.check import check_endpoints


def _seed(tmp_path):
    caps = tmp_path / "docs" / "capabilities"; caps.mkdir(parents=True)
    (caps / "map-the-tests.md").write_text(
        "---\ntype: Capability\nkind: primary\ntier: core\ncategory: operations\n"
        "implemented_by: [tools.capability]\n---\nMap tests.\n", encoding="utf-8")
    code = tmp_path / "docs" / "code"; code.mkdir(parents=True)
    (code / "tools.capability.md").write_text(
        "---\ntype: CodeUnit\nunit: tools.capability\nrole: tooling\n---\nx\n", encoding="utf-8")
    # real tools dir so target resolution (packages()) sees tools.capability, in addition
    # to the docs/code node above (which drives the CodeUnit node inventory):
    (tmp_path / "tools" / "capability").mkdir(parents=True)
    t = tmp_path / "tests" / "capability"; t.mkdir(parents=True)
    (t / "test_check.py").write_text("def test_a():\n    pass\n", encoding="utf-8")
    ti = tmp_path / "tests" / "integration"; ti.mkdir(parents=True)
    (ti / "test_e2e_x.py").write_text(
        "# verifies: capabilities:map-the-tests\ndef test_flow():\n    pass\n", encoding="utf-8")


def test_verifies_harvested_with_test_type(tmp_path):
    _seed(tmp_path)
    edges = harvest(str(tmp_path))
    ve = [e for e in edges if e.type == "verifies"]
    assert any(e.src == "tests:capability.test_check" and e.dst == "code:tools.capability"
               and e.props.get("test_type") == "unit" for e in ve)
    assert any(e.src == "tests:integration.test_e2e_x"
               and e.dst == "capabilities:map-the-tests"
               and e.props.get("test_type") == "e2e" for e in ve)
    assert "capability.test_check" in nodes(str(tmp_path))["Test"]


def test_dangling_marker_flagged(tmp_path):
    _seed(tmp_path)
    ti = tmp_path / "tests" / "integration"
    (ti / "test_ghost.py").write_text(
        "# verifies: use-cases:no-such-uc\ndef test_g():\n    pass\n", encoding="utf-8")
    edges = harvest(str(tmp_path))
    findings = check_endpoints(edges, nodes(str(tmp_path)))
    assert any("no-such-uc" in f.message for f in findings)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/graph/test_verifies_edge.py -v`
Expected: FAIL — `KeyError: 'Test'` / no `verifies` edges.

- [ ] **Step 3: Write minimal implementation**

In `tools/graph/registry.py`, add to `NODE_DOMAINS` after the `UseCase` line:

```python
    "UseCase": "use-cases",
    "Test": "tests",
```

Update the reserved comment to drop `Test`. Append to `EDGES` (after `fulfilled_by`):

```python
    EdgeType("verifies", "verified_by", "Test", "CodeUnit|UseCase|Capability", "derived",
             field="verifies_edges", resolve="id",
             properties=[PropSpec("test_type", enum=["unit", "integration", "e2e"])],
             description="A test proves a code unit works, or an acceptance test proves an intent."),
```

In `tools/graph/reader.py`, add the import (with the other tool imports):

```python
from tools.testmap.reader import load_tests, verifies_edges
```

Add the adapter entry to `_ADAPTERS` (after `UseCase`):

```python
    "Test": (load_tests, "slug"),
```

Add a derived handler and register it (next to `_derived_deps` / `_DERIVED`):

```python
def _derived_verifies(edge: EdgeType, root: str) -> List[Edge]:
    return [Edge(edge.name, src, dst, {"test_type": tt})
            for src, dst, tt in verifies_edges(root)]


_DERIVED = {"dep_edges": _derived_deps, "verifies_edges": _derived_verifies}
```

(Replace the existing single-entry `_DERIVED` assignment with this two-entry one.)

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/graph/test_verifies_edge.py tests/graph -v`
Expected: PASS (new tests + existing graph tests still green).

- [ ] **Step 5: Commit**

```bash
git add tools/graph/registry.py tools/graph/reader.py tests/graph/test_verifies_edge.py
git commit -m "feat(graph): activate Test node + polymorphic verifies edge (derived, test_type prop)"
```

---

### Task 7: Wire the domain, self-register, seed markers, regenerate

Completes the round: domain discoverable, tooling self-registered, a few real `# verifies:` markers seeded so at least one use-case reads VERIFIED, all indexes regenerated, all checks clean-or-advisory.

**Files:**
- Create: `docs/tests/README.md`, `docs/code/tools.testmap.md`, `docs/capabilities/map-the-tests.md`
- Modify: `tools/knowledge/check.py` (`DOMAINS`), `docs/index.md` (cascade row), `Makefile` (targets + health)
- Modify (seed markers): 2-3 real files under `tests/integration/`
- Generated (regenerate, do not hand-edit): `docs/tests/index.md`, `docs/capabilities/index.md`, `docs/code/index.md`, `docs/code/pipeline.md`, `docs/cli/index.md`, `docs/graph/index.md`, `docs/graph/graph.md`

**Interfaces:**
- Consumes: all prior tasks + the graph activation.
- Produces: a fully wired, self-registered tests domain with rendered `docs/tests/index.md`.

- [ ] **Step 1: Cascade registry entry** — in `tools/knowledge/check.py`, add to `DOMAINS` (after `("use-cases", "usecase")`):

```python
    ("use-cases", "usecase"),
    ("tests", "testmap"),
]
```

- [ ] **Step 2: Cascade root row** — in `docs/index.md`, add a table row (must contain `tests/`):

```markdown
| [tests/](tests/README.md) | the test suite as nodes + what it verifies (derived verification axis) | `make testmap-check` |
```

- [ ] **Step 3: Makefile targets + health** — add near the usecase targets:

```makefile
.PHONY: testmap-index
testmap-index: ## Regenerate docs/tests/index.md (test suite nodes + verification rollup)
	@$(PYTHON) -m tools.testmap index

.PHONY: testmap-check
testmap-check: ## Reconcile tests vs code/intent + verification coverage (non-blocking)
	@$(PYTHON) -m tools.testmap check
```

Extend the `health` loop's `for d in ...` list with `testmap` at the end.

- [ ] **Step 4: Concept doc** `docs/tests/README.md`

```markdown
# Tests — how to think about this domain

This bundle is the **test suite as a graph node set**, and what each test **verifies** —
the layer that closes the Requirements Traceability Matrix (intent → capability → code →
**test**). The live view is **[index.md](index.md)** (generated). This page is the mental
model.

## Test nodes are derived, never authored

With ~1,600 test functions, `Test` nodes are **derived** per file by scanning `tests/`
(like the code map). `test_type` (`unit | integration | e2e`, an open set) is derived from
path + filename. There are no Test markdown files to maintain.

## What a test verifies — two sources

- **Derived → code:** a unit test in `tests/<pkg>/` verifies the matching code unit, by the
  tests-mirror-source convention. No authoring.
- **Authored → intent:** an integration/e2e test that validates a use-case's acceptance
  criteria or a capability carries a module-level marker:

  ```python
  # verifies: use-cases:correct-what-the-system-got-wrong
  ```

  The `<domain>:<id>` is prefix-resolved; a marker pointing at a nonexistent node is caught
  by `make graph-check`.

## Verification is derived and orthogonal to implementation

A node's **verification** state (`UNVERIFIED | PARTIALLY_VERIFIED | VERIFIED`) is separate
from its **implementation** coverage (`../use-cases/`, `../capabilities/`). A use-case can
read `FULLY_COVERED` + `UNVERIFIED` — built but not yet proven. Verification rolls up
transitively (a use-case through its fulfilling capabilities' tested code) plus any direct
acceptance marker.

## Reconciling

`make testmap-check` (non-blocking) reports: unknown `test_type`; a test that verifies
nothing the graph can see (no target, no marker); `UNVERIFIED` use-cases (the honest gap);
and index drift. Run `make testmap-index` after adding tests or markers.
```

- [ ] **Step 5: Self-registration nodes**

`docs/code/tools.testmap.md`:

```markdown
---
type: CodeUnit
unit: tools.testmap
role: tooling
key_modules: [reader, verification, render, check]
---
The tests domain: derives Test nodes from the suite and the verifies edge (test→code/intent), and rolls up an orthogonal verification axis over capabilities and use-cases.
```

`docs/capabilities/map-the-tests.md`:

```markdown
---
type: Capability
kind: child
parent: maintain-a-guarded-knowledge-graph
implemented_by: [tools.testmap]
---
Map the test suite as graph nodes and derive what each test verifies, so coverage gains a verification dimension distinct from implementation.
```

- [ ] **Step 6: Seed authored markers** — add a `# verifies:` marker (module-level, just after the docstring) to 2-3 real integration/e2e tests whose intent is clear. Verify the use-case slugs exist in `docs/use-cases/` first. Suggested (confirm the file validates the named intent before adding):

  - `tests/integration/test_e2e_user_edits.py` → `# verifies: use-cases:correct-what-the-system-got-wrong`
  - `tests/integration/test_end_to_end_smoke.py` (if it exercises the full analysis pipeline) → `# verifies: use-cases:surface-the-signal`

  Add only markers whose test genuinely exercises that use-case's acceptance criteria — do not fabricate. At least one must land so a use-case reads `VERIFIED`.

- [ ] **Step 7: Regenerate every affected index**

```bash
make capability-index    # map-the-tests
make code-index          # tools.testmap node
make testmap-index       # docs/tests/index.md
make graph-index         # verifies edges + Test nodes live
make cli-index           # testmap-* targets
```

- [ ] **Step 8: Run the full sweep — expect clean-or-advisory, nothing raising**

```bash
make testmap-check       # advisory: UNVERIFIED use-cases + any unmapped tests (expected)
make graph-check         # clean — every verifies endpoint resolves
make capability-check    # clean — tools.testmap claimed by map-the-tests
make code-check          # clean — tools.testmap node present
make cli-check           # clean
make knowledge-check     # clean — cascade row + DOMAINS entry
make health              # full sweep runs testmap-check
python -m pytest tests/testmap tests/graph -v   # all green
```

Expected: `graph/capability/code/cli/knowledge` checks `clean`; `testmap-check` reports UNVERIFIED use-cases and possibly a few unmapped tests (both expected advisories); at least one use-case shows `VERIFIED` via a seeded marker (confirm with `python -m tools.testmap verification`).

- [ ] **Step 9: Commit**

```bash
git add tools/knowledge/check.py docs/index.md Makefile docs/tests/ \
        docs/code/tools.testmap.md docs/capabilities/map-the-tests.md \
        docs/capabilities/index.md docs/code/index.md docs/code/pipeline.md \
        docs/cli/index.md docs/graph/index.md docs/graph/graph.md \
        tests/integration/
git commit -m "feat(testmap): wire domain + self-register + seed verifies markers"
```

---

## After all tasks

Capture **ADR-0022** (`python -m tools.adr new "Tests domain with an orthogonal verification axis"`, `source:` = the spec, set `supersedes: []`, note in the body that it refines ADR-0021 and ADR-0020; then `make adr-index`). Adding the ADR changes the ADR node count — regenerate `make graph-index` and confirm `graph-check` clean. Run the final whole-branch review on the most capable model, then use **superpowers:finishing-a-development-branch**.

## Knowledge-graph check

Reviewed against `docs/index.md` on 2026-08-06.

| domain | touched? | consulted / reconciled | notes |
| --- | --- | --- | --- |
| tests | yes | the new domain (subject) | — |
| graph | yes | activate `Test` node + derived `verifies` edge (Task 6) | one registry entry + one adapter + one derived handler |
| code | yes (read-only) | `verifies→code` resolves against the code-unit registry; self-register `tools.testmap` node (Task 7) | reuse `tools.code`/`real_code_units` |
| capabilities | yes | read-only for the axis; **one additive** `map-the-tests` self-registration child (Task 7) | no existing capability edited |
| use-cases | yes (read-only) | verification rolls up through `fulfilled_by`; markers may target `use-cases:<id>`; use-case files unchanged | — |
| cli | yes | `testmap-*` + `health` → `cli-index` (Task 7) | — |
| knowledge | yes | cascade row + `DOMAINS` entry (Task 7) | — |
| adr | yes | ADR-0022 (refines 0021, 0020; after tasks) | — |
| glossary / api / prompts / graph-queries | no | — | unaffected |

**Verdict:** reconciled — tests/graph (subject + activation), code/cli/knowledge (convention + wiring) reconciled here; capabilities/use-cases consulted read-only with a single additive self-registration node.
