# Use-Cases Domain Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the use-cases domain — the graph's source layer: one `UseCase` node type (open `form` axis + Cockburn optional block), acceptance criteria as strings, coverage derived from `fulfilled_by`, and a full derived corpus — mirroring the established per-domain tooling pattern.

**Architecture:** A new `tools/usecase/` Python package (`reader → coverage → render → check → __main__`) over markdown nodes in `docs/use-cases/`, plus activation of the graph registry's reserved `UseCase` node type and a new `fulfilled_by` edge. Coverage (`NOT/PARTIALLY/FULLY_COVERED`) is computed, never stored. All guards non-blocking (`return 0`).

**Tech Stack:** Python 3 (stdlib + `yaml` via existing `src.ingestion.front_matter`), pytest, Makefile, GNU Make. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-08-06-use-cases-domain-design.md`.

## Global Constraints

- **Capabilities are read-only for the derivation.** No task un-implements a capability, removes an `implements` edge, or edits the *intent* of an existing capability node. The one capability file added (`map-use-cases`, Task 7) is **self-registration of the new tooling** — additive, mirroring `link-the-domains`→`tools.graph` from the graph round — not part of the use-case derivation.
- **The `fulfilled_by` edge is authored on the use-case side only** (`fulfilled_by:` frontmatter on `UseCase` nodes). Capability files are never opened to record links.
- **All guards are non-blocking:** every `run_all`/`check_*` returns `list[Finding]` and never raises; every CLI `cmd_check` returns `0`.
- **`form` and `category` are open, ordered sets.** `form` lives in `tools/usecase/reader.py` as `FORMS`; `category` reuses `tools.capability.reader.CATEGORIES`. A node whose value is outside the set is *flagged* (advisory), never rejected.
- **Coverage is derived, never stored.** No `status`/`coverage` frontmatter field exists on a node.
- **Mirror the existing per-domain pattern exactly** (`tools/capability/`, `tools/graph/`): same module split, same `Finding` dataclass, same non-blocking CLI shape, same index-sync check.
- **Package/target names:** Python package `tools/usecase/` (no hyphen); make targets `usecase-index` / `usecase-check`; docs folder + cascade slug `docs/use-cases/` (hyphen); graph node domain slug `use-cases`; addressing `use-cases:<slug>`.
- **Naming across tasks (use verbatim):** `UseCase`, `load_use_cases`, `FORMS`, `coverage`, `NOT_COVERED`, `PARTIALLY_COVERED`, `FULLY_COVERED`, `render_index(use_cases, coverage)`, `fulfilled_by` (edge + field).

---

### Task 1: UseCase reader

**Files:**
- Create: `tools/usecase/__init__.py` (empty)
- Create: `tools/usecase/reader.py`
- Test: `tests/usecase/__init__.py` (empty), `tests/usecase/test_reader.py`

**Interfaces:**
- Consumes: `src.ingestion.front_matter.parse_front_matter(text) -> (dict|None, int)`.
- Produces: `FORMS: list[str]`; `UseCase` dataclass (fields below); `load_use_cases(root=".", uc_dir="docs/use-cases") -> list[UseCase]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/usecase/test_reader.py
import os
from tools.usecase.reader import FORMS, UseCase, load_use_cases

CORE = """---
type: UseCase
form: use-case
category: product
actor: analyst
acceptance_criteria:
  - "Given a transcript, when analyzed, then insights are surfaced"
fulfilled_by: [extract-insights-via-lenses]
level: user-goal
---
As an analyst drowning in transcripts, I want the signal surfaced so I stop missing what matters.
"""

def _write(tmp_path, name, text):
    d = tmp_path / "docs" / "use-cases"
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text(text, encoding="utf-8")

def test_loads_core_and_optional_fields(tmp_path):
    _write(tmp_path, "surface-the-signal.md", CORE)
    ucs = load_use_cases(str(tmp_path))
    assert len(ucs) == 1
    u = ucs[0]
    assert u.slug == "surface-the-signal"
    assert u.form == "use-case" and u.category == "product" and u.actor == "analyst"
    assert u.fulfilled_by == ["extract-insights-via-lenses"]
    assert u.acceptance_criteria == ["Given a transcript, when analyzed, then insights are surfaced"]
    assert u.level == "user-goal"
    assert u.statement.startswith("As an analyst")

def test_skips_index_readme_and_non_usecase(tmp_path):
    _write(tmp_path, "index.md", "# Use-Cases\n")
    _write(tmp_path, "README.md", "# concept\n")
    _write(tmp_path, "other.md", "---\ntype: Capability\n---\nnope\n")
    assert load_use_cases(str(tmp_path)) == []

def test_missing_optional_fields_default_empty(tmp_path):
    _write(tmp_path, "bare.md", "---\ntype: UseCase\nform: user-story\ncategory: operations\n---\nA bare intent.\n")
    u = load_use_cases(str(tmp_path))[0]
    assert u.acceptance_criteria == [] and u.fulfilled_by == [] and u.level == ""
    assert "user-story" in FORMS
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/usecase/test_reader.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tools.usecase'`.

- [ ] **Step 3: Write minimal implementation**

```python
# tools/usecase/reader.py
from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field
from typing import List

from src.ingestion.front_matter import parse_front_matter

# `form` is an open, ordered set (like the capability `category` axis). Add a value here.
FORMS = ["user-story", "feature", "requirement", "use-case"]


@dataclass
class UseCase:
    slug: str
    form: str
    category: str
    actor: str
    statement: str
    path: str
    acceptance_criteria: List[str] = field(default_factory=list)
    fulfilled_by: List[str] = field(default_factory=list)
    level: str = ""              # Cockburn: user-goal | summary | subfunction
    preconditions: str = ""
    main_scenario: str = ""
    extensions: str = ""
    end_conditions: str = ""


def load_use_cases(root: str = ".", uc_dir: str = "docs/use-cases") -> List[UseCase]:
    ucs: List[UseCase] = []
    for path in sorted(glob.glob(os.path.join(root, uc_dir, "*.md"))):
        base = os.path.basename(path)
        if base in ("index.md", "README.md"):
            continue
        try:
            text = open(path, encoding="utf-8", errors="ignore").read()
        except OSError:
            continue
        fm, offset = parse_front_matter(text)
        if not fm or fm.get("type") != "UseCase":
            continue
        ucs.append(UseCase(
            slug=os.path.splitext(base)[0],
            form=str(fm.get("form", "")),
            category=str(fm.get("category", "")),
            actor=str(fm.get("actor", "")),
            statement=text[offset:].strip(),
            path=path,
            acceptance_criteria=list(fm.get("acceptance_criteria") or []),
            fulfilled_by=list(fm.get("fulfilled_by") or []),
            level=str(fm.get("level", "")),
            preconditions=str(fm.get("preconditions", "")),
            main_scenario=str(fm.get("main_scenario", "")),
            extensions=str(fm.get("extensions", "")),
            end_conditions=str(fm.get("end_conditions", "")),
        ))
    return ucs
```

Also create empty `tools/usecase/__init__.py` and `tests/usecase/__init__.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/usecase/test_reader.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add tools/usecase/__init__.py tools/usecase/reader.py tests/usecase/
git commit -m "feat(usecase): UseCase reader — open form axis + Cockburn optional block"
```

---

### Task 2: Coverage derivation

**Files:**
- Create: `tools/usecase/coverage.py`
- Test: `tests/usecase/test_coverage.py`

**Interfaces:**
- Consumes: `tools.usecase.reader.UseCase`; `tools.capability.reader.Capability` (fields `.slug`, `.implemented_by`).
- Produces: string constants `NOT_COVERED`, `PARTIALLY_COVERED`, `FULLY_COVERED`; `coverage(use_cases, capabilities, valid_units: set) -> dict[str, str]` (slug → state).

- [ ] **Step 1: Write the failing test**

```python
# tests/usecase/test_coverage.py
from tools.capability.reader import Capability
from tools.usecase.reader import UseCase
from tools.usecase.coverage import (
    NOT_COVERED, PARTIALLY_COVERED, FULLY_COVERED, coverage,
)

def _uc(slug, fulfilled_by):
    return UseCase(slug=slug, form="use-case", category="product", actor="a",
                   statement="s", path="p", fulfilled_by=fulfilled_by)

def _cap(slug, implemented_by):
    return Capability(slug=slug, kind="primary", tier="core", parent="",
                      implemented_by=implemented_by, statement="s", path="p",
                      category="product")

def test_three_states():
    caps = [_cap("built", ["api"]), _cap("unbuilt", [])]
    valid = {"api"}
    ucs = [
        _uc("bare", []),                       # nothing fulfills -> NOT
        _uc("aspirational", ["unbuilt"]),      # fulfilled by unimplemented -> PARTIAL
        _uc("done", ["built"]),                # fulfilled by implemented -> FULL
        _uc("mixed", ["built", "unbuilt"]),    # one gap -> PARTIAL
    ]
    cov = coverage(ucs, caps, valid)
    assert cov["bare"] == NOT_COVERED
    assert cov["aspirational"] == PARTIALLY_COVERED
    assert cov["done"] == FULLY_COVERED
    assert cov["mixed"] == PARTIALLY_COVERED

def test_unresolvable_capability_slug_ignored():
    # a fulfilled_by pointing at a nonexistent capability contributes nothing
    cov = coverage([_uc("ghost", ["no-such-cap"])], [], set())
    assert cov["ghost"] == NOT_COVERED
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/usecase/test_coverage.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tools.usecase.coverage'`.

- [ ] **Step 3: Write minimal implementation**

```python
# tools/usecase/coverage.py
from __future__ import annotations

from typing import Dict, List, Set

from tools.capability.reader import Capability
from tools.usecase.reader import UseCase

NOT_COVERED = "NOT_COVERED"
PARTIALLY_COVERED = "PARTIALLY_COVERED"
FULLY_COVERED = "FULLY_COVERED"


def _implemented(cap: Capability, valid_units: Set[str]) -> bool:
    return any(u in valid_units for u in cap.implemented_by)


def coverage(use_cases: List[UseCase], capabilities: List[Capability],
             valid_units: Set[str]) -> Dict[str, str]:
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/usecase/test_coverage.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add tools/usecase/coverage.py tests/usecase/test_coverage.py
git commit -m "feat(usecase): derived coverage — NOT/PARTIALLY/FULLY transitive through capabilities"
```

---

### Task 3: Renderer

**Files:**
- Create: `tools/usecase/render.py`
- Test: `tests/usecase/test_render.py`

**Interfaces:**
- Consumes: `tools.capability.reader.CATEGORIES`; `tools.usecase.reader.FORMS`, `UseCase`.
- Produces: `render_index(use_cases: list[UseCase], coverage: dict[str, str]) -> str` (pure; deterministic; trailing single newline).

- [ ] **Step 1: Write the failing test**

```python
# tests/usecase/test_render.py
from tools.usecase.reader import UseCase
from tools.usecase.render import render_index
from tools.usecase.coverage import FULLY_COVERED, NOT_COVERED

def _uc(slug, form="use-case", category="product", actor="analyst",
        fulfilled_by=None, ac=None):
    return UseCase(slug=slug, form=form, category=category, actor=actor,
                   statement=f"statement for {slug}", path="p",
                   acceptance_criteria=ac or [], fulfilled_by=fulfilled_by or [])

def test_groups_by_category_then_form_with_coverage():
    ucs = [_uc("z-signal", fulfilled_by=["c"], ac=["x"]),
           _uc("a-import", form="requirement", fulfilled_by=[])]
    cov = {"z-signal": FULLY_COVERED, "a-import": NOT_COVERED}
    out = render_index(ucs, cov)
    assert out.startswith("# Use-Cases")
    assert "## product" in out
    assert "### use-case" in out and "### requirement" in out
    assert "#### z-signal — FULLY_COVERED" in out
    assert "#### a-import — NOT_COVERED" in out
    assert "- **fulfilled_by:** c" in out
    assert "- **acceptance_criteria:** 1" in out          # z-signal has 1
    assert "- **acceptance_criteria:** — none yet" in out # a-import has none
    assert out.endswith("\n") and not out.endswith("\n\n")

def test_empty_category_and_form_omitted():
    out = render_index([_uc("only", category="product")], {"only": NOT_COVERED})
    assert "## operations" not in out and "## support" not in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/usecase/test_render.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tools.usecase.render'`.

- [ ] **Step 3: Write minimal implementation**

```python
# tools/usecase/render.py
from __future__ import annotations

from typing import Dict, List

from tools.capability.reader import CATEGORIES
from tools.usecase.reader import FORMS, UseCase


def render_index(use_cases: List[UseCase], coverage: Dict[str, str]) -> str:
    lines = [
        "# Use-Cases", "",
        "The user-centered intents this system serves — the \"why\" above the "
        "capabilities (`../capabilities/`). Coverage is derived from `fulfilled_by`, "
        "never stored.", "",
    ]
    for category in CATEGORIES:
        cat = [u for u in use_cases if u.category == category]
        if not cat:
            continue  # reserved/empty category — omit
        lines.append(f"## {category}")
        lines.append("")
        for form in FORMS:
            form_ucs = sorted((u for u in cat if u.form == form), key=lambda u: u.slug)
            if not form_ucs:
                continue
            lines.append(f"### {form}")
            lines.append("")
            for u in form_ucs:
                state = coverage.get(u.slug, "NOT_COVERED")
                lines.append(f"#### {u.slug} — {state}")
                lines.append(u.statement)
                lines.append("")
                lines.append(f"- **actor:** {u.actor or '—'}")
                lines.append(f"- **fulfilled_by:** {', '.join(u.fulfilled_by) or '—'}")
                ac = u.acceptance_criteria
                lines.append(f"- **acceptance_criteria:** {len(ac) if ac else '— none yet'}")
                lines.append("")
    return "\n".join(lines).rstrip() + "\n"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/usecase/test_render.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add tools/usecase/render.py tests/usecase/test_render.py
git commit -m "feat(usecase): render_index — grouped by category/form with derived coverage"
```

---

### Task 4: Guard (check)

**Files:**
- Create: `tools/usecase/check.py`
- Test: `tests/usecase/test_check.py`

**Interfaces:**
- Consumes: `tools.capability.reader.CATEGORIES`, `load_capabilities`, `real_code_units`; `tools.usecase.reader.FORMS`, `load_use_cases`; `tools.usecase.coverage.coverage`, `NOT_COVERED`; `tools.usecase.render.render_index`.
- Produces: `Finding` dataclass; `check_forms`, `check_categories`, `check_acceptance_criteria`, `check_uncovered`, `check_index_sync`, `run_all(root=".") -> list[Finding]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/usecase/test_check.py
from tools.usecase.reader import UseCase
from tools.usecase.coverage import NOT_COVERED, PARTIALLY_COVERED
from tools.usecase.check import (
    check_forms, check_categories, check_acceptance_criteria, check_uncovered, run_all,
)

def _uc(slug, form="use-case", category="product", ac=None, fulfilled_by=None):
    return UseCase(slug=slug, form=form, category=category, actor="a",
                   statement="s", path="p", acceptance_criteria=ac or [],
                   fulfilled_by=fulfilled_by or [])

def test_check_forms_flags_unknown():
    assert check_forms([_uc("x", form="job-story")])          # not in FORMS -> flagged
    assert check_forms([_uc("x", form="use-case")]) == []

def test_check_categories_flags_unknown():
    assert check_categories([_uc("x", category="marketing")])
    assert check_categories([_uc("x", category="operations")]) == []

def test_empty_acceptance_criteria_is_advisory():
    assert check_acceptance_criteria([_uc("x", ac=[])])
    assert check_acceptance_criteria([_uc("x", ac=["c"])]) == []

def test_uncovered_flagged():
    cov = {"bare": NOT_COVERED, "part": PARTIALLY_COVERED}
    flagged = check_uncovered([_uc("bare"), _uc("part")], cov)
    assert len(flagged) == 1 and "bare" in flagged[0].message

def test_run_all_never_raises_on_real_repo():
    findings = run_all(".")            # advisory findings allowed; must not raise
    assert isinstance(findings, list)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/usecase/test_check.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tools.usecase.check'`.

- [ ] **Step 3: Write minimal implementation**

```python
# tools/usecase/check.py
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
    return [Finding(f"use-case: {u.slug} has unknown form '{u.form}'")
            for u in ucs if u.form not in FORMS]


def check_categories(ucs: List[UseCase]) -> List[Finding]:
    return [Finding(f"use-case: {u.slug} has unknown category '{u.category}'")
            for u in ucs if u.category not in CATEGORIES]


def check_acceptance_criteria(ucs: List[UseCase]) -> List[Finding]:
    return [Finding(f"use-case: {u.slug} has no acceptance_criteria yet")
            for u in ucs if not u.acceptance_criteria]


def check_uncovered(ucs: List[UseCase], cov: Dict[str, str]) -> List[Finding]:
    return [Finding(f"use-case: {u.slug} is NOT_COVERED — no capability fulfills it")
            for u in ucs if cov.get(u.slug) == NOT_COVERED]


def check_index_sync(index_path: str, ucs: List[UseCase], cov: Dict[str, str]) -> List[Finding]:
    want = render_index(ucs, cov)
    have = open(index_path, encoding="utf-8", errors="ignore").read() if os.path.exists(index_path) else ""
    if want != have:
        return [Finding("use-case: docs/use-cases/index.md out of sync — run make usecase-index")]
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
    findings += check_index_sync(os.path.join(root, "docs/use-cases/index.md"), ucs, cov)
    return findings
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/usecase/test_check.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add tools/usecase/check.py tests/usecase/test_check.py
git commit -m "feat(usecase): non-blocking guard — form/category/criteria/uncovered/index-sync"
```

---

### Task 5: CLI

**Files:**
- Create: `tools/usecase/__main__.py`
- Test: `tests/usecase/test_cli.py`

**Interfaces:**
- Consumes: all of `tools.usecase.*`; `tools.capability.reader.load_capabilities`, `real_code_units`.
- Produces: `python -m tools.usecase {index | check | coverage}`; `main(argv=None) -> int`; module const `INDEX = "docs/use-cases/index.md"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/usecase/test_cli.py
from tools.usecase.__main__ import main

def test_check_returns_zero_non_blocking(capsys):
    assert main(["check"]) == 0
    assert "usecase-check:" in capsys.readouterr().out

def test_coverage_command_runs(capsys):
    assert main(["coverage"]) == 0   # prints one line per use-case; may be empty pre-corpus
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/usecase/test_cli.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tools.usecase.__main__'`.

- [ ] **Step 3: Write minimal implementation**

```python
# tools/usecase/__main__.py
from __future__ import annotations

import argparse
import os
import sys

from tools.capability.reader import load_capabilities, real_code_units
from tools.usecase.check import run_all
from tools.usecase.coverage import coverage
from tools.usecase.reader import load_use_cases
from tools.usecase.render import render_index

INDEX = "docs/use-cases/index.md"


def _coverage(root: str = "."):
    return coverage(load_use_cases(root), load_capabilities(root), real_code_units(root))


def cmd_index(args) -> int:
    os.makedirs(os.path.dirname(INDEX), exist_ok=True)
    with open(INDEX, "w", encoding="utf-8") as fh:
        fh.write(render_index(load_use_cases(), _coverage()))
    print(f"wrote {INDEX}")
    return 0


def cmd_check(args) -> int:
    findings = run_all()
    if findings:
        print(f"usecase-check: {len(findings)} warning(s):")
        for f in findings:
            print(f"  - {f.message}")
    else:
        print("usecase-check: clean")
    return 0  # NON-BLOCKING


def cmd_coverage(args) -> int:
    cov = _coverage()
    for slug in sorted(cov):
        print(f"{cov[slug]:18} {slug}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="tools.usecase")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("index")
    sub.add_parser("check")
    sub.add_parser("coverage")
    args = parser.parse_args(argv)
    return {"index": cmd_index, "check": cmd_check, "coverage": cmd_coverage}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/usecase/test_cli.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add tools/usecase/__main__.py tests/usecase/test_cli.py
git commit -m "feat(usecase): CLI — index | check | coverage (non-blocking)"
```

---

### Task 6: Activate the `UseCase` node + `fulfilled_by` edge in the graph

**Files:**
- Modify: `tools/graph/registry.py` (add node domain + edge type)
- Modify: `tools/graph/reader.py` (add adapter + import)
- Test: `tests/graph/test_usecase_edge.py`

**Interfaces:**
- Consumes: `tools.usecase.reader.load_use_cases` (adapter), `UseCase` (`.slug`, `.fulfilled_by`).
- Produces: registry `NODE_DOMAINS["UseCase"] = "use-cases"`; new `EdgeType("fulfilled_by", "fulfills", "UseCase", "Capability", "authored", field="fulfilled_by", resolve="id")`; harvest emits `Edge("fulfilled_by", "use-cases:<slug>", "capabilities:<cap>")`.

- [ ] **Step 1: Write the failing test**

```python
# tests/graph/test_usecase_edge.py
from tools.graph.reader import harvest, nodes
from tools.graph.check import check_endpoints

def _seed(tmp_path):
    caps = tmp_path / "docs" / "capabilities"; caps.mkdir(parents=True)
    (caps / "surface.md").write_text(
        "---\ntype: Capability\nkind: primary\ntier: core\ncategory: product\n"
        "implemented_by: []\n---\nSurface the signal.\n", encoding="utf-8")
    ucs = tmp_path / "docs" / "use-cases"; ucs.mkdir(parents=True)
    (ucs / "see-the-signal.md").write_text(
        "---\ntype: UseCase\nform: use-case\ncategory: product\nactor: analyst\n"
        "fulfilled_by: [surface]\n---\nAs an analyst, I want the signal.\n", encoding="utf-8")

def test_fulfilled_by_harvested(tmp_path):
    _seed(tmp_path)
    edges = harvest(str(tmp_path))
    fb = [e for e in edges if e.type == "fulfilled_by"]
    assert any(e.src == "use-cases:see-the-signal" and e.dst == "capabilities:surface" for e in fb)
    assert "see-the-signal" in nodes(str(tmp_path))["UseCase"]

def test_dangling_fulfilled_by_flagged(tmp_path):
    _seed(tmp_path)
    (tmp_path / "docs" / "use-cases" / "ghost.md").write_text(
        "---\ntype: UseCase\nform: user-story\ncategory: product\nactor: a\n"
        "fulfilled_by: [no-such-cap]\n---\nGhost intent.\n", encoding="utf-8")
    edges = harvest(str(tmp_path))
    findings = check_endpoints(edges, nodes(str(tmp_path)))
    assert any("no-such-cap" in f.message for f in findings)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/graph/test_usecase_edge.py -v`
Expected: FAIL — `KeyError: 'UseCase'` (no adapter / node domain yet).

- [ ] **Step 3: Write minimal implementation**

In `tools/graph/registry.py`, add the node domain (replace the reserved comment line for UseCase):

```python
NODE_DOMAINS = {
    "CodeUnit": "code",
    "Capability": "capabilities",
    "ADR": "adr",
    "UseCase": "use-cases",
    # reserved: GlossaryTerm→glossary, Prompt→prompts, GraphQuery→graph-queries,
    # Spec→spec, Test→test
}
```

And append to `EDGES`:

```python
    EdgeType("fulfilled_by", "fulfills", "UseCase", "Capability", "authored",
             field="fulfilled_by", resolve="id",
             description="A use-case's intent is reached toward by a capability's implementation."),
```

In `tools/graph/reader.py`, add the import and adapter entry:

```python
from tools.usecase.reader import load_use_cases   # with the other tool imports
```

```python
_ADAPTERS = {
    "Capability": (load_capabilities, "slug"),
    "CodeUnit": (load_units, "unit"),
    "ADR": (lambda root: load_bundle(os.path.join(root, "docs/adr")), "id"),
    "UseCase": (load_use_cases, "slug"),
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/graph/test_usecase_edge.py tests/graph/ -v`
Expected: PASS (new tests + existing graph tests still green).

- [ ] **Step 5: Commit**

```bash
git add tools/graph/registry.py tools/graph/reader.py tests/graph/test_usecase_edge.py
git commit -m "feat(graph): activate UseCase node + fulfilled_by edge (authored use-case side)"
```

---

### Task 7: Wire the domain, self-register, and land Phase-1 exemplars

This task completes Phase 1: the domain is discoverable (cascade, registry, Makefile, README), the new tooling self-registers as a capability + code node, and 2-3 exemplar use-cases prove the pipeline end-to-end across all three coverage states. It ends with every generated index regenerated and every check clean-or-advisory.

**Files:**
- Create: `docs/use-cases/README.md` (concept doc)
- Create: `docs/use-cases/surface-the-signal.md`, `docs/use-cases/onboard-my-transcripts.md`, `docs/use-cases/revisit-a-past-extraction.md` (exemplars)
- Create: `docs/code/tools.usecase.md` (self-register code node)
- Create: `docs/capabilities/map-use-cases.md` (self-register tooling capability — additive)
- Modify: `tools/knowledge/check.py` (`DOMAINS` += use-cases)
- Modify: `docs/index.md` (cascade row)
- Modify: `Makefile` (`usecase-index`, `usecase-check`, `health` loop)
- Generated (regenerate, do not hand-edit): `docs/use-cases/index.md`, `docs/capabilities/index.md`, `docs/code/index.md`, `docs/cli/index.md`, `docs/graph/index.md`, `docs/graph/graph.md`

**Interfaces:**
- Consumes: all prior tasks' modules + the graph activation.
- Produces: a fully wired, self-registered domain with a rendered `docs/use-cases/index.md`.

- [ ] **Step 1: Add the cascade registry entry**

In `tools/knowledge/check.py`, add to `DOMAINS` (keep the list readable):

```python
    ("prompts", "prompt"),
    ("use-cases", "usecase"),
]
```

- [ ] **Step 2: Add the cascade root row**

In `docs/index.md`, add a row to the domain table (must contain the substring `use-cases/`):

```markdown
| [use-cases/](use-cases/README.md) | user-centered intents (requirements/stories/features) with derived coverage over capabilities | `make usecase-check` |
```

- [ ] **Step 3: Add Makefile targets + health loop**

Add near the capability targets:

```makefile
.PHONY: usecase-index
usecase-index: ## Regenerate docs/use-cases/index.md (the use-case corpus + derived coverage)
	@$(PYTHON) -m tools.usecase index

.PHONY: usecase-check
usecase-check: ## Reconcile use-cases vs forms/categories/criteria + coverage (non-blocking)
	@$(PYTHON) -m tools.usecase check
```

And extend the `health` loop to include `usecase`:

```makefile
	@for d in adr cli api glossary prompts graphq code capability knowledge graph usecase; do $(PYTHON) -m tools.$$d check || true; done
```

- [ ] **Step 4: Write the concept doc** `docs/use-cases/README.md`

```markdown
# Use-Cases — how to think about this domain

This bundle records **direct user-centered input** — the intents that (had they been
written first) would have *led to* this system: requirements, user stories, features, and
formal use cases. It is the graph's **source layer**, the "why" above the capabilities
(`../capabilities/`). The live corpus is **[index.md](index.md)** (generated — grouped by
category → form, with derived coverage). This page is the mental model; read it before
authoring a node.

## One node, fidelity in `form`

A use-case is one node type at varying fidelity. `form` is an **open set**:
`user-story | feature | requirement | use-case`. A lightweight `user-story` carries only
the core; a full Cockburn `use-case` adds the optional block (`level`, `preconditions`,
`main_scenario`, `extensions`, `end_conditions`). Add a `form` value in
`tools/usecase/reader.py`.

## Coverage is derived, never stored

A use-case has **no status field**. Its coverage — `NOT_COVERED / PARTIALLY_COVERED /
FULLY_COVERED` — is computed from its `fulfilled_by` capabilities and how far *their* code
reaches. An uncovered or partially-covered intent is **legitimate and expected**: it is
the domain surfacing where intent outruns implementation.

## Capabilities are read-only here

Links are authored **on the use-case side** (`fulfilled_by:`), so recording that a
capability serves an intent never edits a capability file. `fulfills` (capability →
use-case) is the derived inverse, read backward in the graph.

## Frontmatter

```yaml
---
type: UseCase
form: user-story | feature | requirement | use-case
category: product | operations | support     # reuses the capability axis (open set)
actor: <who wants it — person, operator, or external system>
acceptance_criteria:                          # list of strings; may be omitted / empty
  - "Given …, when …, then …"                 # Given/When/Then or a rule sentence
fulfilled_by: [<capability slugs from ../capabilities/>]   # may be [] — legitimate
level: user-goal | summary | subfunction      # optional (Cockburn use-case form)
# optional: preconditions, main_scenario, extensions, end_conditions
---
Narrative: "As a <actor>, I want <action> so that <benefit>." Reach past the code to the
human problem — never restate a capability.
```

## Reconciling

`make usecase-check` (non-blocking) reports: unknown `form`/`category`; empty
`acceptance_criteria` (advisory — not yet testable); `NOT_COVERED` intents (advisory —
nothing fulfills them); and index drift. Cross-domain endpoint integrity (a `fulfilled_by`
pointing at a nonexistent capability) is covered by `make graph-check`. Run
`make usecase-index` after adding or editing a node.
```

- [ ] **Step 5: Write the self-registration nodes**

`docs/code/tools.usecase.md`:

```markdown
---
type: CodeUnit
unit: tools.usecase
role: tooling
key_modules: [reader, coverage, render, check]
---
The use-cases domain: user-centered intents (requirements/stories/features/use-cases) with coverage derived over the capability tree, rendered and guarded.
```

`docs/capabilities/map-use-cases.md` (additive self-registration — a child of the knowledge-graph capability, exactly like `link-the-domains`):

```markdown
---
type: Capability
kind: child
parent: maintain-a-guarded-knowledge-graph
implemented_by: [tools.usecase]
---
Record the user-centered intents the system serves and derive how far current capabilities cover them.
```

- [ ] **Step 6: Write the three exemplar use-cases** (one per coverage state)

`docs/use-cases/surface-the-signal.md` (→ FULLY_COVERED via an implemented capability):

```markdown
---
type: UseCase
form: use-case
category: product
actor: analyst
acceptance_criteria:
  - "Given a corpus of transcripts, when the lens engine runs, then extracted insights are returned grouped by dimension"
fulfilled_by: [extract-insights-via-lenses]
level: user-goal
---
As an analyst drowning in raw interview transcripts, I want the meaningful signal surfaced automatically so I stop missing what matters across hundreds of conversations.
```

`docs/use-cases/onboard-my-transcripts.md` (→ PARTIALLY_COVERED via an unimplemented capability):

```markdown
---
type: UseCase
form: user-story
category: product
actor: researcher
acceptance_criteria:
  - "Given a folder of raw transcript files, when I import them, then each becomes an analyzable source in the system"
fulfilled_by: [import-transcripts]
---
As a researcher with a backlog of past interviews, I want to bring my existing transcripts into the system so my prior work is analyzable, not stranded outside it.
```

`docs/use-cases/revisit-a-past-extraction.md` (→ NOT_COVERED — the trajectory-test overshoot, nothing fulfills it yet):

```markdown
---
type: UseCase
form: requirement
category: product
actor: analyst
acceptance_criteria:
  - "Given an earlier extraction I now know was wrong, when I revisit and correct it, then downstream insights reflect the correction with the change recorded"
fulfilled_by: []
---
As an analyst who has learned more since an early pass, I want to revisit and correct a past extraction so my conclusions improve as my understanding does — without redoing everything by hand.
```

> **Note on the exemplars:** `extract-insights-via-lenses` must be a real, implemented capability slug and `import-transcripts` a real but unimplemented one (`implemented_by: []`). Confirm both against `docs/capabilities/` before writing; if either slug differs, substitute the correct one so the three coverage states actually resolve as intended. These three exemplars are retained into the Phase-2 corpus (Task 8), not thrown away.

- [ ] **Step 7: Regenerate every affected index**

```bash
make capability-index   # picks up map-use-cases
make code-index         # picks up docs/code/tools.usecase.md node
make usecase-index      # writes docs/use-cases/index.md
make graph-index        # fulfilled_by edges now live + new capability node/edges
make cli-index          # usecase-* targets catalogued
```

- [ ] **Step 8: Run the full sweep — expect clean-or-advisory, nothing raising**

```bash
make usecase-check      # advisory: revisit-a-past-extraction NOT_COVERED (expected)
make graph-check        # clean — no dangling fulfilled_by
make capability-check   # clean — tools.usecase now claimed by map-use-cases
make code-check         # clean — tools.usecase node present
make cli-check          # clean — targets catalogued
make knowledge-check    # clean — cascade row + DOMAINS entry present
make health             # full sweep runs usecase-check without error
python -m pytest tests/usecase tests/graph -v   # all green
```

Expected: `graph-check`, `capability-check`, `code-check`, `cli-check`, `knowledge-check` all `clean`; `usecase-check` reports exactly the advisory `NOT_COVERED` (and any `— none yet` if an exemplar lacks criteria — the three above all have criteria) and nothing else; pytest green.

- [ ] **Step 9: Commit**

```bash
git add tools/knowledge/check.py docs/index.md Makefile docs/use-cases/ \
        docs/code/tools.usecase.md docs/capabilities/ docs/code/index.md \
        docs/cli/index.md docs/graph/index.md docs/graph/graph.md
git commit -m "feat(usecase): wire domain + self-register tooling + Phase-1 exemplars"
```

---

### Task 8: Phase 2 — derive the full use-case corpus

This is a **content** task, not a code-TDD task, but it is a reviewer-gated deliverable. Its output is the corpus of use-case nodes reconstructed from the capability tree, with `fulfilled_by` links drawn and coverage reviewed. Under subagent-driven execution, dispatch this as a derivation subagent; the controller and owner review the corpus before merge.

**Files:**
- Create: one `docs/use-cases/<slug>.md` per derived use-case (retain the Task-7 exemplars)
- Regenerate: `docs/use-cases/index.md`, `docs/graph/index.md`, `docs/graph/graph.md`

**Method (from the spec — enforced in review):**

- [ ] **Step 1: Enumerate the capability tree.** Read `docs/capabilities/index.md` and the capability nodes. Group them into problem-clusters (product analysis, ingestion/onboarding, resolution/correction, retrieval/synthesis, gallery/workbench surfaces, operations/knowledge-graph tooling, etc.).

- [ ] **Step 2: For each cluster, reconstruct the originating intent** — the human problem that, written first, would have led to those capabilities. Write a `UseCase` node backward from the capability to the actor and benefit. Apply:
  - **Anti-restatement rule:** the `statement` must name an actor and a benefit meaningful to someone who has never seen the code. "The system extracts fragments" → restatement (reject). "As an analyst drowning in transcripts, I want the signal surfaced…" → use-case.
  - **Trajectory test:** follow each problem honestly *past* the current build. A correct corpus **overshoots** — it contains `NOT_COVERED` / `PARTIALLY_COVERED` intents the system points toward but hasn't built (e.g. onboarding/import, revisit-and-correct). **Zero uncovered intents is a red flag** that we restated rather than derived.
  - Cover **product, operations, and support** actors — operations/support use-cases (operator, maintainer) are first-class.
  - Choose `form` by fidelity; set `acceptance_criteria` where the "definition of done" is genuinely known (leave empty otherwise — empty is an honest signal, not a gap to fabricate).

- [ ] **Step 3: Draw `fulfilled_by`** from each use-case to the capability slugs that reach toward it (many-to-many; a use-case may list several, or none). Author only on the use-case side — never edit a capability file.

- [ ] **Step 4: Regenerate + guard.**

```bash
make usecase-index
make graph-index
make usecase-check   # advisory NOT_COVERED/none-yet findings are EXPECTED output, not errors
make graph-check     # must be clean — every fulfilled_by resolves
make health
python -m pytest tests/usecase tests/graph -v
```

- [ ] **Step 5: Owner review before merge.** Present the corpus and its coverage distribution. Confirm: no restatements; a real spread of coverage states (overshoot present); operations/support represented; every `fulfilled_by` resolves (`graph-check` clean). Revise per feedback.

- [ ] **Step 6: Commit**

```bash
git add docs/use-cases/ docs/graph/index.md docs/graph/graph.md
git commit -m "feat(usecase): derive the full use-case corpus (reconstructed originating intents)"
```

---

## After all tasks

Capture **ADR-0021** (`python -m tools.adr new "Use-cases domain as the graph source layer"`, `source:` = the spec, then `make adr-index`), run the final whole-branch review on the most capable model, then use **superpowers:finishing-a-development-branch**.

## Knowledge-graph check

Reviewed against `docs/index.md` on 2026-08-06.

| domain | touched? | consulted / reconciled | notes |
| --- | --- | --- | --- |
| use-cases | yes | the new domain (subject) | — |
| graph | yes | activate `UseCase` node + `fulfilled_by` edge (Task 6) | one registry entry + one adapter |
| capabilities | yes | read-only for derivation; **one additive** `map-use-cases` self-registration child (Task 7) | honors the read-only constraint — no existing capability edited |
| code | yes | self-register `tools.usecase` node (Task 7); deps derived | — |
| cli | yes | `usecase-*` + `health` targets → `cli-index` (Task 7) | — |
| knowledge | yes | cascade row + `DOMAINS` entry (Task 7) | — |
| adr | yes | ADR-0021 (after tasks) | — |
| glossary / api / prompts / graph-queries | no | — | unaffected |

**Verdict:** reconciled — use-cases/graph (subject + activation) and code/cli/knowledge (self-registration + wiring) reconciled in the plan; capabilities consulted read-only with a single additive self-registration node.
