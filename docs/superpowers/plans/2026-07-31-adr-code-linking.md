# ADR↔Code Linking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Link ADRs to the code they govern, bidirectionally and cross-checked: ADRs declare `governs:` paths, code carries `governed-by:` docstring markers, a generated `docs/adr/by-code.md` reverse index, and four non-blocking guard checks.

**Architecture:** Extends the existing `tools/adr/` package (merged in PR #19). Adds one field to the model, one new generic marker-scanner module, one generated artifact, four checks, and a backfill. The marker machinery is kept node-agnostic so later API/CLI/code-doc sub-projects reuse it.

**Tech Stack:** Python 3 (stdlib + PyYAML via existing `parse_front_matter`), pytest, Make, git.

## Global Constraints

- **Non-blocking, always.** Every check returns `list[Finding]`; none raises on a detected problem. `make adr-check` / the CLI exit 0 regardless.
- **`governs` = repo-relative paths**; directories end with `/`, files do not.
- **Code markers** are a `governed-by: ADR-NNNN[, ADR-MMMM]` line in a module docstring (file target) or a package `__init__.py` docstring / directory `README.md` (directory target). Parsed by line-scan, no AST.
- **`scan_markers` stays node-agnostic** — it finds `governed-by:` markers and returns raw ref tokens; it does not know what an ADR is (so C/D/E can reuse it).
- **`by-code.md` is generated** (added to `RESERVED`), never hand-edited.
- **DRY:** reuse the existing `Adr`, `load_bundle`, `git_committer_ts`, `Finding`. Tooling in `tools/adr/`; tests in `tests/adr/`.
- Run tests with the project interpreter: `~/.pyenv/shims/python -m pytest <path> -v` (bare `python` is not on the non-interactive PATH).

---

### Task 1: Add `governs` to the ADR model

**Files:**
- Modify: `tools/adr/model.py`
- Test: `tests/adr/test_model.py`

**Interfaces:**
- Produces: `Adr.governs: list[str]` (default `[]`), parsed by `parse_adr`.

- [ ] **Step 1: Write the failing test** (append to `tests/adr/test_model.py`)

```python
def test_parse_adr_reads_governs_and_defaults_empty():
    with_governs = (
        "---\ntype: ADR\nid: 3\ntitle: X\nstatus: accepted\ndate: 2026-07-04\n"
        "governs:\n  - src/projections/\n  - src/x.py\n---\nbody\n"
    )
    from tools.adr.model import parse_adr
    adr = parse_adr(with_governs)
    assert adr.governs == ["src/projections/", "src/x.py"]

    without = "---\ntype: ADR\nid: 4\ntitle: Y\nstatus: accepted\ndate: 2026-07-04\n---\nbody\n"
    assert parse_adr(without).governs == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.pyenv/shims/python -m pytest tests/adr/test_model.py::test_parse_adr_reads_governs_and_defaults_empty -v`
Expected: FAIL — `AttributeError: 'Adr' object has no attribute 'governs'`

- [ ] **Step 3: Implement**

In `tools/adr/model.py`, add the field to the `Adr` dataclass after `tags`:

```python
    tags: List[str] = field(default_factory=list)
    governs: List[str] = field(default_factory=list)
    source: Optional[str] = None
```

And in `parse_adr`, add the parse line after `tags=...`:

```python
        tags=list(fm.get("tags") or []),
        governs=[str(p) for p in (fm.get("governs") or [])],
        source=fm.get("source"),
```

- [ ] **Step 4: Run test to verify it passes**

Run: `~/.pyenv/shims/python -m pytest tests/adr/test_model.py -v`
Expected: PASS (all model tests)

- [ ] **Step 5: Commit**

```bash
git add tools/adr/model.py tests/adr/test_model.py
git commit -m "feat(adr): governs field on the ADR model"
```

---

### Task 2: Generic code-marker scanner (`code_links.py`)

**Files:**
- Create: `tools/adr/code_links.py`
- Test: `tests/adr/test_code_links.py`

**Interfaces:**
- Produces:
  - `extract_refs(text: str) -> list[str]` — raw tokens after each `governed-by:`
  - `adr_ids_from_refs(refs: list[str]) -> list[int]`
  - `scan_markers(root: str, subdirs=("src",)) -> dict[str, list[str]]` — path → raw ref tokens. File markers key by file path; `__init__.py`/`README.md` markers key by directory path (trailing `/`).

- [ ] **Step 1: Write the failing test**

```python
# tests/adr/test_code_links.py
from tools.adr.code_links import extract_refs, adr_ids_from_refs, scan_markers

def test_extract_refs_and_ids():
    text = '"""Module.\n\ngoverned-by: ADR-0003, ADR-0001\n"""\n'
    refs = extract_refs(text)
    assert refs == ["ADR-0003", "ADR-0001"]
    assert adr_ids_from_refs(refs) == [3, 1]

def test_scan_markers_keys_files_and_dirs(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "pkg").mkdir()
    # file marker -> keyed by file path
    (tmp_path / "src" / "svc.py").write_text('"""Svc.\ngoverned-by: ADR-0003\n"""\n', encoding="utf-8")
    # __init__.py marker -> keyed by directory path
    (tmp_path / "src" / "pkg" / "__init__.py").write_text('"""Pkg.\ngoverned-by: ADR-0005\n"""\n', encoding="utf-8")
    # README marker -> keyed by directory path
    (tmp_path / "src" / "pkg" / "README.md").write_text("governed-by: ADR-0009\n", encoding="utf-8")
    # unmarked file -> absent
    (tmp_path / "src" / "plain.py").write_text('"""nothing here"""\n', encoding="utf-8")
    markers = scan_markers(str(tmp_path))
    assert markers["src/svc.py"] == ["ADR-0003"]
    assert sorted(markers["src/pkg/"]) == ["ADR-0005", "ADR-0009"]
    assert "src/plain.py" not in markers
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.pyenv/shims/python -m pytest tests/adr/test_code_links.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tools.adr.code_links'`

- [ ] **Step 3: Implement**

```python
# tools/adr/code_links.py
from __future__ import annotations

import os
import re
from typing import Dict, List

_MARKER = re.compile(r"governed-by:\s*(.+)", re.IGNORECASE)
_ADR_ID = re.compile(r"ADR[-\s]?(\d{1,4})", re.IGNORECASE)


def extract_refs(text: str) -> List[str]:
    refs: List[str] = []
    for m in _MARKER.finditer(text):
        refs += [tok for tok in re.split(r"[,\s]+", m.group(1).strip()) if tok]
    return refs


def adr_ids_from_refs(refs: List[str]) -> List[int]:
    ids: List[int] = []
    for ref in refs:
        m = _ADR_ID.search(ref)
        if m:
            ids.append(int(m.group(1)))
    return ids


def scan_markers(root: str, subdirs=("src",)) -> Dict[str, List[str]]:
    """Map repo-relative path -> raw governed-by ref tokens. File markers key by
    the file path; __init__.py / README.md markers key by the directory (trailing /)."""
    result: Dict[str, List[str]] = {}
    for base in subdirs:
        start = os.path.join(root, base)
        if not os.path.isdir(start):
            continue
        for dirpath, _dirs, filenames in os.walk(start):
            for fn in filenames:
                if not (fn.endswith(".py") or fn == "README.md"):
                    continue
                full = os.path.join(dirpath, fn)
                try:
                    text = open(full, encoding="utf-8").read()
                except Exception:
                    continue
                refs = extract_refs(text)
                if not refs:
                    continue
                if fn == "__init__.py" or fn == "README.md":
                    key = os.path.relpath(dirpath, root).replace(os.sep, "/") + "/"
                else:
                    key = os.path.relpath(full, root).replace(os.sep, "/")
                result.setdefault(key, []).extend(refs)
    return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `~/.pyenv/shims/python -m pytest tests/adr/test_code_links.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add tools/adr/code_links.py tests/adr/test_code_links.py
git commit -m "feat(adr): node-agnostic governed-by marker scanner"
```

---

### Task 3: Reverse index `by-code.md`

**Files:**
- Modify: `tools/adr/index.py` (add `render_by_code`, write it, add to `RESERVED`)
- Modify: `tools/adr/check.py` (import `render_by_code`, add to `RENDERERS`)
- Modify: `docs/adr/by-code.md` (generated — created by running `make adr-index`)
- Test: `tests/adr/test_index.py`

**Interfaces:**
- Produces: `render_by_code(adrs: list[Adr]) -> str`; `RESERVED` now includes `by-code.md`; `write_generated` writes it.

- [ ] **Step 1: Write the failing test** (append to `tests/adr/test_index.py`)

```python
def test_render_by_code_and_write(tmp_path):
    from tools.adr.index import render_by_code, write_generated, load_bundle, RESERVED
    assert "by-code.md" in RESERVED
    (tmp_path / "0003-p.md").write_text(
        "---\ntype: ADR\nid: 3\ntitle: P\nstatus: accepted\ndate: 2026-07-04\n"
        "governs:\n  - src/projections/\n---\nbody\n", encoding="utf-8")
    adrs = load_bundle(str(tmp_path))
    table = render_by_code(adrs)
    assert "| src/projections/ | 0003 |" in table
    write_generated(str(tmp_path))
    assert (tmp_path / "by-code.md").read_text() == table
    # by-code.md is reserved -> load_bundle must not parse it as an ADR
    assert [a.id for a in load_bundle(str(tmp_path))] == [3]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.pyenv/shims/python -m pytest tests/adr/test_index.py::test_render_by_code_and_write -v`
Expected: FAIL — `ImportError: cannot import name 'render_by_code'` (or `by-code.md` not in RESERVED)

- [ ] **Step 3: Implement**

In `tools/adr/index.py`, extend `RESERVED` and add the renderer + write:

```python
RESERVED = {"index.md", "log.md", "by-code.md"}
```

```python
def render_by_code(adrs: List[Adr]) -> str:
    mapping: dict = {}
    for a in adrs:
        for path in a.governs:
            mapping.setdefault(path, []).append(a.id)
    lines = ["# Code → ADR map", "", "| code path | governed by |", "| --- | --- |"]
    for path in sorted(mapping):
        ids = ", ".join(f"{i:04d}" for i in sorted(mapping[path]))
        lines.append(f"| {path} | {ids} |")
    return "\n".join(lines) + "\n"
```

In `write_generated`, add a third write:

```python
    with open(os.path.join(adr_dir, "by-code.md"), "w", encoding="utf-8") as fh:
        fh.write(render_by_code(adrs))
```

In `tools/adr/check.py`, extend the import and the `RENDERERS` map:

```python
from tools.adr.index import RESERVED, load_bundle, render_by_code, render_index, render_log
```
```python
RENDERERS = {"index.md": render_index, "log.md": render_log, "by-code.md": render_by_code}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `~/.pyenv/shims/python -m pytest tests/adr/test_index.py tests/adr/test_check.py -v`
Expected: PASS (index + check suites; `check_generated_in_sync` now covers `by-code.md`)

- [ ] **Step 5: Generate the real artifact + commit**

```bash
make adr-index                 # writes docs/adr/by-code.md (empty table until Task 5 backfill)
make adr-check                 # must exit 0; by-code.md now in sync
git add tools/adr/index.py tools/adr/check.py tests/adr/test_index.py docs/adr/by-code.md
git commit -m "feat(adr): generated by-code.md reverse index"
```

---

### Task 4: The four governs guard checks

**Files:**
- Modify: `tools/adr/check.py`
- Test: `tests/adr/test_check.py`

**Interfaces:**
- Consumes: `tools.adr.code_links.scan_markers`, `adr_ids_from_refs`
- Produces: `_path_covered_by`, `check_governs_resolve`, `check_code_markers_resolve`, `check_governs_agreement`, `check_governs_staleness`; `run_all` gains `root="."` and runs all four.

- [ ] **Step 1: Write the failing test** (append to `tests/adr/test_check.py`)

```python
from tools.adr.check import (
    _path_covered_by, check_governs_resolve, check_code_markers_resolve,
    check_governs_agreement, check_governs_staleness,
)

def test_path_covered_by():
    assert _path_covered_by("src/x.py", ["src/x.py"])
    assert _path_covered_by("src/pkg/x.py", ["src/pkg/"])   # parent-dir match
    assert not _path_covered_by("src/other.py", ["src/pkg/"])

def test_governs_resolve_flags_missing(tmp_path):
    a = _adr(id=3, governs=["src/gone/"])
    msgs = " ".join(f.message for f in check_governs_resolve([a], root=str(tmp_path)))
    assert "0003 governs src/gone/ which does not exist" in msgs

def test_code_markers_resolve_flags_dangling():
    markers = {"src/x.py": ["ADR-0099"]}
    msgs = " ".join(f.message for f in check_code_markers_resolve(markers, [1, 3]))
    assert "src/x.py claims ADR-0099 which does not exist" in msgs

def test_governs_agreement_both_directions():
    a = _adr(id=3, governs=["src/projections/"])
    # direction 1: ADR governs a path nothing marks -> finding
    msgs1 = " ".join(f.message for f in check_governs_agreement([a], {}))
    assert "0003 governs src/projections/ but nothing there is marked" in msgs1
    # direction 2: marker claims ADR that doesn't govern the path -> finding
    markers = {"src/other.py": ["ADR-0003"]}
    msgs2 = " ".join(f.message for f in check_governs_agreement([a], markers))
    assert "src/other.py is marked governed-by ADR-0003 but 0003 does not govern it" in msgs2
    # satisfied: marker under a governed parent dir, and dir marker present
    markers_ok = {"src/projections/": ["ADR-0003"]}
    assert check_governs_agreement([a], markers_ok) == []

def test_governs_staleness_with_injected_ts():
    a = _adr(id=3, governs=["src/projections/"], path="docs/adr/0003.md")
    def fake_ts(p):
        return 200 if p == "src/projections/" else 100   # governed code newer
    msgs = " ".join(f.message for f in check_governs_staleness([a], ts_fn=fake_ts))
    assert "0003: governed code src/projections/ changed after the ADR" in msgs
```

> Note: the existing `_adr` helper in this test file builds an `Adr`; it accepts
> keyword overrides. If it does not yet pass `governs` through, extend it to include
> `governs=kw.get("governs", [])` in the same step.

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.pyenv/shims/python -m pytest tests/adr/test_check.py -k governs -v`
Expected: FAIL — the new functions don't exist / `_adr` lacks `governs`

- [ ] **Step 3: Implement**

In `tools/adr/check.py`, add the import near the top:

```python
from tools.adr.code_links import adr_ids_from_refs, scan_markers
```

Add the helper + four checks (after `check_parseable`):

```python
def _path_covered_by(path: str, governs: List[str]) -> bool:
    for g in governs:
        if g == path:
            return True
        if g.endswith("/") and path.startswith(g):
            return True
    return False


def check_governs_resolve(adrs: List[Adr], root: str = ".") -> List[Finding]:
    findings: List[Finding] = []
    for a in adrs:
        for p in a.governs:
            if not os.path.exists(os.path.join(root, p)):
                findings.append(Finding(f"{a.id:04d} governs {p} which does not exist"))
    return findings


def check_code_markers_resolve(markers: dict, adr_ids: List[int]) -> List[Finding]:
    findings: List[Finding] = []
    known = set(adr_ids)
    for path in sorted(markers):
        for mid in adr_ids_from_refs(markers[path]):
            if mid not in known:
                findings.append(Finding(f"{path} claims ADR-{mid:04d} which does not exist"))
    return findings


def check_governs_agreement(adrs: List[Adr], markers: dict) -> List[Finding]:
    findings: List[Finding] = []
    by_id = {a.id: a for a in adrs}
    for a in adrs:
        for p in a.governs:
            if a.id not in adr_ids_from_refs(markers.get(p, [])):
                findings.append(
                    Finding(f"{a.id:04d} governs {p} but nothing there is marked governed-by ADR-{a.id:04d}")
                )
    for path in sorted(markers):
        for mid in adr_ids_from_refs(markers[path]):
            a = by_id.get(mid)
            if a is None:
                continue  # dangling ref handled by check_code_markers_resolve
            if not _path_covered_by(path, a.governs):
                findings.append(
                    Finding(f"{path} is marked governed-by ADR-{mid:04d} but {mid:04d} does not govern it")
                )
    return findings


def check_governs_staleness(adrs: List[Adr],
                            ts_fn: Callable[[str], Optional[int]] = git_committer_ts) -> List[Finding]:
    findings: List[Finding] = []
    for a in adrs:
        if not a.path:
            continue
        adr_ts = ts_fn(a.path)
        if adr_ts is None:
            continue
        for p in a.governs:
            p_ts = ts_fn(p)
            if p_ts is not None and p_ts > adr_ts:
                findings.append(Finding(f"{a.id:04d}: governed code {p} changed after the ADR — revisit?"))
    return findings
```

Update `run_all`:

```python
def run_all(adr_dir: str, specs_dir: str, root: str = ".") -> List[Finding]:
    adrs = load_bundle(adr_dir)
    markers = scan_markers(root)
    findings: List[Finding] = []
    findings += check_structural(adrs)
    findings += check_generated_in_sync(adr_dir, adrs)
    findings += check_specs_reference_adr(specs_dir)
    findings += check_staleness(adrs)
    findings += check_parseable(adr_dir)
    findings += check_governs_resolve(adrs, root)
    findings += check_code_markers_resolve(markers, [a.id for a in adrs])
    findings += check_governs_agreement(adrs, markers)
    findings += check_governs_staleness(adrs)
    return findings
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `~/.pyenv/shims/python -m pytest tests/adr/test_check.py -v`
Expected: PASS (all check tests, new + existing)

- [ ] **Step 5: Commit**

```bash
git add tools/adr/check.py tests/adr/test_check.py
git commit -m "feat(adr): governs resolve/agreement/staleness guard checks"
```

---

### Task 5: Backfill `governs` + `governed-by` markers

Content task: wire the ~8-10 ADRs that map to code, add the matching markers, regenerate, and get `adr-check` clean.

**Files:**
- Modify: `docs/adr/0001-*.md`, `0003-*.md`, `0005-*.md`, `0009-*.md`, `0011-*.md`, `0013-*.md`, … (add `governs`)
- Modify: the governed modules/packages (add `governed-by` markers to docstrings / `__init__.py` / `README.md`)
- Modify (generated): `docs/adr/by-code.md`

- [ ] **Step 1: Decide the mapping and add `governs` to each ADR**

Read each ADR and the `src/` tree; add a `governs:` list only where the decision cleanly maps to code. Illustrative (finalize against the real tree):

| ADR | governs |
|---|---|
| 0003 projection service sole writer | `src/projections/` |
| 0005 layered mine | `src/ingestion/`, `src/enrichment/`, `src/lens/`, `src/export/` |
| 0009 lens engine generic | `src/lens/` |
| 0011 deterministic+review resolution | `src/resolution/` |
| 0013 read-side OKF exporter | `src/export/` |

ADRs with no clean code target (0007 focused-calls, 0015 the ADR system, etc.) omit `governs`. Verify each `governs` path exists.

- [ ] **Step 2: Add the matching `governed-by` markers**

For each governed path, add the back-marker at the granularity's home:
- directory target `src/lens/` → add to `src/lens/__init__.py` docstring (create/extend the module docstring):
  ```python
  """<existing summary>

  governed-by: ADR-0005, ADR-0009
  """
  ```
- file target → the file's module docstring.

Add markers **only** for paths an ADR actually governs; every governed path must get a marker naming that ADR (direction-1 agreement), and every marker's ADR must govern that path or a parent (direction-2).

- [ ] **Step 3: Regenerate + check until clean**

```bash
make adr-index          # regenerates docs/adr/by-code.md with the real mapping
make adr-check          # iterate until clean
```
Resolve any `governs ... does not exist`, agreement mismatches, or dangling-marker findings. Staleness findings on freshly-touched code are acceptable to leave (informational) — but note them; they should be minimal since the ADR files are being committed alongside.

- [ ] **Step 4: Commit**

```bash
git add docs/adr/ src/
git commit -m "docs(adr): backfill governs + governed-by markers linking ADRs to code"
```

---

### Task 6: `adr where <path>` CLI

The payoff of the reverse index: stand at a path, get its governing ADRs.

**Files:**
- Modify: `tools/adr/__main__.py`
- Test: `tests/adr/test_cli.py`

**Interfaces:**
- Produces: `python -m tools.adr where <path>` → prints governing ADRs (exit 0).

- [ ] **Step 1: Write the failing test** (append to `tests/adr/test_cli.py`)

```python
def test_cli_where_reports_governing_adr(tmp_path):
    import subprocess, sys
    (tmp_path / "0003-p.md").write_text(
        "---\ntype: ADR\nid: 3\ntitle: Projections\nstatus: accepted\ndate: 2026-07-04\n"
        "governs:\n  - src/projections/\n---\nbody\n", encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, "-m", "tools.adr", "where", "src/projections/svc.py", "--adr-dir", str(tmp_path)],
        capture_output=True, text=True)
    assert proc.returncode == 0
    assert "ADR-0003" in proc.stdout
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.pyenv/shims/python -m pytest tests/adr/test_cli.py::test_cli_where_reports_governing_adr -v`
Expected: FAIL — `invalid choice: 'where'`

- [ ] **Step 3: Implement**

In `tools/adr/__main__.py`, add the handler (import `load_bundle`, `_path_covered_by`):

```python
def cmd_where(args) -> int:
    from tools.adr.index import load_bundle
    from tools.adr.check import _path_covered_by
    adrs = load_bundle(args.adr_dir)
    hits = [a for a in adrs if _path_covered_by(args.path, a.governs)]
    if hits:
        for a in hits:
            print(f"ADR-{a.id:04d} {a.title}")
    else:
        print(f"no ADR governs {args.path}")
    return 0
```

Register the subparser (alongside the others, using the shared `common` parent) and add to the dispatch dict:

```python
    p_where = sub.add_parser("where", parents=[common]); p_where.add_argument("path")
```
```python
        "where": cmd_where,
```

- [ ] **Step 4: Run test to verify it passes**

Run: `~/.pyenv/shims/python -m pytest tests/adr/test_cli.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tools/adr/__main__.py tests/adr/test_cli.py
git commit -m "feat(adr): adr where <path> — reverse lookup of governing ADRs"
```

---

## Final verification

- [ ] `~/.pyenv/shims/python -m pytest tests/adr/ -v` — all green.
- [ ] `make adr-check` — clean on the real backfilled bundle (staleness-on-governs findings, if any, noted and accepted).
- [ ] `make adr-index` then `git status` — `docs/adr/by-code.md` regenerates identically (in sync).
- [ ] `python -m tools.adr where src/lens/engine.py` (or a real governed path) prints the governing ADR(s).
