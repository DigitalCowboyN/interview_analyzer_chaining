# ADR + OKF Knowledge System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up a durable, drift-resistant architectural-decision corpus (`docs/adr/` as an OKF v0.1-conformant bundle), a root `CLAUDE.md` instruction surface, and a five-layer non-blocking read→capture→guard loop around them.

**Architecture:** A small pure-Python package `tools/adr/` (model → load/render → checks → intent) with a thin CLI (`python -m tools.adr {index|check|new|context|nudge}`). Content lives in `docs/adr/` (authored ADRs + generated `index.md`/`log.md`). The loop is wired through Claude Code hooks and a non-blocking git `pre-commit` hook. Everything warns; nothing blocks.

**Tech Stack:** Python 3 (stdlib + PyYAML via the existing `src.ingestion.front_matter.parse_front_matter`), pytest, flake8/black, Make, Claude Code hooks (`.claude/settings.local.json`), git hooks.

## Global Constraints

- **Non-blocking, always.** Every check, hook, and git hook exits 0 and only prints warnings — never fails a command, blocks a commit, or halts a tool call. (spec: "All layers are non-blocking.")
- **`index.md` and `log.md` are generated, never hand-edited** (reserved OKF names).
- **ADR frontmatter schema is fixed:** `type: ADR`, `id` (int), `title`, `status ∈ {proposed, accepted, superseded, deprecated}`, `date`, `supersedes` (list[int]), `superseded_by` (list[int]), `tags` (list[str]), `source` (repo-relative path).
- **DRY:** reuse `src.ingestion.front_matter.parse_front_matter` for YAML frontmatter; do not re-implement a parser.
- **Backfill is human-curated**, not machine-generated. Tooling generates only `index.md`/`log.md`.
- **Tooling lives in `tools/adr/`; tests in `tests/adr/`.** Follow repo test conventions (unit tests unmarked; `integration` marker only for env-gated tests).

---

### Task 1: ADR model (`tools/adr/model.py`)

**Files:**
- Create: `tools/adr/__init__.py` (empty)
- Create: `tools/adr/model.py`
- Test: `tests/adr/__init__.py` (empty), `tests/adr/test_model.py`

**Interfaces:**
- Consumes: `src.ingestion.front_matter.parse_front_matter(text) -> (Optional[dict], int)`
- Produces:
  - `VALID_STATUS: set[str]`
  - `@dataclass Adr` with fields `id:int, title:str, status:str, date:str, supersedes:list[int], superseded_by:list[int], tags:list[str], source:Optional[str], path:Optional[str], body:str`
  - `validate_frontmatter(fm: dict) -> list[str]` (returns human-readable problem strings; empty = valid)
  - `parse_adr(text: str, path: Optional[str] = None) -> Adr`

- [ ] **Step 1: Write the failing test**

```python
# tests/adr/test_model.py
import pytest
from tools.adr.model import parse_adr, validate_frontmatter, Adr

GOOD = """---
type: ADR
id: 1
title: EventStoreDB is the single source of truth
status: accepted
date: 2026-07-04
supersedes: []
superseded_by: []
tags: [event-sourcing]
source: docs/architecture/README.md
---
## Context
Body text here.
"""

def test_parse_adr_reads_all_fields():
    adr = parse_adr(GOOD, path="docs/adr/0001-esdb.md")
    assert adr.id == 1
    assert adr.status == "accepted"
    assert adr.tags == ["event-sourcing"]
    assert adr.source == "docs/architecture/README.md"
    assert adr.path == "docs/adr/0001-esdb.md"
    assert "## Context" in adr.body

def test_validate_frontmatter_flags_missing_keys_and_bad_status():
    problems = validate_frontmatter({"type": "ADR", "id": 1, "title": "x", "status": "bogus"})
    joined = " ".join(problems)
    assert "date" in joined            # missing required key
    assert "bogus" in joined           # invalid status

def test_validate_frontmatter_accepts_good():
    fm = {"type": "ADR", "id": 1, "title": "x", "status": "accepted", "date": "2026-07-04"}
    assert validate_frontmatter(fm) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/adr/test_model.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tools.adr'`

- [ ] **Step 3: Write minimal implementation**

```python
# tools/adr/model.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.ingestion.front_matter import parse_front_matter

VALID_STATUS = {"proposed", "accepted", "superseded", "deprecated"}
REQUIRED_KEYS = ("type", "id", "title", "status", "date")


@dataclass
class Adr:
    id: int
    title: str
    status: str
    date: str
    supersedes: List[int] = field(default_factory=list)
    superseded_by: List[int] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    source: Optional[str] = None
    path: Optional[str] = None
    body: str = ""


def validate_frontmatter(fm: Dict[str, Any]) -> List[str]:
    problems: List[str] = []
    for key in REQUIRED_KEYS:
        if key not in fm:
            problems.append(f"missing required key: {key}")
    status = fm.get("status")
    if status is not None and status not in VALID_STATUS:
        problems.append(f"invalid status: {status!r} (want one of {sorted(VALID_STATUS)})")
    return problems


def _int_list(value: Any) -> List[int]:
    return [int(x) for x in (value or [])]


def parse_adr(text: str, path: Optional[str] = None) -> Adr:
    fm, offset = parse_front_matter(text)
    if fm is None:
        raise ValueError(f"{path or '<text>'}: missing front matter")
    return Adr(
        id=int(fm["id"]),
        title=str(fm["title"]),
        status=str(fm["status"]),
        date=str(fm["date"]),
        supersedes=_int_list(fm.get("supersedes")),
        superseded_by=_int_list(fm.get("superseded_by")),
        tags=list(fm.get("tags") or []),
        source=fm.get("source"),
        path=path,
        body=text[offset:],
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/adr/test_model.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add tools/adr/__init__.py tools/adr/model.py tests/adr/__init__.py tests/adr/test_model.py
git commit -m "feat(adr): ADR frontmatter model + validation"
```

---

### Task 2: Bundle load + index/log rendering (`tools/adr/index.py`)

**Files:**
- Create: `tools/adr/index.py`
- Test: `tests/adr/test_index.py`

**Interfaces:**
- Consumes: `tools.adr.model.parse_adr`, `Adr`
- Produces:
  - `RESERVED: set[str]` = `{"index.md", "log.md"}`
  - `load_bundle(adr_dir: str) -> list[Adr]` (skips reserved names, sorted by id)
  - `render_index(adrs: list[Adr]) -> str`
  - `render_log(adrs: list[Adr]) -> str`
  - `write_generated(adr_dir: str) -> None` (writes index.md + log.md)

- [ ] **Step 1: Write the failing test**

```python
# tests/adr/test_index.py
from tools.adr.index import load_bundle, render_index, render_log, write_generated

def _write(dir_, name, body):
    (dir_ / name).write_text(body, encoding="utf-8")

ADR_TMPL = """---
type: ADR
id: {id}
title: {title}
status: {status}
date: {date}
supersedes: {supersedes}
superseded_by: []
tags: []
source: docs/x.md
---
body
"""

def test_load_bundle_skips_reserved_and_sorts(tmp_path):
    _write(tmp_path, "index.md", "# generated\n")
    _write(tmp_path, "0002-b.md", ADR_TMPL.format(id=2, title="B", status="accepted", date="2026-07-05", supersedes="[]"))
    _write(tmp_path, "0001-a.md", ADR_TMPL.format(id=1, title="A", status="accepted", date="2026-07-04", supersedes="[]"))
    adrs = load_bundle(str(tmp_path))
    assert [a.id for a in adrs] == [1, 2]        # sorted, index.md skipped

def test_render_index_and_log(tmp_path):
    _write(tmp_path, "0001-a.md", ADR_TMPL.format(id=1, title="A", status="accepted", date="2026-07-04", supersedes="[]"))
    adrs = load_bundle(str(tmp_path))
    idx = render_index(adrs)
    assert "| 0001 | A | accepted |" in idx
    log = render_log(adrs)
    assert "0001" in log and "2026-07-04" in log

def test_write_generated_is_idempotent(tmp_path):
    _write(tmp_path, "0001-a.md", ADR_TMPL.format(id=1, title="A", status="accepted", date="2026-07-04", supersedes="[]"))
    write_generated(str(tmp_path))
    first = (tmp_path / "index.md").read_text()
    write_generated(str(tmp_path))
    assert (tmp_path / "index.md").read_text() == first
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/adr/test_index.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tools.adr.index'`

- [ ] **Step 3: Write minimal implementation**

```python
# tools/adr/index.py
from __future__ import annotations

import glob
import os
from typing import List

from tools.adr.model import Adr, parse_adr

RESERVED = {"index.md", "log.md"}


def load_bundle(adr_dir: str) -> List[Adr]:
    adrs: List[Adr] = []
    for path in sorted(glob.glob(os.path.join(adr_dir, "*.md"))):
        if os.path.basename(path) in RESERVED:
            continue
        with open(path, encoding="utf-8") as fh:
            adrs.append(parse_adr(fh.read(), path=path))
    return sorted(adrs, key=lambda a: a.id)


def render_index(adrs: List[Adr]) -> str:
    lines = ["# ADR Index", "", "| id | title | status |", "| --- | --- | --- |"]
    for a in adrs:
        lines.append(f"| {a.id:04d} | {a.title} | {a.status} |")
    return "\n".join(lines) + "\n"


def render_log(adrs: List[Adr]) -> str:
    lines = ["# Decision Log", ""]
    for a in sorted(adrs, key=lambda a: (a.date, a.id)):
        sup = ""
        if a.supersedes:
            sup = " (supersedes " + ", ".join(f"{i:04d}" for i in a.supersedes) + ")"
        lines.append(f"- {a.date} — **{a.id:04d}** {a.title} · _{a.status}_{sup}")
    return "\n".join(lines) + "\n"


def write_generated(adr_dir: str) -> None:
    adrs = load_bundle(adr_dir)
    with open(os.path.join(adr_dir, "index.md"), "w", encoding="utf-8") as fh:
        fh.write(render_index(adrs))
    with open(os.path.join(adr_dir, "log.md"), "w", encoding="utf-8") as fh:
        fh.write(render_log(adrs))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/adr/test_index.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add tools/adr/index.py tests/adr/test_index.py
git commit -m "feat(adr): bundle loader + index/log generation"
```

---

### Task 3: The three guard checks (`tools/adr/check.py`)

**Files:**
- Create: `tools/adr/check.py`
- Test: `tests/adr/test_check.py`

**Interfaces:**
- Consumes: `tools.adr.model` (`Adr`, `VALID_STATUS`), `tools.adr.index` (`render_index`, `render_log`, `RESERVED`)
- Produces:
  - `@dataclass Finding(message: str)`
  - `check_structural(adrs: list[Adr]) -> list[Finding]`
  - `check_generated_in_sync(adr_dir: str, adrs: list[Adr]) -> list[Finding]`
  - `check_specs_reference_adr(specs_dir: str) -> list[Finding]`
  - `check_staleness(adrs: list[Adr], ts_fn=git_committer_ts) -> list[Finding]`
  - `git_committer_ts(path: str) -> Optional[int]`
  - `run_all(adr_dir: str, specs_dir: str) -> list[Finding]`

- [ ] **Step 1: Write the failing test**

```python
# tests/adr/test_check.py
from tools.adr.model import Adr
from tools.adr.check import (
    check_structural, check_specs_reference_adr, check_staleness, Finding,
)

def _adr(**kw):
    base = dict(id=1, title="A", status="accepted", date="2026-07-04",
                supersedes=[], superseded_by=[], tags=[], source="docs/x.md",
                path=f"docs/adr/{kw.get('id',1):04d}.md", body="")
    base.update(kw); return Adr(**base)

def test_structural_flags_duplicate_id_and_one_directional_supersede():
    a = _adr(id=1, supersedes=[2])
    b = _adr(id=2, superseded_by=[])          # missing back-edge
    c = _adr(id=2)                             # duplicate id 2
    msgs = " ".join(f.message for f in check_structural([a, b, c]))
    assert "duplicate id 0002" in msgs
    assert "0001 supersedes 0002" in msgs      # one-directional edge flagged

def test_structural_flags_bad_status():
    msgs = " ".join(f.message for f in check_structural([_adr(id=1, status="bogus")]))
    assert "invalid status" in msgs

def test_specs_reference_adr_warns_when_decisions_locked_no_adr(tmp_path):
    (tmp_path / "s1.md").write_text("## Decisions locked\nwe chose X\n", encoding="utf-8")
    (tmp_path / "s2.md").write_text("## Decisions locked\nsee ADR-0003\n", encoding="utf-8")
    msgs = [f.message for f in check_specs_reference_adr(str(tmp_path))]
    assert any("s1.md" in m for m in msgs)
    assert not any("s2.md" in m for m in msgs)   # references an ADR → no warning

def test_staleness_warns_when_source_newer_than_adr():
    a = _adr(id=1, source="docs/x.md", path="docs/adr/0001.md")
    def fake_ts(path):
        return 200 if path == "docs/x.md" else 100   # source newer than adr
    msgs = [f.message for f in check_staleness([a], ts_fn=fake_ts)]
    assert any("0001" in m and "docs/x.md" in m for m in msgs)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/adr/test_check.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tools.adr.check'`

- [ ] **Step 3: Write minimal implementation**

```python
# tools/adr/check.py
from __future__ import annotations

import glob
import os
import re
import subprocess
from dataclasses import dataclass
from typing import Callable, List, Optional

from tools.adr.index import RESERVED, load_bundle, render_index, render_log
from tools.adr.model import VALID_STATUS, Adr

DECISION_MARKERS = ("decisions locked", "rejected alternative")
ADR_REF = re.compile(r"\bADR[-\s]?\d{1,4}\b|docs/adr/\d{4}", re.IGNORECASE)


@dataclass
class Finding:
    message: str


def check_structural(adrs: List[Adr]) -> List[Finding]:
    findings: List[Finding] = []
    seen: dict = {}
    for a in adrs:
        if a.id in seen:
            findings.append(Finding(f"duplicate id {a.id:04d}: {a.path} and {seen[a.id]}"))
        seen[a.id] = a.path
        if a.status not in VALID_STATUS:
            findings.append(Finding(f"{a.id:04d}: invalid status {a.status!r}"))
    by_id = {a.id: a for a in adrs}
    for a in adrs:
        for target in a.supersedes:
            other = by_id.get(target)
            if other is None:
                findings.append(Finding(f"{a.id:04d} supersedes unknown {target:04d}"))
            elif a.id not in other.superseded_by:
                findings.append(
                    Finding(f"{a.id:04d} supersedes {target:04d} but {target:04d}.superseded_by lacks it")
                )
    return findings


def check_generated_in_sync(adr_dir: str, adrs: List[Adr]) -> List[Finding]:
    findings: List[Finding] = []
    for name, render in (("index.md", render_index), ("log.md", render_log)):
        path = os.path.join(adr_dir, name)
        want = render(adrs)
        have = open(path, encoding="utf-8").read() if os.path.exists(path) else ""
        if want != have:
            findings.append(Finding(f"{name} out of sync — run `make adr-index`"))
    return findings


def check_specs_reference_adr(specs_dir: str) -> List[Finding]:
    findings: List[Finding] = []
    for path in sorted(glob.glob(os.path.join(specs_dir, "*.md"))):
        text = open(path, encoding="utf-8").read()
        low = text.lower()
        if any(m in low for m in DECISION_MARKERS) and not ADR_REF.search(text):
            findings.append(Finding(f"{os.path.basename(path)} locks decisions but references no ADR"))
    return findings


def git_committer_ts(path: str) -> Optional[int]:
    try:
        out = subprocess.run(
            ["git", "log", "-1", "--format=%ct", "--", path],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        return int(out) if out else None
    except Exception:
        return None


def check_staleness(adrs: List[Adr],
                    ts_fn: Callable[[str], Optional[int]] = git_committer_ts) -> List[Finding]:
    findings: List[Finding] = []
    for a in adrs:
        if not a.source or not a.path:
            continue
        src_ts, adr_ts = ts_fn(a.source), ts_fn(a.path)
        if src_ts is not None and adr_ts is not None and src_ts > adr_ts:
            findings.append(Finding(f"{a.id:04d}: source {a.source} changed after the ADR"))
    return findings


def run_all(adr_dir: str, specs_dir: str) -> List[Finding]:
    adrs = load_bundle(adr_dir)
    findings: List[Finding] = []
    findings += check_structural(adrs)
    findings += check_generated_in_sync(adr_dir, adrs)
    findings += check_specs_reference_adr(specs_dir)
    findings += check_staleness(adrs)
    return findings
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/adr/test_check.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add tools/adr/check.py tests/adr/test_check.py
git commit -m "feat(adr): structural, spec-reference, sync, and staleness checks"
```

---

### Task 4: Architectural-intent matcher (`tools/adr/intent.py`)

**Files:**
- Create: `tools/adr/intent.py`
- Test: `tests/adr/test_intent.py`

**Interfaces:**
- Produces: `is_architectural(prompt: str) -> bool`

- [ ] **Step 1: Write the failing test**

```python
# tests/adr/test_intent.py
import pytest
from tools.adr.intent import is_architectural

@pytest.mark.parametrize("prompt", [
    "Let's brainstorm the design for the new exporter",
    "Should we switch the queue to Redis? What are the trade-offs?",
    "write a spec for the ingestion refactor",
])
def test_matches_architectural_intent(prompt):
    assert is_architectural(prompt) is True

@pytest.mark.parametrize("prompt", [
    "fix the failing test in test_reader.py",
    "what does line 42 do?",
    "bump the black version",
])
def test_ignores_non_architectural(prompt):
    assert is_architectural(prompt) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/adr/test_intent.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tools.adr.intent'`

- [ ] **Step 3: Write minimal implementation**

```python
# tools/adr/intent.py
from __future__ import annotations

import re

_KEYWORDS = (
    "architect", "design", "decision", "trade-off", "tradeoff", "should we",
    "brainstorm", "spec", "approach", "refactor", "adr", "alternative",
)
_PATTERN = re.compile("|".join(re.escape(k) for k in _KEYWORDS), re.IGNORECASE)


def is_architectural(prompt: str) -> bool:
    return bool(_PATTERN.search(prompt or ""))
```

> Note: "plan" was intentionally left out of the keyword set — it fires on
> everyday "what's the plan for today" chatter and inflates false positives.
> Add it later only if the read-side hook feels too quiet.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/adr/test_intent.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add tools/adr/intent.py tests/adr/test_intent.py
git commit -m "feat(adr): architectural-intent matcher for the read-side hook"
```

---

### Task 5: CLI + scaffolding + Makefile wiring (`tools/adr/__main__.py`, `scaffold.py`)

**Files:**
- Create: `tools/adr/scaffold.py`
- Create: `tools/adr/__main__.py`
- Modify: `Makefile` (add `adr-check`, `adr-index` targets)
- Test: `tests/adr/test_scaffold.py`, `tests/adr/test_cli.py`

**Interfaces:**
- Consumes: `tools.adr.index.write_generated`, `tools.adr.check.run_all`, `tools.adr.intent.is_architectural`
- Produces:
  - `tools.adr.scaffold.next_id(adr_dir: str) -> int`
  - `tools.adr.scaffold.new_adr(adr_dir: str, title: str) -> str` (returns created path)
  - CLI: `python -m tools.adr {index|check|new "<title>"|context|nudge}` — all exit 0

- [ ] **Step 1: Write the failing test**

```python
# tests/adr/test_scaffold.py
from tools.adr.scaffold import next_id, new_adr
from tools.adr.model import parse_adr

def test_next_id_starts_at_one_and_increments(tmp_path):
    assert next_id(str(tmp_path)) == 1
    new_adr(str(tmp_path), "First decision")
    assert next_id(str(tmp_path)) == 2

def test_new_adr_writes_parseable_stub(tmp_path):
    path = new_adr(str(tmp_path), "Use Redis for the queue")
    adr = parse_adr(open(path, encoding="utf-8").read(), path=path)
    assert adr.id == 1
    assert adr.status == "proposed"
    assert "use-redis-for-the-queue" in path
    assert "## Decision" in adr.body
```

```python
# tests/adr/test_cli.py
import subprocess, sys

def test_cli_check_exits_zero_even_with_findings(tmp_path):
    # a bundle with a broken supersede edge would produce findings, but must still exit 0
    (tmp_path / "0001-a.md").write_text(
        "---\ntype: ADR\nid: 1\ntitle: A\nstatus: accepted\ndate: 2026-07-04\n"
        "supersedes: [2]\nsuperseded_by: []\ntags: []\nsource: docs/x.md\n---\nbody\n",
        encoding="utf-8")
    specs = tmp_path / "specs"; specs.mkdir()
    proc = subprocess.run(
        [sys.executable, "-m", "tools.adr", "check", "--adr-dir", str(tmp_path), "--specs-dir", str(specs)],
        capture_output=True, text=True)
    assert proc.returncode == 0            # non-blocking guarantee
    assert "0001 supersedes unknown 0002" in (proc.stdout + proc.stderr)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/adr/test_scaffold.py tests/adr/test_cli.py -v`
Expected: FAIL — `ModuleNotFoundError` / `No module named tools.adr.scaffold`

- [ ] **Step 3: Write minimal implementation**

```python
# tools/adr/scaffold.py
from __future__ import annotations

import os
import re

from tools.adr.index import load_bundle

_TEMPLATE = """---
type: ADR
id: {id}
title: {title}
status: proposed
date: {date}
supersedes: []
superseded_by: []
tags: []
source:
---
## Context

## Decision

## Consequences

## Alternatives considered
"""


def _slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def next_id(adr_dir: str) -> int:
    adrs = load_bundle(adr_dir) if os.path.isdir(adr_dir) else []
    return (max((a.id for a in adrs), default=0)) + 1


def new_adr(adr_dir: str, title: str, date: str = "TODO-SET-DATE") -> str:
    os.makedirs(adr_dir, exist_ok=True)
    nid = next_id(adr_dir)
    path = os.path.join(adr_dir, f"{nid:04d}-{_slug(title)}.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(_TEMPLATE.format(id=nid, title=title, date=date))
    return path
```

```python
# tools/adr/__main__.py
from __future__ import annotations

import argparse
import json
import sys

from tools.adr.check import run_all
from tools.adr.index import write_generated
from tools.adr.intent import is_architectural
from tools.adr.scaffold import new_adr

DEFAULT_ADR_DIR = "docs/adr"
DEFAULT_SPECS_DIR = "docs/superpowers/specs"


def _read_stdin_json() -> dict:
    try:
        return json.loads(sys.stdin.read() or "{}")
    except Exception:
        return {}


def cmd_index(args) -> int:
    write_generated(args.adr_dir)
    print(f"regenerated {args.adr_dir}/index.md and log.md")
    return 0


def cmd_check(args) -> int:
    findings = run_all(args.adr_dir, args.specs_dir)
    if findings:
        print(f"adr-check: {len(findings)} warning(s):")
        for f in findings:
            print(f"  - {f.message}")
    else:
        print("adr-check: clean")
    return 0  # NON-BLOCKING: always 0


def cmd_new(args) -> int:
    path = new_adr(args.adr_dir, args.title)
    print(f"created {path} — fill it in, set `source:` and `date:`, then `make adr-index`")
    return 0


def cmd_context(args) -> int:
    # UserPromptSubmit hook: stdout is injected as context. Quiet unless architectural.
    prompt = _read_stdin_json().get("prompt", "")
    if is_architectural(prompt):
        try:
            print(open(f"{args.adr_dir}/index.md", encoding="utf-8").read())
            print("(Before locking a decision, consult these ADRs; supersede rather than silently override.)")
        except FileNotFoundError:
            pass
    return 0


def cmd_nudge(args) -> int:
    # PostToolUse(Write) hook: remind to capture decisions when a spec lands.
    path = _read_stdin_json().get("tool_input", {}).get("file_path", "")
    if "docs/superpowers/specs/" in path.replace("\\", "/"):
        print("This spec may lock decisions — capture them as ADR(s) "
              "(`python -m tools.adr new \"<title>\"`) and set `source:` to this spec.")
    return 0


def main(argv=None) -> int:
    # Shared options live on a parent parser so they are valid AFTER the
    # subcommand (e.g. `tools.adr check --adr-dir X --specs-dir Y`).
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--adr-dir", default=DEFAULT_ADR_DIR)
    common.add_argument("--specs-dir", default=DEFAULT_SPECS_DIR)

    parser = argparse.ArgumentParser(prog="tools.adr")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("index", parents=[common])
    sub.add_parser("check", parents=[common])
    p_new = sub.add_parser("new", parents=[common]); p_new.add_argument("title")
    sub.add_parser("context", parents=[common])
    sub.add_parser("nudge", parents=[common])
    args = parser.parse_args(argv)
    return {
        "index": cmd_index, "check": cmd_check, "new": cmd_new,
        "context": cmd_context, "nudge": cmd_nudge,
    }[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
```

Add to `Makefile` (place beside `lint`):

```makefile
adr-check:
	@$(PYTHON) -m tools.adr check

adr-index:
	@$(PYTHON) -m tools.adr index
```

- [ ] **Step 4: Run tests + a real CLI smoke**

Run: `python -m pytest tests/adr/test_scaffold.py tests/adr/test_cli.py -v`
Expected: PASS (4 passed)
Run: `make adr-check`
Expected: prints `adr-check: clean` (empty bundle) and exits 0

- [ ] **Step 5: Commit**

```bash
git add tools/adr/scaffold.py tools/adr/__main__.py Makefile tests/adr/test_scaffold.py tests/adr/test_cli.py
git commit -m "feat(adr): CLI (index/check/new/context/nudge) + scaffold + make targets"
```

---

### Task 6: Backfill the ADR corpus (`docs/adr/*`)

This task authors content, not code. Produce ~15 ADRs by harvesting the existing
decision corpus, then generate `index.md`/`log.md` and prove the bundle is clean.

**Files:**
- Create: `docs/adr/0001-*.md` … `docs/adr/00NN-*.md` (one per decision)
- Create (generated): `docs/adr/index.md`, `docs/adr/log.md`

**Harvest sources → decisions (finalize the exact split while writing):**
- `docs/architecture/README.md` "Load-bearing ideas:" (line 48) → ESDB as single source of truth; CQRS write/read split; event-sourced projection into Neo4j.
- `docs/superpowers/specs/2026-07-04-mine-layers-design.md` → layered mine (ingestion → analysis → lens → export); overlay-not-rewrite.
- `docs/superpowers/specs/2026-07-10-layer4-schema-v2-design.md` → schema v2 decision(s).
- `docs/superpowers/specs/2026-07-10-okf-export-design.md` → read-side exporter over Neo4j, zero per-lens code.
- `docs/superpowers/specs/2026-07-16-m46-graphrag-ask-design.md` → graphrag approach **and the supersession edge** over the 2026-07-04 spec's "borrow neo4j-graphrag-python" line.

- [ ] **Step 1: Scaffold and author each ADR**

For each decision, run `python -m tools.adr new "<title>"`, then fill the stub. Every ADR follows this shape (worked example):

```markdown
---
type: ADR
id: 1
title: EventStoreDB is the single source of truth
status: accepted
date: 2026-07-04
supersedes: []
superseded_by: []
tags: [event-sourcing, write-side]
source: docs/architecture/README.md
---
## Context
The system needs an authoritative record of everything that happened so read
models (Neo4j) can be rebuilt and corrections replayed.

## Decision
EventStoreDB holds the canonical event log; Neo4j is a disposable projection.

## Consequences
Read models are rebuildable; all writes go through events; projection lag is a
first-class concern.

## Alternatives considered
Neo4j-as-source-of-truth (rejected: no replay, corrections destructive).
```

**Model the supersession edge explicitly.** The graphrag ADR carries
`supersedes: [<id of the neo4j-graphrag ADR>]`, and that older ADR is set to
`status: superseded` with `superseded_by: [<graphrag ADR id>]`. This is the
drift the format exists to make visible — get it right.

- [ ] **Step 2: Generate index + log**

Run: `make adr-index`
Expected: writes `docs/adr/index.md` and `docs/adr/log.md`; the index lists every ADR.

- [ ] **Step 3: Prove the bundle is clean**

Run: `make adr-check`
Expected: `adr-check: clean` (no findings), exit 0. If a supersede edge warns, fix the back-edge; if index is out of sync, re-run `make adr-index`.

- [ ] **Step 4: Commit**

```bash
git add docs/adr/
git commit -m "docs(adr): backfill architectural decision corpus (~15 ADRs) + generated index/log"
```

---

### Task 7: Root `CLAUDE.md` instruction surface

**Files:**
- Create: `CLAUDE.md`

**Interfaces:** none (documentation). Must contain an "Architecture Decision Records" section stating the read/capture policy so the policy survives even if hooks are disabled.

- [ ] **Step 1: Write `CLAUDE.md`**

Model the structure on `getzep/graphiti`'s CLAUDE.md, adapted to this repo's real facts (verify each against the repo before writing):

```markdown
# CLAUDE.md

Guidance for agents working in this repository.

## What this is
An event-sourced transcript-mining system: EventStoreDB is the source of truth;
projections build a Neo4j read model; a lens engine extracts insights; Layer 5
exports an OKF bundle. FastAPI serves the read side; a Next.js frontend
(`frontend/`) renders workbench + gallery.

## Dev commands
- Lint / format: `make lint` (flake8), `make format` (black)
- Tests: `make test` · unit only: `make test-unit` · integration: `make test-integration` (env-gated)
- Run API / worker: `make run-api`, `make run-worker` · UI: `make ui-dev`
- ADRs: `make adr-check` (validate), `make adr-index` (regenerate index/log)

## Layout
`src/` — application code (ingestion, projections, lens, export, api).
`tools/adr/` — the ADR knowledge tooling. `docs/adr/` — the decision corpus.
`docs/superpowers/{specs,plans}/` — design specs and implementation plans.

## Architecture Decision Records (policy)
- **Before locking any architectural decision, consult `docs/adr/index.md`.** If
  your decision changes an existing one, write a new ADR and set `supersedes`
  (and the old ADR's `superseded_by`) — never silently override in prose.
- **After a brainstorm locks decisions, capture them:** `python -m tools.adr new "<title>"`, fill it in, set `source:` to the spec, then `make adr-index`.
- ADRs are durable (what/why); specs are disposable (how, this milestone). ADRs
  link out to specs — don't duplicate.
- `make adr-check` reports drift (never blocks): schema, id uniqueness,
  bidirectional supersede edges, specs that lock decisions without an ADR, and
  ADRs whose `source` changed after them.
```

- [ ] **Step 2: Verify the commands are real**

Run: `make help`
Expected: the targets referenced above (`lint`, `format`, `test-unit`, `run-api`, `adr-check`) all exist. Fix any mismatch in `CLAUDE.md`.

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: add root CLAUDE.md with ADR read/capture policy"
```

---

### Task 8: Wire the loop — Claude hooks + non-blocking pre-commit

**Files:**
- Modify: `.claude/settings.local.json` (add `hooks`)
- Create: `.githooks/pre-commit`
- Modify: `Makefile` (add `hooks-install` target)
- Test: `tests/adr/test_hooks_wiring.py`

**Interfaces:**
- Consumes: `python -m tools.adr {context|nudge|check}`
- Produces: `make hooks-install` (sets `git config core.hooksPath .githooks`)

- [ ] **Step 1: Write the failing test**

```python
# tests/adr/test_hooks_wiring.py
import json, os

def test_settings_registers_both_hooks():
    cfg = json.load(open(".claude/settings.local.json", encoding="utf-8"))
    hooks = cfg.get("hooks", {})
    ups = json.dumps(hooks.get("UserPromptSubmit", []))
    ptu = json.dumps(hooks.get("PostToolUse", []))
    assert "tools.adr context" in ups        # read side
    assert "tools.adr nudge" in ptu           # capture side

def test_precommit_hook_is_executable_and_nonblocking():
    path = ".githooks/pre-commit"
    assert os.path.exists(path)
    assert os.access(path, os.X_OK)
    body = open(path, encoding="utf-8").read()
    assert "adr" in body and "exit 0" in body   # never blocks the commit
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/adr/test_hooks_wiring.py -v`
Expected: FAIL — settings has no `hooks` key / `.githooks/pre-commit` missing

- [ ] **Step 3: Add the hooks**

Merge into `.claude/settings.local.json` (preserve existing keys; add `hooks`):

```json
{
  "hooks": {
    "UserPromptSubmit": [
      { "hooks": [ { "type": "command", "command": "python -m tools.adr context" } ] }
    ],
    "PostToolUse": [
      { "matcher": "Write", "hooks": [ { "type": "command", "command": "python -m tools.adr nudge" } ] }
    ]
  }
}
```

Create `.githooks/pre-commit` (non-blocking — reports, never fails):

```bash
#!/usr/bin/env bash
# Non-blocking ADR drift report. Never fails the commit.
python -m tools.adr check || true
exit 0
```

Add to `Makefile`:

```makefile
hooks-install:
	@git config core.hooksPath .githooks
	@echo "git hooks installed (core.hooksPath=.githooks)"
```

- [ ] **Step 4: Make the hook executable, install, verify**

```bash
chmod +x .githooks/pre-commit
make hooks-install
python -m pytest tests/adr/test_hooks_wiring.py -v
```
Expected: tests PASS (2 passed); `make hooks-install` prints confirmation.

- [ ] **Step 5: End-to-end smoke of the read + capture hooks**

```bash
echo '{"prompt":"let us brainstorm the design"}' | python -m tools.adr context
echo '{"tool_input":{"file_path":"docs/superpowers/specs/x-design.md"}}' | python -m tools.adr nudge
echo '{"prompt":"fix a typo"}' | python -m tools.adr context
```
Expected: first prints the ADR index + consult reminder; second prints the capture nudge; third prints nothing (non-architectural). All exit 0.

- [ ] **Step 6: Commit**

```bash
git add .claude/settings.local.json .githooks/pre-commit Makefile tests/adr/test_hooks_wiring.py
git commit -m "feat(adr): wire read/capture hooks + non-blocking pre-commit drift report"
```

---

## Final verification

- [ ] `make test-unit` — all `tests/adr/` tests pass.
- [ ] `make lint` — clean (add `tools` to the lint scope in the Makefile if flake8 should cover it).
- [ ] `make adr-check` — `adr-check: clean` on the real backfilled bundle.
- [ ] Manually confirm `docs/adr/index.md` lists every ADR and the m46 supersession edge is bidirectional.
