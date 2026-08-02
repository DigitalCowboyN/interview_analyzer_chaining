# Code-Documentation Domain (E) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A code-documentation domain: package + key-module nodes with an authored role classification, dependencies and I/O DERIVED from the import graph, a generated Mermaid pipeline map, and a non-blocking guard that catches undocumented new cross-package dependencies.

**Architecture:** New `tools/code/` package (reader → render → check → CLI). The reader parses the authored node files and attaches derived deps/I/O from `src/` imports; the renderer emits a catalog + a Mermaid dependency graph; the guard reconciles.

**Tech Stack:** Python 3 (stdlib `ast`/`re`/`glob`; PyYAML via `parse_front_matter`), pytest, Make.

## Global Constraints

- **Non-blocking, always.** Checks return `list[Finding]`; none raises; `make code-check` exits 0.
- **Deps + I/O are DERIVED from imports, never authored.** Nodes author only `role` + prose (+ `key_modules` list for packages). The generated map carries the derived edges.
- **Package + key modules only** (no per-`.py` nodes); static import edges only (no call-graph).
- **No capability / use-case links** this round (rounds 2–3).
- `docs/code/index.md` + `docs/code/pipeline.md` are generated (never hand-edited).
- `CodeUnit` / `Finding` local to `tools/code`. Tests in `tests/code/`.
- Run tests with `~/.pyenv/shims/python -m pytest <path> -p no:cacheprovider -q -o addopts=""`.

---

### Task 1: `reader.py` — dep graph, I/O, unit loading

**Files:**
- Create: `tools/code/__init__.py` (empty), `tools/code/reader.py`
- Test: `tests/code/__init__.py` (empty), `tests/code/test_reader.py`

**Interfaces:**
- Produces: `@dataclass CodeUnit(unit, role, key_modules, depends_on, io, description, path)`; `KEY_MODULES: list[str]`; `packages(root) -> list[str]`; `dep_edges(root) -> dict[str, list[str]]`; `io_of(unit, root) -> list[str]`; `load_units(root=".", code_dir="docs/code") -> list[CodeUnit]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/code/test_reader.py
from tools.code.reader import packages, dep_edges, io_of, load_units, CodeUnit

def test_packages_and_dep_edges(tmp_path):
    (tmp_path / "src" / "a").mkdir(parents=True)
    (tmp_path / "src" / "b").mkdir(parents=True)
    (tmp_path / "src" / "a" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "src" / "a" / "m.py").write_text("from src.b import x\n", encoding="utf-8")
    (tmp_path / "src" / "b" / "__init__.py").write_text("", encoding="utf-8")
    assert set(packages(str(tmp_path))) == {"a", "b"}
    edges = dep_edges(str(tmp_path))
    assert edges["a"] == ["b"] and edges.get("b", []) == []

def test_io_of_detects_signals(tmp_path):
    p = tmp_path / "src" / "x"; p.mkdir(parents=True)
    (p / "h.py").write_text("from src.events import E\nimport neo4j\n", encoding="utf-8")
    io = io_of("x", str(tmp_path))
    assert "ESDB" in io and "Neo4j" in io

def test_load_units_attaches_derived(tmp_path):
    (tmp_path / "src" / "ingestion").mkdir(parents=True)
    (tmp_path / "src" / "ingestion" / "m.py").write_text("from src.events import E\n", encoding="utf-8")
    (tmp_path / "src" / "events").mkdir(parents=True)
    (tmp_path / "src" / "events" / "__init__.py").write_text("", encoding="utf-8")
    cd = tmp_path / "docs" / "code"; cd.mkdir(parents=True)
    (cd / "ingestion.md").write_text(
        "---\ntype: CodeUnit\nunit: ingestion\nrole: pipeline-layer\nkey_modules: [m]\n---\nIngests.\n",
        encoding="utf-8")
    units = load_units(str(tmp_path))
    u = next(x for x in units if x.unit == "ingestion")
    assert u.role == "pipeline-layer" and "events" in u.depends_on and "ESDB" in u.io
    assert "Ingests" in u.description
```

- [ ] **Step 2: Run to verify fail**

Run: `~/.pyenv/shims/python -m pytest tests/code/test_reader.py -v`
Expected: FAIL — no module `tools.code`

- [ ] **Step 3: Implement**

```python
# tools/code/reader.py
from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List

from src.ingestion.front_matter import parse_front_matter

# curated load-bearing modules (dotted, relative to src/)
KEY_MODULES = [
    "ingestion.orchestrator", "ingestion.stitcher", "ingestion.speaker_inference",
    "enrichment.orchestrator", "enrichment.executor",
    "lens.engine", "export.reader", "export.renderer", "export.bundler",
    "ui.reader", "ask.reader", "ask.engine",
    "resolution.engine", "agents.agent_factory",
]

_IMPORT = re.compile(r"(?:from|import)\s+src\.(\w+)")


@dataclass
class CodeUnit:
    unit: str
    role: str = ""
    key_modules: List[str] = field(default_factory=list)
    depends_on: List[str] = field(default_factory=list)
    io: List[str] = field(default_factory=list)
    description: str = ""
    path: str = ""


def packages(root: str = ".") -> List[str]:
    src = os.path.join(root, "src")
    out = []
    if os.path.isdir(src):
        for name in sorted(os.listdir(src)):
            p = os.path.join(src, name)
            if os.path.isdir(p) and name != "__pycache__":
                out.append(name)
    return out


def _files_of(unit: str, root: str) -> List[str]:
    # package -> all its .py; dotted module -> that one file
    if "." in unit:
        return [os.path.join(root, "src", *unit.split(".")) + ".py"]
    return glob.glob(os.path.join(root, "src", unit, "**", "*.py"), recursive=True)


def dep_edges(root: str = ".") -> Dict[str, List[str]]:
    pkgs = packages(root)
    edges: Dict[str, set] = {p: set() for p in pkgs}
    for pkg in pkgs:
        for f in _files_of(pkg, root):
            try:
                text = open(f, encoding="utf-8", errors="ignore").read()
            except Exception:
                continue
            for m in _IMPORT.finditer(text):
                dep = m.group(1)
                if dep != pkg and dep in edges:
                    edges[pkg].add(dep)
    return {p: sorted(s) for p, s in edges.items()}


def io_of(unit: str, root: str = ".") -> List[str]:
    io = set()
    for f in _files_of(unit, root):
        try:
            t = open(f, encoding="utf-8", errors="ignore").read()
        except Exception:
            continue
        if re.search(r"from src\.events|import src\.events|EventStore|esdb", t, re.I):
            io.add("ESDB")
        if re.search(r"neo4j|GraphDatabase|from src\.persistence", t, re.I):
            io.add("Neo4j")
        if re.search(r"from src\.agents|AgentFactory|openai|anthropic", t, re.I):
            io.add("LLM")
        if re.search(r"FastAPI|APIRouter|uvicorn", t):
            io.add("HTTP")
        if re.search(r"open\(|Path\(|\.read_text|glob\.", t):
            io.add("files")
    return sorted(io)


def load_units(root: str = ".", code_dir: str = "docs/code") -> List[CodeUnit]:
    edges = dep_edges(root)
    units: List[CodeUnit] = []
    for path in sorted(glob.glob(os.path.join(root, code_dir, "*.md"))):
        if os.path.basename(path) in ("index.md", "pipeline.md"):
            continue
        text = open(path, encoding="utf-8").read()
        fm, offset = parse_front_matter(text)
        if not fm or "unit" not in fm:
            continue
        unit = str(fm["unit"])
        units.append(CodeUnit(
            unit=unit, role=str(fm.get("role", "")),
            key_modules=list(fm.get("key_modules") or []),
            depends_on=edges.get(unit, []) or dep_edges_for_module(unit, root),
            io=io_of(unit, root), description=text[offset:], path=path,
        ))
    return units


def dep_edges_for_module(unit: str, root: str) -> List[str]:
    if "." not in unit:
        return []
    deps = set()
    for f in _files_of(unit, root):
        try:
            t = open(f, encoding="utf-8", errors="ignore").read()
        except Exception:
            continue
        pkg = unit.split(".")[0]
        for m in _IMPORT.finditer(t):
            if m.group(1) != pkg:
                deps.add(m.group(1))
    return sorted(deps)
```

- [ ] **Step 4: Run to verify pass**

Run: `~/.pyenv/shims/python -m pytest tests/code/test_reader.py -v` → PASS (3)

- [ ] **Step 5: Commit**

```bash
git add tools/code/__init__.py tools/code/reader.py tests/code/__init__.py tests/code/test_reader.py
git commit -m "feat(code): reader — package/module dep graph + I/O derivation + unit loading"
```

---

### Task 2: `render.py` — catalog + Mermaid pipeline map

**Files:** Create `tools/code/render.py`; Test `tests/code/test_render.py`

**Interfaces:** `render_index(units) -> str`; `render_pipeline(units) -> str` (Mermaid)

- [ ] **Step 1: Write the failing test**

```python
# tests/code/test_render.py
from tools.code.reader import CodeUnit
from tools.code.render import render_index, render_pipeline

UNITS = [
    CodeUnit("ingestion", "pipeline-layer", ["orchestrator"], ["events", "agents"], ["ESDB", "LLM"], "Ingests.", "p"),
    CodeUnit("events", "infrastructure", [], [], ["ESDB"], "Event store.", "p"),
]

def test_render_index_groups_by_role():
    out = render_index(UNITS)
    assert "## pipeline-layer" in out and "ingestion" in out
    assert "ESDB" in out and "events, agents" in out

def test_render_pipeline_is_mermaid():
    out = render_pipeline(UNITS)
    assert out.strip().startswith("```mermaid") or "graph LR" in out
    assert "ingestion --> events" in out and "ingestion --> agents" in out
```

- [ ] **Step 2: Run to verify fail** — no module `tools.code.render`

- [ ] **Step 3: Implement**

```python
# tools/code/render.py
from __future__ import annotations

from typing import List

from tools.code.reader import CodeUnit


def render_index(units: List[CodeUnit]) -> str:
    by_role: dict = {}
    for u in units:
        by_role.setdefault(u.role or "(unclassified)", []).append(u)
    lines = ["# Code map", "", "See `pipeline.md` for the dependency graph.", ""]
    for role in sorted(by_role):
        lines.append(f"## {role}")
        lines.append("")
        lines.append("| unit | io | depends_on |")
        lines.append("| --- | --- | --- |")
        for u in sorted(by_role[role], key=lambda u: u.unit):
            lines.append(f"| {u.unit} | {', '.join(u.io)} | {', '.join(u.depends_on)} |")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_pipeline(units: List[CodeUnit]) -> str:
    lines = ["# Dependency / pipeline map", "", "```mermaid", "graph LR"]
    for u in sorted(units, key=lambda u: u.unit):
        if not u.depends_on:
            lines.append(f"    {u.unit}")
        for dep in u.depends_on:
            lines.append(f"    {u.unit} --> {dep}")
    lines.append("```")
    return "\n".join(lines) + "\n"
```

- [ ] **Step 4: Run to verify pass** — PASS

- [ ] **Step 5: Commit**

```bash
git add tools/code/render.py tests/code/test_render.py
git commit -m "feat(code): catalog + Mermaid pipeline-map renderers"
```

---

### Task 3: `check.py` — the guard

**Files:** Create `tools/code/check.py`; Test `tests/code/test_check.py`

**Interfaces:** `@dataclass Finding`; `check_coverage(packages, units)`, `check_classification(units)`, `check_map_in_sync(index_path, pipeline_path, units)`, `check_stale(units, real_units)`, `run_all(root=".")`

- [ ] **Step 1: Write the failing test**

```python
# tests/code/test_check.py
from tools.code.reader import CodeUnit
from tools.code.check import check_coverage, check_classification, check_stale, Finding

def test_coverage_flags_undocumented_package():
    msgs = " ".join(f.message for f in check_coverage(["resolution", "events"], [CodeUnit("events", "infrastructure")]))
    assert "resolution" in msgs

def test_classification_flags_missing_role():
    msgs = " ".join(f.message for f in check_classification([CodeUnit("x", "")]))
    assert "x" in msgs

def test_stale_flags_unit_not_in_code():
    msgs = " ".join(f.message for f in check_stale([CodeUnit("gone", "surface")], ["events", "api"]))
    assert "gone" in msgs
```

- [ ] **Step 2: Run to verify fail** — no module `tools.code.check`

- [ ] **Step 3: Implement**

```python
# tools/code/check.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List

from tools.code.reader import CodeUnit, KEY_MODULES, load_units, packages
from tools.code.render import render_index, render_pipeline


@dataclass
class Finding:
    message: str


def check_coverage(pkgs: List[str], units: List[CodeUnit]) -> List[Finding]:
    have = {u.unit for u in units}
    return [Finding(f"code: package src/{p} has no doc node") for p in pkgs if p not in have]


def check_classification(units: List[CodeUnit]) -> List[Finding]:
    return [Finding(f"code: unit {u.unit} has no role classification") for u in units if not u.role]


def check_map_in_sync(index_path: str, pipeline_path: str, units: List[CodeUnit]) -> List[Finding]:
    findings: List[Finding] = []
    for path, render in ((index_path, render_index), (pipeline_path, render_pipeline)):
        want = render(units)
        have = open(path, encoding="utf-8").read() if os.path.exists(path) else ""
        if want != have:
            findings.append(Finding(f"code: {os.path.basename(path)} out of sync — run make code-index (new dependency?)"))
    return findings


def check_stale(units: List[CodeUnit], real_units: List[str]) -> List[Finding]:
    real = set(real_units)
    return [Finding(f"code: doc node {u.unit} no longer exists in src") for u in units if u.unit not in real]


def run_all(root: str = ".") -> List[Finding]:
    pkgs = packages(root)
    units = load_units(root)
    real = pkgs + KEY_MODULES
    findings: List[Finding] = []
    findings += check_coverage(pkgs, units)
    findings += check_classification(units)
    findings += check_map_in_sync(os.path.join(root, "docs/code/index.md"),
                                  os.path.join(root, "docs/code/pipeline.md"), units)
    findings += check_stale(units, real)
    return findings
```

- [ ] **Step 4: Run to verify pass** — PASS (3)

- [ ] **Step 5: Commit**

```bash
git add tools/code/check.py tests/code/test_check.py
git commit -m "feat(code): guard — coverage, classification, map-in-sync, stale"
```

---

### Task 4: CLI + Makefile

**Files:** Create `tools/code/__main__.py`; Modify `Makefile`; Test `tests/code/test_cli.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/code/test_cli.py
import subprocess, sys

def test_cli_check_exits_zero():
    proc = subprocess.run([sys.executable, "-m", "tools.code", "check"], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "code-check" in proc.stdout
```

- [ ] **Step 2: Run to verify fail** — no `tools.code.__main__`

- [ ] **Step 3: Implement** — mirror the other tools' `__main__.py`: subcommands `index` (writes both `docs/code/index.md` via `render_index(load_units())` and `docs/code/pipeline.md` via `render_pipeline(load_units())`, `os.makedirs` the dir) and `check` (prints `run_all()` findings or `code-check: clean`, `return 0`). Add Makefile targets near `graphq-check`:

```makefile
.PHONY: code-index
code-index: ## Regenerate docs/code/index.md + pipeline.md (code map)
	@$(PYTHON) -m tools.code index

.PHONY: code-check
code-check: ## Reconcile the code map vs the import graph (non-blocking)
	@$(PYTHON) -m tools.code check
```

- [ ] **Step 4: Run test + smoke** — test PASS; `~/.pyenv/shims/python -m tools.code check` exit 0 (findings expected until Task 5).

- [ ] **Step 5: Commit**

```bash
git add tools/code/__main__.py Makefile tests/code/test_cli.py
git commit -m "feat(code): CLI (index/check) + make targets"
```

---

### Task 5: Backfill the code map

**Files:** Create `docs/code/<pkg>.md` × 16 + `docs/code/<key-module>.md` × ~14 + generated `docs/code/index.md`, `docs/code/pipeline.md`

- [ ] **Step 1: List what needs nodes**

```bash
~/.pyenv/shims/python -c "from tools.code.reader import packages, KEY_MODULES, dep_edges, io_of; \
[print('pkg ', p, 'deps=', dep_edges('.').get(p), 'io=', io_of(p,'.')) for p in packages('.')]; \
[print('mod ', m) for m in KEY_MODULES]"
```

- [ ] **Step 2: Author one node per package + key module**

Frontmatter (author `role` + `key_modules` + a terse description; deps/io are auto-derived, do NOT put them in frontmatter):
```markdown
---
type: CodeUnit
unit: ingestion
role: pipeline-layer
key_modules: [orchestrator, stitcher, speaker_inference, front_matter]
---
Layer 1: ingest transcripts, capture front matter, map speakers, stitch utterances.
```
Roles to assign (judge from `data-flow.md` / `system-overview.md` + the module docstrings):
- **pipeline-layer:** ingestion, enrichment, lens, export, projections, resolution
- **surface:** api, ui, ask
- **infrastructure:** events, persistence, commands, io, utils
- **model:** models
- **agent:** agents
Key modules get their own nodes (`unit: export.reader`, role = its package's role or a finer one). Keep descriptions terse; mine the module docstrings for accuracy.

- [ ] **Step 3: Generate + reconcile**

```bash
make code-index          # writes docs/code/index.md + pipeline.md
~/.pyenv/shims/python -m tools.code check   # iterate until clean
```
`clean` = every package covered, every node classified, index + pipeline in sync, no stale nodes.

- [ ] **Step 4: Commit**

```bash
git add docs/code/
git commit -m "docs(code): backfill code map (16 packages + key modules) + generated index & Mermaid pipeline"
```

---

## Final verification

- [ ] `~/.pyenv/shims/python -m pytest tests/code/ -v` — all green.
- [ ] `make code-check` — clean on the real backfilled map.
- [ ] `make code-index` then `git status` — `docs/code/index.md` + `pipeline.md` regenerate identically.
- [ ] Open `docs/code/pipeline.md` — the Mermaid graph renders the real package dependency DAG.
- [ ] `make cli-index` — regenerate the CLI catalog to include `code-*` (then `cli-check` clean).
