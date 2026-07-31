# Glossary / Taxonomy Domain Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A `docs/glossary/` bundle of the project's vocabulary (7 enums + 7 analysis dimensions), each with a definition and allowed values, plus a non-blocking guard that reconciles it against the code (AST-read) and holds the canonical dimension values for F.

**Architecture:** New `tools/glossary/` package (reader → model → check → render → scaffold → CLI). The reader parses source with `ast` (no app import). Enum values are code-checked; dimension values are authored.

**Tech Stack:** Python 3 (stdlib `ast`; PyYAML via the existing `parse_front_matter`), pytest, Make.

## Global Constraints

- **Non-blocking, always.** Checks return `list[Finding]`; none raises; `make glossary-check` / the CLI exit 0.
- **AST-based, no app import** — the reader parses source text with `ast`; it never imports `src.*` runtime code.
- **Asymmetric values:** `kind: enum` term `values` are checked against the code enum; `kind: dimension` term `values` are authored (not code-checked).
- **v1 = enums + dimensions.** No node labels / rel types / lens node_types.
- `docs/glossary/index.md` is generated (reserved), never hand-edited.
- `CodeTerm` / `Term` / `Finding` are local to `tools/glossary`. Tooling in `tools/glossary/`; tests in `tests/glossary/`.
- Run tests with `~/.pyenv/shims/python -m pytest <path> -v`.

---

### Task 1: `reader.py` — AST extraction of enums + dimensions

**Files:**
- Create: `tools/glossary/__init__.py` (empty), `tools/glossary/reader.py`
- Test: `tests/glossary/__init__.py` (empty), `tests/glossary/test_reader.py`

**Interfaces:**
- Produces:
  - `@dataclass CodeTerm(name: str, kind: str, source: str, values: list[str])`
  - `code_enums(root=".", subdirs=("src",)) -> dict[str, CodeTerm]`
  - `code_dimensions(root=".", model_path="src/models/analysis_result.py", class_name="AnalysisResult") -> dict[str, CodeTerm]`

- [ ] **Step 1: Write the failing test**

```python
# tests/glossary/test_reader.py
from tools.glossary.reader import code_enums, code_dimensions, CodeTerm

def test_code_enums_extracts_members(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "e.py").write_text(
        "from enum import Enum\n"
        "class Color(str, Enum):\n    RED = 'red'\n    BLUE = 'blue'\n\n"
        "class Plain:\n    x = 1\n", encoding="utf-8")
    enums = code_enums(str(tmp_path))
    assert "Color" in enums and enums["Color"].kind == "enum"
    assert enums["Color"].values == ["RED", "BLUE"]
    assert enums["Color"].source == "src/e.py"
    assert "Plain" not in enums          # non-enum ignored

def test_code_dimensions_reads_annotated_fields(tmp_path):
    p = tmp_path / "src" / "models"; p.mkdir(parents=True)
    (p / "analysis_result.py").write_text(
        "from pydantic import BaseModel\n"
        "class AnalysisResult(BaseModel):\n"
        "    '''Attributes: not a field.'''\n"
        "    function_type: str\n    purpose: str\n", encoding="utf-8")
    dims = code_dimensions(str(tmp_path))
    assert set(dims) == {"function_type", "purpose"}
    assert dims["function_type"].kind == "dimension"
    assert "Attributes" not in dims       # docstring not a field
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.pyenv/shims/python -m pytest tests/glossary/test_reader.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tools.glossary'`

- [ ] **Step 3: Implement**

```python
# tools/glossary/reader.py
from __future__ import annotations

import ast
import os
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class CodeTerm:
    name: str
    kind: str            # "enum" | "dimension"
    source: str          # repo-relative path
    values: List[str] = field(default_factory=list)


def _is_enum(classdef: ast.ClassDef) -> bool:
    for b in classdef.bases:
        name = b.id if isinstance(b, ast.Name) else getattr(b, "attr", "")
        if name == "Enum" or name.endswith("Enum"):
            return True
    return False


def code_enums(root: str = ".", subdirs=("src",)) -> Dict[str, CodeTerm]:
    out: Dict[str, CodeTerm] = {}
    for base in subdirs:
        start = os.path.join(root, base)
        if not os.path.isdir(start):
            continue
        for dirpath, _dirs, files in os.walk(start):
            for fn in files:
                if not fn.endswith(".py"):
                    continue
                full = os.path.join(dirpath, fn)
                try:
                    tree = ast.parse(open(full, encoding="utf-8").read())
                except Exception:
                    continue
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef) and _is_enum(node):
                        members = [
                            t.targets[0].id for t in node.body
                            if isinstance(t, ast.Assign) and len(t.targets) == 1
                            and isinstance(t.targets[0], ast.Name)
                        ]
                        rel = os.path.relpath(full, root).replace(os.sep, "/")
                        out[node.name] = CodeTerm(node.name, "enum", rel, members)
    return out


def code_dimensions(root: str = ".",
                    model_path: str = "src/models/analysis_result.py",
                    class_name: str = "AnalysisResult") -> Dict[str, CodeTerm]:
    out: Dict[str, CodeTerm] = {}
    p = os.path.join(root, model_path)
    if not os.path.exists(p):
        return out
    try:
        tree = ast.parse(open(p, encoding="utf-8").read())
    except Exception:
        return out
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for b in node.body:
                if isinstance(b, ast.AnnAssign) and isinstance(b.target, ast.Name):
                    out[b.target.id] = CodeTerm(b.target.id, "dimension", model_path, [])
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `~/.pyenv/shims/python -m pytest tests/glossary/test_reader.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add tools/glossary/__init__.py tools/glossary/reader.py tests/glossary/__init__.py tests/glossary/test_reader.py
git commit -m "feat(glossary): AST reader for code enums + analysis dimensions"
```

---

### Task 2: `model.py` — glossary term files

**Files:**
- Create: `tools/glossary/model.py`
- Test: `tests/glossary/test_model.py`

**Interfaces:**
- Consumes: `src.ingestion.front_matter.parse_front_matter`
- Produces: `@dataclass Term(term, kind, source, values, definition, path)`; `parse_term(text, path=None) -> Term`; `RESERVED = {"index.md"}`; `load_glossary(dir) -> list[Term]`

- [ ] **Step 1: Write the failing test**

```python
# tests/glossary/test_model.py
from tools.glossary.model import parse_term, load_glossary, Term

TERM = ("---\ntype: Term\nterm: ActorType\nkind: enum\n"
        "source: src/events/envelope.py\nvalues: [HUMAN, SYSTEM, AI]\n---\nWho caused an event.\n")

def test_parse_term(tmp_path):
    t = parse_term(TERM, path="docs/glossary/actortype.md")
    assert t.term == "ActorType" and t.kind == "enum"
    assert t.values == ["HUMAN", "SYSTEM", "AI"]
    assert "Who caused" in t.definition

def test_load_glossary_skips_index(tmp_path):
    (tmp_path / "index.md").write_text("# generated\n", encoding="utf-8")
    (tmp_path / "actortype.md").write_text(TERM, encoding="utf-8")
    terms = load_glossary(str(tmp_path))
    assert [t.term for t in terms] == ["ActorType"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.pyenv/shims/python -m pytest tests/glossary/test_model.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tools.glossary.model'`

- [ ] **Step 3: Implement**

```python
# tools/glossary/model.py
from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field
from typing import List, Optional

from src.ingestion.front_matter import parse_front_matter

RESERVED = {"index.md"}


@dataclass
class Term:
    term: str
    kind: str
    source: Optional[str]
    values: List[str] = field(default_factory=list)
    definition: str = ""
    path: Optional[str] = None


def parse_term(text: str, path: Optional[str] = None) -> Term:
    fm, offset = parse_front_matter(text)
    if fm is None:
        raise ValueError(f"{path or '<text>'}: missing front matter")
    return Term(
        term=str(fm["term"]),
        kind=str(fm["kind"]),
        source=fm.get("source"),
        values=[str(v) for v in (fm.get("values") or [])],
        definition=text[offset:],
        path=path,
    )


def load_glossary(glossary_dir: str) -> List[Term]:
    terms: List[Term] = []
    for p in sorted(glob.glob(os.path.join(glossary_dir, "*.md"))):
        if os.path.basename(p) in RESERVED:
            continue
        try:
            terms.append(parse_term(open(p, encoding="utf-8").read(), path=p))
        except Exception:
            continue  # malformed tolerated (best-effort)
    return terms
```

- [ ] **Step 4: Run test to verify it passes**

Run: `~/.pyenv/shims/python -m pytest tests/glossary/test_model.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add tools/glossary/model.py tests/glossary/test_model.py
git commit -m "feat(glossary): Term model + glossary loader"
```

---

### Task 3: `check.py` — the reconciliation guard

**Files:**
- Create: `tools/glossary/check.py`
- Test: `tests/glossary/test_check.py`

**Interfaces:**
- Consumes: `tools.glossary.reader` (`code_enums`, `code_dimensions`, `CodeTerm`), `tools.glossary.model` (`Term`, `load_glossary`), `tools.glossary.render.render_index`
- Produces: `@dataclass Finding`; `check_coverage`, `check_enum_values`, `check_stale_source`, `check_index_in_sync`, `run_all(root=".") -> list[Finding]`

- [ ] **Step 1: Write the failing test**

```python
# tests/glossary/test_check.py
from tools.glossary.reader import CodeTerm
from tools.glossary.model import Term
from tools.glossary.check import (
    check_coverage, check_enum_values, check_stale_source, Finding,
)

def _code(**kw): return {kw["name"]: CodeTerm(kw["name"], kw["kind"], kw.get("source", "src/x.py"), kw.get("values", []))}

def test_coverage_flags_uncovered_code_term():
    code = _code(name="ActorType", kind="enum", values=["HUMAN"])
    msgs = " ".join(f.message for f in check_coverage(code, []))
    assert "ActorType" in msgs and "no glossary term" in msgs

def test_enum_values_reconciled_only_for_enums():
    code = {**_code(name="ActorType", kind="enum", values=["HUMAN", "AI"])}
    term = Term("ActorType", "enum", "src/x.py", ["HUMAN"], "", "p")   # missing AI
    msgs = " ".join(f.message for f in check_enum_values(code, [term]))
    assert "ActorType" in msgs and "AI" in msgs
    # dimension values are NOT reconciled
    dcode = {**_code(name="purpose", kind="dimension", values=[])}
    dterm = Term("purpose", "dimension", "src/m.py", ["statement", "question"], "", "p")
    assert check_enum_values(dcode, [dterm]) == []

def test_stale_source_flags_term_not_in_code():
    term = Term("GoneEnum", "enum", "src/x.py", [], "", "p")
    msgs = " ".join(f.message for f in check_stale_source({}, [term]))
    assert "GoneEnum" in msgs
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.pyenv/shims/python -m pytest tests/glossary/test_check.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tools.glossary.check'`

- [ ] **Step 3: Implement**

```python
# tools/glossary/check.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List

from tools.glossary.reader import CodeTerm, code_dimensions, code_enums
from tools.glossary.model import Term, load_glossary
from tools.glossary.render import render_index


@dataclass
class Finding:
    message: str


def check_coverage(code: Dict[str, CodeTerm], terms: List[Term]) -> List[Finding]:
    have = {t.term for t in terms}
    findings: List[Finding] = []
    for name in sorted(code):
        if name not in have:
            ct = code[name]
            findings.append(Finding(f"code defines {ct.kind} {name} ({ct.source}) with no glossary term"))
    return findings


def check_enum_values(code: Dict[str, CodeTerm], terms: List[Term]) -> List[Finding]:
    by_name = {t.term: t for t in terms}
    findings: List[Finding] = []
    for name, ct in code.items():
        if ct.kind != "enum":
            continue
        gt = by_name.get(name)
        if gt is None:
            continue  # coverage handles missing
        if set(gt.values) != set(ct.values):
            missing = sorted(set(ct.values) - set(gt.values))
            extra = sorted(set(gt.values) - set(ct.values))
            findings.append(Finding(f"glossary term {name} values differ from code (missing: {missing}, extra: {extra})"))
    return findings


def check_stale_source(code: Dict[str, CodeTerm], terms: List[Term]) -> List[Finding]:
    findings: List[Finding] = []
    for t in terms:
        if t.term not in code:
            findings.append(Finding(f"glossary term {t.term}: no longer defined in code (source {t.source})"))
    return findings


def check_index_in_sync(index_path: str, terms: List[Term]) -> List[Finding]:
    want = render_index(terms)
    have = open(index_path, encoding="utf-8").read() if os.path.exists(index_path) else ""
    if want != have:
        return [Finding("docs/glossary/index.md out of sync — run make glossary-index")]
    return []


def run_all(root: str = ".") -> List[Finding]:
    code = {**code_enums(root), **code_dimensions(root)}
    terms = load_glossary(os.path.join(root, "docs/glossary"))
    findings: List[Finding] = []
    findings += check_coverage(code, terms)
    findings += check_enum_values(code, terms)
    findings += check_stale_source(code, terms)
    findings += check_index_in_sync(os.path.join(root, "docs/glossary/index.md"), terms)
    return findings
```

- [ ] **Step 4: Run test to verify it passes**

Run: `~/.pyenv/shims/python -m pytest tests/glossary/test_check.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tools.glossary.render'` (render is Task 4)

> The check imports `render_index`; create a minimal `render.py` now so the module
> imports, then flesh it out in Task 4. Add this `tools/glossary/render.py`:

```python
# tools/glossary/render.py
from __future__ import annotations

from typing import List

from tools.glossary.model import Term


def render_index(terms: List[Term]) -> str:
    lines = ["# Glossary", "", "| term | kind | source |", "| --- | --- | --- |"]
    for t in sorted(terms, key=lambda t: (t.kind, t.term)):
        lines.append(f"| {t.term} | {t.kind} | {t.source} |")
    return "\n".join(lines) + "\n"
```

Re-run: `~/.pyenv/shims/python -m pytest tests/glossary/test_check.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add tools/glossary/check.py tools/glossary/render.py tests/glossary/test_check.py
git commit -m "feat(glossary): coverage, enum-value, stale-source, index-sync checks"
```

---

### Task 4: `render` test + `scaffold.py` + CLI + Makefile

**Files:**
- Create: `tools/glossary/scaffold.py`, `tools/glossary/__main__.py`
- Modify: `Makefile` (add `glossary-index`, `glossary-check`)
- Test: `tests/glossary/test_render.py`, `tests/glossary/test_cli.py`

**Interfaces:**
- Produces: `scaffold.new_term(name, kind, root=".") -> str`; `python -m tools.glossary {index|check|scaffold}`

- [ ] **Step 1: Write the failing test**

```python
# tests/glossary/test_render.py
from tools.glossary.model import Term
from tools.glossary.render import render_index

def test_render_index():
    out = render_index([Term("ActorType", "enum", "src/events/envelope.py", ["HUMAN"], "", "p")])
    assert "| ActorType | enum | src/events/envelope.py |" in out
```

```python
# tests/glossary/test_cli.py
import subprocess, sys

def test_cli_check_exits_zero():
    proc = subprocess.run([sys.executable, "-m", "tools.glossary", "check"], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "glossary-check" in proc.stdout

def test_scaffold_enum_prefills_values(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "e.py").write_text(
        "from enum import Enum\nclass Color(str, Enum):\n    RED = 'r'\n    BLUE = 'b'\n", encoding="utf-8")
    from tools.glossary.scaffold import new_term
    path = new_term("Color", "enum", root=str(tmp_path))
    body = open(path, encoding="utf-8").read()
    assert "term: Color" in body and "values: [RED, BLUE]" in body
    assert "source: src/e.py" in body
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.pyenv/shims/python -m pytest tests/glossary/test_render.py tests/glossary/test_cli.py -v`
Expected: FAIL — `No module named tools.glossary.scaffold` / `tools.glossary.__main__`

- [ ] **Step 3: Implement**

```python
# tools/glossary/scaffold.py
from __future__ import annotations

import os

from tools.glossary.reader import code_dimensions, code_enums


def new_term(name: str, kind: str, root: str = ".") -> str:
    ct = (code_enums(root) if kind == "enum" else code_dimensions(root)).get(name)
    values = ct.values if ct else []
    source = ct.source if ct else ""
    slug = name.lower().replace("_", "-")
    path = os.path.join(root, "docs/glossary", f"{slug}.md")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    vals = "[" + ", ".join(values) + "]"
    content = (f"---\ntype: Term\nterm: {name}\nkind: {kind}\nsource: {source}\n"
               f"values: {vals}\n---\nTODO: define {name}.\n")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)
    return path
```

```python
# tools/glossary/__main__.py
from __future__ import annotations

import argparse
import os
import sys

from tools.glossary.check import run_all
from tools.glossary.model import load_glossary
from tools.glossary.render import render_index
from tools.glossary.scaffold import new_term

GLOSSARY = "docs/glossary"


def cmd_index(args) -> int:
    os.makedirs(GLOSSARY, exist_ok=True)
    with open(os.path.join(GLOSSARY, "index.md"), "w", encoding="utf-8") as fh:
        fh.write(render_index(load_glossary(GLOSSARY)))
    print(f"wrote {GLOSSARY}/index.md")
    return 0


def cmd_check(args) -> int:
    findings = run_all()
    if findings:
        print(f"glossary-check: {len(findings)} warning(s):")
        for f in findings:
            print(f"  - {f.message}")
    else:
        print("glossary-check: clean")
    return 0  # NON-BLOCKING


def cmd_scaffold(args) -> int:
    path = new_term(args.name, args.kind)
    print(f"created {path} — fill in the definition" + (" and values" if args.kind == "dimension" else ""))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="tools.glossary")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("index")
    sub.add_parser("check")
    p_s = sub.add_parser("scaffold"); p_s.add_argument("name"); p_s.add_argument("kind", choices=["enum", "dimension"])
    args = parser.parse_args(argv)
    return {"index": cmd_index, "check": cmd_check, "scaffold": cmd_scaffold}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
```

Add to `Makefile` (near `api-check`), self-documented:

```makefile
.PHONY: glossary-index
glossary-index: ## Regenerate docs/glossary/index.md
	@$(PYTHON) -m tools.glossary index

.PHONY: glossary-check
glossary-check: ## Reconcile the glossary against code vocabulary (non-blocking)
	@$(PYTHON) -m tools.glossary check
```

- [ ] **Step 4: Run tests + smoke**

Run: `~/.pyenv/shims/python -m pytest tests/glossary/ -v`
Expected: PASS
Run: `~/.pyenv/shims/python -m tools.glossary check`
Expected: exit 0; findings expected (14 uncovered code terms + index out of sync) until Task 5

- [ ] **Step 5: Commit**

```bash
git add tools/glossary/scaffold.py tools/glossary/__main__.py Makefile tests/glossary/test_render.py tests/glossary/test_cli.py
git commit -m "feat(glossary): scaffold + CLI (index/check/scaffold) + make targets"
```

---

### Task 5: Backfill the glossary + generate index

Content task: author the ~14 term files and get `glossary-check` clean.

**Files:**
- Create: `docs/glossary/*.md` (14 term files) + generated `docs/glossary/index.md`

- [ ] **Step 1: Scaffold + author each term**

For each of the 7 enums, run `python -m tools.glossary scaffold <Name> enum` (pre-fills
`values` + `source` from code) and write a one-line definition:
`TranscriptFormat`, `EditorType`, `TagType`, `SentenceStatus`, `InterviewStatus`,
`ActorType`, `AggregateType`.

For each of the 7 dimensions, run `python -m tools.glossary scaffold <name> dimension`
and author **both** the definition and the allowed `values` — read the values from the
prompts/agent code (search `src/agents/`, prompt templates, and the extractor prompts
for the enumerated options each dimension accepts, e.g. `purpose: statement/question/…`):
`function_type`, `structure_type`, `purpose`, `topic_level_1`, `topic_level_3`,
`overall_keywords`, `domain_keywords`. (Keyword/topic dimensions are open-ended — say so
in the definition and leave `values: []`.)

- [ ] **Step 2: Generate the index + reconcile**

```bash
make glossary-index
make glossary-check       # iterate until clean
```
`clean` means: every code enum + dimension has a term; every enum term's `values`
equal the code enum's members; no stale terms; index in sync.

- [ ] **Step 3: Commit**

```bash
git add docs/glossary/
git commit -m "docs(glossary): backfill 7 enums + 7 dimensions with definitions + values"
```

---

## Final verification

- [ ] `~/.pyenv/shims/python -m pytest tests/glossary/ -v` — all green.
- [ ] `make glossary-check` — clean on the real backfilled glossary.
- [ ] `make glossary-index` then `git status` — `docs/glossary/index.md` regenerates identically.
- [ ] `make cli-index` — regenerate the CLI catalog to include the new `glossary-*` targets (then `make cli-check` clean).
