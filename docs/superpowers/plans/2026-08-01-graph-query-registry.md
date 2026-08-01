# Graph-Query Registry + Glossary Graph-Vocabulary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Grow the glossary to hold the graph vocabulary (labels/rels/props from the projection write side), then stand up a graph-query registry (`tools/graphq/`) that catalogs the read-query bundles and guards query→schema drift, query→consumers output-contract, and scope/shape.

**Architecture:** G1 extends `tools/glossary` (a `graph_vocabulary` extractor + check wiring) + ~64 authored terms. G2/G3 are a new `tools/graphq/` package (AST-finds query functions, regex-parses their Cypher, reads `graphq:` docstring markers, derives consumers) with a non-blocking guard.

**Tech Stack:** Python 3 stdlib (`ast`, `re`; PyYAML via existing helpers), pytest, Make.

## Global Constraints

- **Non-blocking, always.** Checks return `list[Finding]`; none raises; `make {glossary,graphq}-check` exit 0.
- **Write = schema truth.** Graph vocabulary is extracted from `src/projections/`. Read-query schema-drift reconciles against that extraction directly.
- **Regex Cypher extraction.** Labels (`(:Label)` CamelCase) and rel types (`[:REL_TYPE]` UPPER) are reliable; property parse is best-effort — a missed prop is silently *not* checked, never a false drift.
- **Read bundles only.** Graphq catalogs `src/**/reader.py` + inline `src/api/routers/*.py` queries. Projection handlers (write side) are NOT catalogued — they are the schema source.
- **Living glossary:** graph-* `kind`s are free-form; terse definitions are fine.
- `CodeTerm`/`QueryEntry`/`Finding` local to their packages. Tests in `tests/glossary/`, `tests/graphq/`.
- Run tests with `~/.pyenv/shims/python -m pytest <path> -p no:cacheprovider -q -o addopts=""`.

---

### Task 1: Glossary — `graph_vocabulary` extractor + check wiring

**Files:**
- Modify: `tools/glossary/reader.py` (add `graph_vocabulary`), `tools/glossary/check.py` (extend coverage/stale for graph-* kinds)
- Test: `tests/glossary/test_reader.py`, `tests/glossary/test_check.py` (append)

**Interfaces:**
- Produces: `graph_vocabulary(root=".", subdir="src/projections") -> dict[str, CodeTerm]` (kinds `graph-label`/`rel-type`/`graph-property`).

- [ ] **Step 1: Write the failing tests** (append)

```python
# tests/glossary/test_reader.py (append)
def test_graph_vocabulary_extracts_labels_rels_props(tmp_path):
    from tools.glossary.reader import graph_vocabulary
    p = tmp_path / "src" / "projections"; p.mkdir(parents=True)
    (p / "h.py").write_text(
        'q = "MERGE (c:Claim {claim_id: $id}) SET c.confidence = 0.9 "\n'
        '    "MERGE (s:Speaker)-[:MADE_BY]->(c)"\n', encoding="utf-8")
    gv = graph_vocabulary(str(tmp_path))
    assert gv["Claim"].kind == "graph-label" and gv["Speaker"].kind == "graph-label"
    assert gv["MADE_BY"].kind == "rel-type"
    assert "claim_id" in gv and gv["claim_id"].kind == "graph-property"
    assert "confidence" in gv
```

```python
# tests/glossary/test_check.py (append)
def test_coverage_and_stale_cover_graph_kinds():
    from tools.glossary.reader import CodeTerm
    from tools.glossary.model import Term
    from tools.glossary.check import check_coverage, check_stale_source
    code = {"Claim": CodeTerm("Claim", "graph-label", "src/projections/h.py", [])}
    # code defines a graph-label with no glossary term -> coverage finding
    assert check_coverage(code, [])
    # a graph-label glossary term not in code -> stale
    t = Term("GoneLabel", "graph-label", "src/projections/h.py", [], "", "p")
    assert check_stale_source({}, [t])
    # covered -> no findings
    ct = Term("Claim", "graph-label", "src/projections/h.py", [], "", "p")
    assert check_coverage(code, [ct]) == [] and check_stale_source(code, [ct]) == []
```

- [ ] **Step 2: Run to verify fail**

Run: `~/.pyenv/shims/python -m pytest tests/glossary/test_reader.py::test_graph_vocabulary_extracts_labels_rels_props tests/glossary/test_check.py::test_coverage_and_stale_cover_graph_kinds -v`
Expected: FAIL (`graph_vocabulary` missing / graph kinds not stale-checked)

- [ ] **Step 3: Implement**

In `tools/glossary/reader.py`:

```python
_GV_LABEL = re.compile(r"[(\[]\s*\w*:([A-Z][A-Za-z]+)")
_GV_REL = re.compile(r"\[\s*\w*:([A-Z_]{3,})")
_GV_PROP = re.compile(r"\b\w+\.(\w+)\s*=|SET\s+\w+\.(\w+)|REQUIRE\s+\w+\.(\w+)|\{\s*(\w+)\s*:")


def graph_vocabulary(root: str = ".", subdir: str = "src/projections") -> Dict[str, CodeTerm]:
    out: Dict[str, CodeTerm] = {}
    start = os.path.join(root, subdir)
    if not os.path.isdir(start):
        return out
    for dirpath, _dirs, files in os.walk(start):
        for fn in files:
            if not fn.endswith(".py"):
                continue
            full = os.path.join(dirpath, fn)
            text = open(full, encoding="utf-8", errors="ignore").read()
            rel = os.path.relpath(full, root).replace(os.sep, "/")
            for m in _GV_LABEL.finditer(text):
                name = m.group(1)
                if name.isupper():
                    continue  # a rel type caught in a rel pattern, not a label
                out.setdefault(name, CodeTerm(name, "graph-label", rel, []))
            for m in _GV_REL.finditer(text):
                out.setdefault(m.group(1), CodeTerm(m.group(1), "rel-type", rel, []))
            for m in _GV_PROP.finditer(text):
                p = next((g for g in m.groups() if g), None)
                if p and not p[0].isupper():
                    out.setdefault(p, CodeTerm(p, "graph-property", rel, []))
    return out
```

In `tools/glossary/check.py`:
- Add `graph_vocabulary` to the import from `tools.glossary.reader`.
- In `run_all`, extend the maps:
```python
    gv = graph_vocabulary(root)
    ...
    findings += check_coverage({**enums, **dims, **gv}, terms)
    findings += check_enum_values({**enums, **dims, **lits}, terms)
    findings += check_stale_source({**enums, **dims, **lits, **gv}, terms)
```
- In `check_stale_source`, extend the kind list so graph-* terms are code-checked by name:
```python
        elif t.kind in ("enum", "dimension", "graph-label", "rel-type", "graph-property"):
            if t.term not in code:
                findings.append(Finding(f"glossary term {t.term}: no longer defined in code (source {t.source})"))
```
(`check_coverage` is unchanged — it just receives the bigger code map.)

- [ ] **Step 4: Run to verify pass**

Run: `~/.pyenv/shims/python -m pytest tests/glossary/ -v` — all green (existing + 2 new; existing enum/dimension/literal behavior unchanged).

- [ ] **Step 5: Commit**

```bash
git add tools/glossary/reader.py tools/glossary/check.py tests/glossary/test_reader.py tests/glossary/test_check.py
git commit -m "feat(glossary): graph_vocabulary extractor + coverage/stale for graph-* kinds"
```

---

### Task 2: Backfill the graph-vocabulary glossary terms

Content task: author the ~64 `graph-label` / `rel-type` / `graph-property` terms and get `glossary-check` clean.

**Files:**
- Create: `docs/glossary/<label-slug>.md` × ~20, `docs/glossary/<rel-slug>.md` × ~17, `docs/glossary/<prop-slug>.md` × ~27
- Modify (generated): `docs/glossary/index.md`

- [ ] **Step 1: Enumerate the real vocabulary**

```bash
~/.pyenv/shims/python -c "from tools.glossary.reader import graph_vocabulary; import json; \
[print(t.kind, name, t.source) for name,t in sorted(graph_vocabulary('.').items())]"
```
This is the authoritative list (labels/rels/props actually in `src/projections/`).

- [ ] **Step 2: Author one term file per item**

Frontmatter per term (terse; a property term may be one line of definition):
```markdown
---
type: Term
term: Claim
kind: graph-label
source: src/projections/handlers/claim_handlers.py
values: []
---
A graph node representing a claim a speaker makes (see also the claim-kind term).
```
Use a slug of the term for the filename (`claim.md`, `made-by.md`, `claim-id.md`). Keep the definitions short — the point is coverage, not prose. Cross-link where obvious (a `FunctionType` label relates to the `function_type` dimension; `claim-id` to `Claim`).

- [ ] **Step 3: Generate + reconcile**

```bash
make glossary-index
make glossary-check      # iterate until clean: every extracted label/rel/prop has a term; no stale graph-* terms
```
`clean` = coverage satisfied for all graph-vocab + no stale. (The enum/dimension/claim-kind/entity-type terms from earlier stay clean.)

- [ ] **Step 4: Commit**

```bash
git add docs/glossary/
git commit -m "docs(glossary): backfill graph vocabulary (~20 labels + ~17 rel types + ~27 properties)"
```

---

### Task 3: Graphq reader (`tools/graphq/reader.py`)

**Files:**
- Create: `tools/graphq/__init__.py` (empty), `tools/graphq/reader.py`
- Test: `tests/graphq/__init__.py` (empty), `tests/graphq/test_reader.py`

**Interfaces:**
- Produces: `@dataclass QueryEntry(bundle, name, purpose, scope, audience, labels, rels, props, returns, consumers)`; `parse_cypher(text) -> (labels, rels, props, returns)`; `parse_graphq_marker(docstring) -> dict`; `load_queries(root=".") -> list[QueryEntry]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/graphq/test_reader.py
from tools.graphq.reader import parse_cypher, parse_graphq_marker, load_queries, QueryEntry

def test_parse_cypher():
    q = "MATCH (i:Interview)-[:HAS_SENTENCE]->(s:Sentence) RETURN s.text AS text, s.sentence_id AS sid"
    labels, rels, props, returns = parse_cypher(q)
    assert set(labels) == {"Interview", "Sentence"}
    assert rels == ["HAS_SENTENCE"]
    assert returns == ["text", "sid"]

def test_parse_graphq_marker():
    doc = "Low-conf queue.\n\ngraphq: purpose=export scope=domain-broad audience=[export, api]\n"
    m = parse_graphq_marker(doc)
    assert m["purpose"] == "export" and m["scope"] == "domain-broad" and m["audience"] == ["export", "api"]

def test_load_queries_finds_query_function(tmp_path):
    d = tmp_path / "src" / "export"; d.mkdir(parents=True)
    (d / "reader.py").write_text(
        'def worklist_rows(session):\n'
        '    """Queue.\n\n    graphq: purpose=export scope=domain-broad audience=[export]\n    """\n'
        '    return session.run("MATCH (i:Interview) RETURN i.interview_id AS interview_id")\n',
        encoding="utf-8")
    entries = load_queries(str(tmp_path))
    e = next(x for x in entries if x.name == "worklist_rows")
    assert e.bundle == "src/export/reader.py" and e.purpose == "export"
    assert e.labels == ["Interview"] and e.returns == ["interview_id"]
```

- [ ] **Step 2: Run to verify fail**

Run: `~/.pyenv/shims/python -m pytest tests/graphq/test_reader.py -v`
Expected: FAIL — no module `tools.graphq`

- [ ] **Step 3: Implement**

```python
# tools/graphq/reader.py
from __future__ import annotations

import ast
import glob
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

_LABEL = re.compile(r"[(\[]\s*\w*:([A-Z][A-Za-z]+)")
_REL = re.compile(r"\[\s*\w*:([A-Z_]{3,})")
_PROP = re.compile(r"\b[a-z_]\w*\.(\w+)")
_RETURN = re.compile(r"\bRETURN\b(.+?)(?:\bORDER\b|\bLIMIT\b|\bSKIP\b|$)", re.I | re.S)
_ALIAS = re.compile(r"\bAS\s+(\w+)")
_MARKER = re.compile(r"graphq:\s*(.+)")

READ_GLOBS = ("src/**/reader.py", "src/api/routers/*.py")


@dataclass
class QueryEntry:
    bundle: str
    name: str
    purpose: str = ""
    scope: str = ""
    audience: List[str] = field(default_factory=list)
    labels: List[str] = field(default_factory=list)
    rels: List[str] = field(default_factory=list)
    props: List[str] = field(default_factory=list)
    returns: List[str] = field(default_factory=list)
    consumers: List[str] = field(default_factory=list)


def parse_cypher(text: str) -> Tuple[List[str], List[str], List[str], List[str]]:
    labels = sorted({m.group(1) for m in _LABEL.finditer(text) if not m.group(1).isupper()})
    rels = sorted({m.group(1) for m in _REL.finditer(text)})
    props = sorted({m.group(1) for m in _PROP.finditer(text)})
    returns: List[str] = []
    rm = _RETURN.search(text)
    if rm:
        returns = [a for a in _ALIAS.findall(rm.group(1))]
    return labels, rels, props, returns


def parse_graphq_marker(docstring: str) -> Dict:
    out: Dict = {}
    m = _MARKER.search(docstring or "")
    if not m:
        return out
    body = m.group(1)
    for key in ("purpose", "scope"):
        km = re.search(rf"{key}=([\w\-]+)", body)
        if km:
            out[key] = km.group(1)
    am = re.search(r"audience=\[([^\]]*)\]", body)
    if am:
        out["audience"] = [x.strip() for x in am.group(1).split(",") if x.strip()]
    return out


def _cypher_of(fn: ast.AST) -> str:
    parts = []
    for node in ast.walk(fn):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if re.search(r"\b(MATCH|MERGE|RETURN|CALL|CREATE)\b", node.value):
                parts.append(node.value)
    return "\n".join(parts)


def load_queries(root: str = ".", read_globs=READ_GLOBS) -> List[QueryEntry]:
    seen = set()
    files = []
    for g in read_globs:
        for f in glob.glob(os.path.join(root, g), recursive=True):
            if f not in seen:
                seen.add(f); files.append(f)
    entries: List[QueryEntry] = []
    for f in sorted(files):
        rel = os.path.relpath(f, root).replace(os.sep, "/")
        try:
            tree = ast.parse(open(f, encoding="utf-8").read())
        except Exception:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            cypher = _cypher_of(node)
            if not cypher:
                continue
            labels, rels, props, returns = parse_cypher(cypher)
            meta = parse_graphq_marker(ast.get_docstring(node) or "")
            entries.append(QueryEntry(
                bundle=rel, name=node.name,
                purpose=meta.get("purpose", ""), scope=meta.get("scope", ""),
                audience=meta.get("audience", []),
                labels=labels, rels=rels, props=props, returns=returns,
                consumers=derive_consumers(node.name, root),
            ))
    return entries


def derive_consumers(fn_name: str, root: str = ".") -> List[str]:
    roles = set()
    call = re.compile(rf"\b{re.escape(fn_name)}\s*\(")
    for f in glob.glob(os.path.join(root, "src", "**", "*.py"), recursive=True):
        rel = os.path.relpath(f, root).replace(os.sep, "/")
        if rel.endswith("reader.py"):
            continue  # skip the definition sites
        try:
            if call.search(open(f, encoding="utf-8", errors="ignore").read()):
                roles.add(rel.split("/")[1] if rel.startswith("src/") else rel)
        except Exception:
            continue
    return sorted(roles)
```

- [ ] **Step 4: Run to verify pass**

Run: `~/.pyenv/shims/python -m pytest tests/graphq/test_reader.py -v` → PASS (3)

- [ ] **Step 5: Commit**

```bash
git add tools/graphq/__init__.py tools/graphq/reader.py tests/graphq/__init__.py tests/graphq/test_reader.py
git commit -m "feat(graphq): reader — query discovery, Cypher parse, graphq markers, consumers"
```

---

### Task 4: Graphq renderer (`tools/graphq/render.py`)

**Files:** Create `tools/graphq/render.py`; Test `tests/graphq/test_render.py`

**Interfaces:** `render_catalog(entries) -> str` (grouped by bundle)

- [ ] **Step 1: Write the failing test**

```python
# tests/graphq/test_render.py
from tools.graphq.reader import QueryEntry
from tools.graphq.render import render_catalog

def test_render_catalog():
    e = QueryEntry("src/export/reader.py", "worklist_rows", "export", "domain-broad",
                   ["export"], ["Interview"], [], [], ["interview_id"], ["api"])
    out = render_catalog([e])
    assert "## src/export/reader.py" in out
    assert "worklist_rows" in out and "domain-broad" in out and "Interview" in out and "interview_id" in out
```

- [ ] **Step 2: Run to verify fail** — `~/.pyenv/shims/python -m pytest tests/graphq/test_render.py -v` (no module)

- [ ] **Step 3: Implement**

```python
# tools/graphq/render.py
from __future__ import annotations

from typing import List

from tools.graphq.reader import QueryEntry


def render_catalog(entries: List[QueryEntry]) -> str:
    by_bundle: dict = {}
    for e in entries:
        by_bundle.setdefault(e.bundle, []).append(e)
    lines = ["# Graph-query registry", ""]
    for bundle in sorted(by_bundle):
        lines.append(f"## {bundle}")
        lines.append("")
        lines.append("| query | purpose | scope | audience | consumers | labels | rels | returns |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for e in sorted(by_bundle[bundle], key=lambda e: e.name):
            lines.append(f"| {e.name} | {e.purpose} | {e.scope} | {', '.join(e.audience)} | "
                         f"{', '.join(e.consumers)} | {', '.join(e.labels)} | {', '.join(e.rels)} | "
                         f"{', '.join(e.returns)} |")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
```

- [ ] **Step 4: Run to verify pass** — PASS

- [ ] **Step 5: Commit**

```bash
git add tools/graphq/render.py tests/graphq/test_render.py
git commit -m "feat(graphq): catalog renderer"
```

---

### Task 5: Graphq guard (`tools/graphq/check.py`)

**Files:** Create `tools/graphq/check.py`; Test `tests/graphq/test_check.py`

**Interfaces:** `@dataclass Finding`; `check_schema_drift(entries, vocab)`, `check_output_contract(entries, root)`, `check_missing_marker(entries)`, `check_catalog_in_sync(path, entries)`, `run_all(root=".")`

- [ ] **Step 1: Write the failing test**

```python
# tests/graphq/test_check.py
from tools.graphq.reader import QueryEntry
from tools.graphq.check import check_schema_drift, check_missing_marker, Finding

def test_schema_drift_flags_unknown_label():
    vocab = {"Interview": None, "HAS_SENTENCE": None}   # keys = known names
    e = QueryEntry("b.py", "q", "export", "task", ["export"], ["Interview", "Ghost"], ["HAS_SENTENCE"], [], [], [])
    msgs = " ".join(f.message for f in check_schema_drift([e], vocab))
    assert "Ghost" in msgs and "Interview" not in msgs.replace("Ghost", "")

def test_missing_marker_flags_unannotated():
    e = QueryEntry("b.py", "q", "", "", [], ["Interview"], [], [], [], [])
    msgs = " ".join(f.message for f in check_missing_marker([e]))
    assert "q" in msgs
```

- [ ] **Step 2: Run to verify fail** — no module `tools.graphq.check`

- [ ] **Step 3: Implement**

```python
# tools/graphq/check.py
from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass
from typing import Dict, List

from tools.graphq.reader import QueryEntry, load_queries
from tools.graphq.render import render_catalog
from tools.glossary.reader import graph_vocabulary


@dataclass
class Finding:
    message: str


def check_schema_drift(entries: List[QueryEntry], vocab: Dict) -> List[Finding]:
    known = set(vocab)
    findings: List[Finding] = []
    for e in entries:
        for label in e.labels:
            if label not in known:
                findings.append(Finding(f"{e.bundle}:{e.name} references label :{label} not produced by any projection"))
        for rel in e.rels:
            if rel not in known:
                findings.append(Finding(f"{e.bundle}:{e.name} references rel [:{rel}] not produced by any projection"))
    return findings


def check_output_contract(entries: List[QueryEntry], root: str = ".") -> List[Finding]:
    findings: List[Finding] = []
    src = {os.path.relpath(f, root).replace(os.sep, "/"): open(f, encoding="utf-8", errors="ignore").read()
           for f in glob.glob(os.path.join(root, "src", "**", "*.py"), recursive=True)}
    for e in entries:
        if not e.returns:
            continue
        returned = set(e.returns)
        call = re.compile(rf"\b{re.escape(e.name)}\s*\(")
        access = re.compile(r"""(?:\[["']|\.get\(["'])(\w+)["']""")
        for rel, text in src.items():
            if rel.endswith("reader.py") or not call.search(text):
                continue
            for m in access.finditer(text):
                fld = m.group(1)
                # only flag fields that look like query outputs (appear in some query's returns)
                if fld not in returned and any(fld in x.returns for x in entries):
                    findings.append(Finding(f"{rel} reads field '{fld}' not returned by {e.bundle}:{e.name}"))
    return findings


def check_missing_marker(entries: List[QueryEntry]) -> List[Finding]:
    return [Finding(f"{e.bundle}:{e.name} has no graphq: marker (purpose/scope/audience)")
            for e in entries if not e.purpose]


def check_catalog_in_sync(catalog_path: str, entries: List[QueryEntry]) -> List[Finding]:
    want = render_catalog(entries)
    have = open(catalog_path, encoding="utf-8").read() if os.path.exists(catalog_path) else ""
    return [Finding("docs/graph-queries/index.md out of sync — run make graphq-index")] if want != have else []


def run_all(root: str = ".") -> List[Finding]:
    entries = load_queries(root)
    vocab = graph_vocabulary(root)
    findings: List[Finding] = []
    findings += check_schema_drift(entries, vocab)
    findings += check_output_contract(entries, root)
    findings += check_missing_marker(entries)
    findings += check_catalog_in_sync(os.path.join(root, "docs/graph-queries/index.md"), entries)
    return findings
```

> Note on `check_output_contract`: it is heuristic (direct callers, dict-access patterns) and
> deliberately conservative — it only flags a field access when that field is a real query
> output *somewhere* (so unrelated `dict["x"]` accesses don't false-positive). Misses are
> acceptable; false positives are the thing to avoid.

- [ ] **Step 4: Run to verify pass** — `~/.pyenv/shims/python -m pytest tests/graphq/test_check.py -v` PASS

- [ ] **Step 5: Commit**

```bash
git add tools/graphq/check.py tests/graphq/test_check.py
git commit -m "feat(graphq): guard — schema-drift, output-contract, missing-marker, catalog-sync"
```

---

### Task 6: Graphq CLI + Makefile

**Files:** Create `tools/graphq/__main__.py`; Modify `Makefile`; Test `tests/graphq/test_cli.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/graphq/test_cli.py
import subprocess, sys

def test_cli_check_exits_zero():
    proc = subprocess.run([sys.executable, "-m", "tools.graphq", "check"], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "graphq-check" in proc.stdout
```

- [ ] **Step 2: Run to verify fail** — no `tools.graphq.__main__`

- [ ] **Step 3: Implement** — mirror the other tools' `__main__.py` (subcommands `index`/`check`; `index` writes `docs/graph-queries/index.md` via `render_catalog(load_queries())`; `check` prints `run_all()` findings, `return 0`). Add Makefile targets:

```makefile
.PHONY: graphq-index
graphq-index: ## Regenerate docs/graph-queries/index.md (graph-query registry)
	@$(PYTHON) -m tools.graphq index

.PHONY: graphq-check
graphq-check: ## Reconcile graph queries vs schema + consumers (non-blocking)
	@$(PYTHON) -m tools.graphq check
```

- [ ] **Step 4: Run test + smoke** — test PASS; `~/.pyenv/shims/python -m tools.graphq check` exit 0 (findings expected until Task 7).

- [ ] **Step 5: Commit**

```bash
git add tools/graphq/__main__.py Makefile tests/graphq/test_cli.py
git commit -m "feat(graphq): CLI (index/check) + make targets"
```

---

### Task 7: Backfill `graphq:` markers + generate catalog

**Files:** Modify `src/export/reader.py`, `src/ui/reader.py`, `src/ask/reader.py`, `src/resolution/reader.py`, inline `src/api/routers/segments.py` (add a `graphq:` line to each query function's docstring); Create `docs/graph-queries/index.md`

- [ ] **Step 1: List the query functions needing markers**

```bash
~/.pyenv/shims/python -c "from tools.graphq.reader import load_queries; \
[print(e.bundle, e.name, '(marker!)' if not e.purpose else '') for e in load_queries('.')]"
```

- [ ] **Step 2: Add a `graphq:` line to each query function's docstring**

For each query function, add one line inside its docstring:
`graphq: purpose=<export|ui|ask|resolution> scope=<task|domain-broad> audience=[<roles>]`
- `purpose` = the bundle's purpose (export/ui/ask/resolution).
- `scope` = judge per query: single-entity/one-parameter fetches → `task`; broad rollups / worklists / context retrieval → `domain-broad`.
- `audience` = the derived consumers + `agents` for the `ask` bundle (agent-facing retrieval).
Preserve the existing docstring text; add the `graphq:` line after it.

- [ ] **Step 3: Generate the catalog + reconcile**

```bash
make graphq-index
~/.pyenv/shims/python -m tools.graphq check     # iterate
```
Expected end state: **no missing-marker** findings (all annotated), catalog in sync. Any **schema-drift** findings are REAL (a query referencing a label/rel the projections don't create) — investigate and report them (do not silence); likewise any genuine **output-contract** finding. If a drift is a parser artifact (e.g. a label in a comment), note it. Report the final `graphq check` output verbatim.

- [ ] **Step 4: Commit**

```bash
git add src/ docs/graph-queries/index.md
git commit -m "feat(graphq): backfill graphq: markers + generated registry catalog"
```

---

## Final verification

- [ ] `~/.pyenv/shims/python -m pytest tests/glossary/ tests/graphq/ -v` — all green.
- [ ] `make glossary-check` — clean (graph vocabulary covered).
- [ ] `make graphq-check` — no missing-marker / catalog-sync findings; any schema-drift/output-contract findings are triaged as real-or-noise and reported.
- [ ] `make graphq-index` / `make glossary-index` then `git status` — generated files regenerate identically.
- [ ] `make cli-index` — regenerate the CLI catalog to include `graphq-*` (then `cli-check`, `adr-check`, `prompt-check` behave).
