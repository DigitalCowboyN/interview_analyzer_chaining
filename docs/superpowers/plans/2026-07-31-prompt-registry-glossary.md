# Probabilistic-Components Registry (F) + Glossary Extension Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct + extend + "living"-frame the glossary, and stand up a probabilistic-components registry (`tools/prompts/`) that documents prompts as code (flagged probabilistic), records their purpose/audience, derives their consumers, and reconciles values↔glossary and audience↔code.

**Architecture:** Part 1 extends `tools/glossary` (a `Literal[...]` reader) + glossary content. Part 2 is a new `tools/prompts/` package (reader → render → check → CLI) that reads `prompts/*.yaml` (with new `used_for`/`audience` metadata), derives consumers from code, and reconciles against the glossary.

**Tech Stack:** Python 3 (stdlib `ast`/`re`; PyYAML via `parse_front_matter` and direct `yaml`), pytest, Make.

## Global Constraints

- **Non-blocking, always.** Checks return `list[Finding]`; none raises; `make {glossary,prompt}-check` exit 0.
- **Direction: registry → glossary.** Value-mismatch findings name the *glossary* as the thing to fix. Code-pinned terms (enums, `claim-kind` Literal) → code is truth.
- **Living domain:** glossary `kind` is free-form; guards never assume completeness.
- **Consumers derived per prompt FILE** (the loader loads the whole file); `lens_*.yaml` → role `lens` by convention (loaded via `lens.prompts_file`).
- **Value extraction handles two shapes:** `"field": "a|b|c"` Format lines (claims, entity) and `Options:` bullet lists (dimensions).
- `Endpoint`/`Term`/`Finding`/`PromptEntry` stay local to their tool packages. Tests in `tests/prompts/` and `tests/glossary/`.
- Run tests with `~/.pyenv/shims/python -m pytest <path> -v`.

---

### Task 1: Glossary — `code_literals` reader + `code_symbol` reconciliation

**Files:**
- Modify: `tools/glossary/reader.py` (add `code_literals`), `tools/glossary/model.py` (add `code_symbol`), `tools/glossary/check.py` (extend `check_enum_values` to match code-literal terms)
- Test: `tests/glossary/test_reader.py`, `tests/glossary/test_check.py` (append)

**Interfaces:**
- Produces: `code_literals(root=".", subdirs=("src",)) -> dict[str, CodeTerm]` keyed by `"ClassName.field"`; `Term.code_symbol: Optional[str]`.

- [ ] **Step 1: Write the failing tests** (append)

```python
# tests/glossary/test_reader.py  (append)
def test_code_literals_extracts_field_literals(tmp_path):
    from tools.glossary.reader import code_literals
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "m.py").write_text(
        "from typing import Literal\n"
        "class Claim:\n    kind: Literal['assertion', 'commitment', 'request']\n", encoding="utf-8")
    lits = code_literals(str(tmp_path))
    assert "Claim.kind" in lits
    assert lits["Claim.kind"].values == ["assertion", "commitment", "request"]
    assert lits["Claim.kind"].source == "src/m.py"
```

```python
# tests/glossary/test_check.py  (append)
def test_enum_values_matches_code_symbol_literal():
    from tools.glossary.reader import CodeTerm
    from tools.glossary.model import Term
    from tools.glossary.check import check_enum_values
    code = {"Claim.kind": CodeTerm("Claim.kind", "literal", "src/m.py", ["assertion", "commitment", "request"])}
    term = Term("claim-kind", "claim-kind", "src/m.py", ["assertion", "commitment"], "", "p")  # missing request
    term.code_symbol = "Claim.kind"
    msgs = " ".join(f.message for f in check_enum_values(code, [term]))
    assert "claim-kind" in msgs and "request" in msgs
```

- [ ] **Step 2: Run to verify they fail**

Run: `~/.pyenv/shims/python -m pytest tests/glossary/test_reader.py::test_code_literals_extracts_field_literals tests/glossary/test_check.py::test_enum_values_matches_code_symbol_literal -v`
Expected: FAIL (`code_literals` / `code_symbol` missing)

- [ ] **Step 3: Implement**

In `tools/glossary/reader.py`, add (uses `ast`):

```python
def code_literals(root: str = ".", subdirs=("src",)) -> Dict[str, CodeTerm]:
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
                rel = os.path.relpath(full, root).replace(os.sep, "/")
                for node in ast.walk(tree):
                    if not isinstance(node, ast.ClassDef):
                        continue
                    for b in node.body:
                        if isinstance(b, ast.AnnAssign) and isinstance(b.target, ast.Name):
                            ann = b.annotation
                            if isinstance(ann, ast.Subscript) and _name_of(ann.value) == "Literal":
                                vals = [e.value for e in _literal_elts(ann.slice)
                                        if isinstance(e, ast.Constant) and isinstance(e.value, str)]
                                if vals:
                                    out[f"{node.name}.{b.target.id}"] = CodeTerm(
                                        f"{node.name}.{b.target.id}", "literal", rel, vals)
    return out


def _name_of(n):
    return n.id if isinstance(n, ast.Name) else getattr(n, "attr", "")


def _literal_elts(sl):
    node = sl.value if isinstance(sl, ast.Index) else sl  # py<3.9 compat
    return node.elts if isinstance(node, ast.Tuple) else [node]
```

In `tools/glossary/model.py`, add `code_symbol` to `Term` (after `values`) and parse it:

```python
    values: List[str] = field(default_factory=list)
    code_symbol: Optional[str] = None
```
```python
        values=[str(v) for v in (fm.get("values") or [])],
        code_symbol=fm.get("code_symbol"),
```

In `tools/glossary/check.py`, extend `check_enum_values` so a term with `code_symbol` reconciles against `code[code_symbol]`:

```python
def check_enum_values(code: Dict[str, CodeTerm], terms: List[Term]) -> List[Finding]:
    findings: List[Finding] = []
    for t in terms:
        key = getattr(t, "code_symbol", None) or (t.term if t.kind == "enum" else None)
        if key is None:
            continue
        ct = code.get(key)
        if ct is None:
            continue
        if set(t.values) != set(ct.values):
            missing = sorted(set(ct.values) - set(t.values))
            extra = sorted(set(t.values) - set(ct.values))
            findings.append(Finding(f"glossary term {t.term} values differ from code (missing: {missing}, extra: {extra})"))
    return findings
```

Wire `code_literals` into `run_all`'s code map:

```python
    code = {**code_enums(root), **code_dimensions(root), **code_literals(root)}
```
(add `code_literals` to the import from `tools.glossary.reader`).

- [ ] **Step 4: Run to verify pass**

Run: `~/.pyenv/shims/python -m pytest tests/glossary/ -v`
Expected: PASS (existing + new; the earlier `check_enum_values` tests still hold — enum terms match by name via the `t.term` fallback).

- [ ] **Step 5: Commit**

```bash
git add tools/glossary/reader.py tools/glossary/model.py tools/glossary/check.py tests/glossary/test_reader.py tests/glossary/test_check.py
git commit -m "feat(glossary): Literal[...] reader + code_symbol reconciliation (for claim-kind)"
```

---

### Task 2: Glossary content — fix values, living README, new terms

**Files:**
- Create: `docs/glossary/README.md`, `docs/glossary/claim-kind.md`, `docs/glossary/entity-type.md`
- Modify: `docs/glossary/purpose.md`, `docs/glossary/topic-level-1.md`, `docs/glossary/topic-level-3.md`
- Modify (generated): `docs/glossary/index.md`

- [ ] **Step 1: Fix the drifted dimension values** (read the exact lists from `prompts/core_extractors.yaml`)

Set `purpose.md` `values` to the 24 (`Statement, Query, Exclamation, Answer, Commentary, Observation, Retraction, Mockery, Objection, Clarification, Conclusion, Confession, Speculation, Recitation, Correction, Explanation, Qualification, Threat, Warning, Advisory, Request, Addendum, Musing, Amendment`) and drop the "initial set" caveat. Set `topic-level-1.md` and `topic-level-3.md` `values` to the 15 (`goals, tools, processes, experiences, observations, pain points, responsibilities, collaborations, reporting, managing, mentoring, strategy, operations, small talk, niceties`); update the definitions to say these are the fixed categories the prompt enumerates.

- [ ] **Step 2: Add the living-domain README**

`docs/glossary/README.md`:
```markdown
# Glossary — a living domain reference

This is a **growing** reference for the project's vocabulary. It expands in both
**type** (new kinds: enum, dimension, claim-kind, entity-type, and more over time) and
**amount** (new terms and values). The absence of a term means "not yet catalogued,"
not "does not exist."

- `kind` is free-form — new vocabulary kinds are added without any tooling change.
- Enum / Literal terms are checked against code; registry-pinned terms (dimensions,
  entity-type) are authored from the prompt registry, which is the operative source of
  truth (the prompt registry guard keeps them honest).
```

- [ ] **Step 3: Add the two new terms**

`docs/glossary/claim-kind.md` (code-pinned via the Literal):
```markdown
---
type: Term
term: claim-kind
kind: claim-kind
source: src/models/extractor_responses.py
code_symbol: Claim.kind
values: [assertion, commitment, request]
---
The kind of claim a speaker makes: an assertion (statement of fact/belief), a
commitment (something they will do), or a request (something they ask of another).
```
(If the model class is not named `Claim`, set `code_symbol` to the real `ClassName.kind` — confirm with `code_literals`.)

`docs/glossary/entity-type.md` (registry-pinned):
```markdown
---
type: Term
term: entity-type
kind: entity-type
source: prompts/core_extractors.yaml
values: [person, organization, product, tool, other]
---
The type of an extracted entity mention, per the entity_mentions prompt's output enum.
```

- [ ] **Step 4: Regenerate + reconcile**

```bash
make glossary-index
make glossary-check      # iterate until clean (claim-kind reconciles vs the Literal; others coverage-clean)
```
Confirm `code_symbol` for claim-kind matches the real class (`python -m tools.glossary check` should not flag it).

- [ ] **Step 5: Commit**

```bash
git add docs/glossary/
git commit -m "docs(glossary): fix purpose/topic values, living README, add claim-kind + entity-type"
```

---

### Task 3: Prompts reader (`tools/prompts/reader.py`)

**Files:**
- Create: `tools/prompts/__init__.py` (empty), `tools/prompts/reader.py`
- Test: `tests/prompts/__init__.py` (empty), `tests/prompts/test_reader.py`

**Interfaces:**
- Produces: `@dataclass PromptEntry(file, key, used_for, audience, values, consumers)`; `extract_values(text) -> list[str]`; `derive_consumers(prompt_filename, root=".") -> list[str]`; `load_prompt_entries(root=".") -> list[PromptEntry]`

- [ ] **Step 1: Write the failing test**

```python
# tests/prompts/test_reader.py
from tools.prompts.reader import extract_values, derive_consumers, load_prompt_entries, PromptEntry

def test_extract_values_both_shapes():
    fmt = 'Report {"entity_type": "person|organization|tool"} for each.'
    assert extract_values(fmt) == ["person", "organization", "tool"]
    bullets = "Choose one.\nOptions:\n  - declarative\n  - interrogative\n  - imperative\n"
    assert extract_values(bullets) == ["declarative", "interrogative", "imperative"]
    assert extract_values("free-form prompt, no enum") == []

def test_derive_consumers_and_lens_convention(tmp_path):
    (tmp_path / "src" / "enrichment").mkdir(parents=True)
    (tmp_path / "src" / "enrichment" / "o.py").write_text('load_yaml("prompts/core_extractors.yaml")', encoding="utf-8")
    assert derive_consumers("core_extractors.yaml", root=str(tmp_path)) == ["enrichment"]
    assert derive_consumers("lens_persona.yaml", root=str(tmp_path)) == ["lens"]   # convention
    assert derive_consumers("task_prompts.yaml", root=str(tmp_path)) == []          # orphan

def test_load_prompt_entries_reads_metadata(tmp_path):
    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts" / "core_extractors.yaml").write_text(
        "function_type:\n  used_for: [classification]\n  audience: [enrichment]\n"
        "  prompt: |\n    Options:\n      - declarative\n      - interrogative\n", encoding="utf-8")
    (tmp_path / "src" / "enrichment").mkdir(parents=True)
    (tmp_path / "src" / "enrichment" / "o.py").write_text('"prompts/core_extractors.yaml"', encoding="utf-8")
    entries = load_prompt_entries(str(tmp_path))
    e = next(x for x in entries if x.key == "function_type")
    assert e.used_for == ["classification"] and e.audience == ["enrichment"]
    assert e.values == ["declarative", "interrogative"] and e.consumers == ["enrichment"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `~/.pyenv/shims/python -m pytest tests/prompts/test_reader.py -v`
Expected: FAIL — `No module named 'tools.prompts'`

- [ ] **Step 3: Implement**

```python
# tools/prompts/reader.py
from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass, field
from typing import List

import yaml

_FMT = re.compile(r'"[a-z_]+":\s*"([a-z][a-z|_\-]+)"')
STAGE_BY_PREFIX = {
    "src/enrichment": "enrichment", "src/ingestion": "ingestion",
    "src/ask": "ask", "src/lens": "lens", "src/api": "api",
}


@dataclass
class PromptEntry:
    file: str
    key: str
    used_for: List[str] = field(default_factory=list)
    audience: List[str] = field(default_factory=list)
    values: List[str] = field(default_factory=list)
    consumers: List[str] = field(default_factory=list)


def extract_values(text: str) -> List[str]:
    for m in _FMT.finditer(text):
        if "|" in m.group(1):
            return m.group(1).split("|")
    bullets = re.findall(r'^\s*-\s*([A-Za-z][\w\- ]*?)\s*$', text, re.M)
    if len(bullets) >= 3:
        return [b.strip() for b in bullets]
    return []


def derive_consumers(prompt_filename: str, root: str = ".") -> List[str]:
    base = os.path.basename(prompt_filename)
    if base.startswith("lens_"):
        return ["lens"]                       # loaded dynamically via lens.prompts_file
    roles = set()
    needle = f"prompts/{base}"
    for f in glob.glob(os.path.join(root, "src", "**", "*.py"), recursive=True):
        try:
            if needle in open(f, encoding="utf-8", errors="ignore").read():
                rel = os.path.relpath(f, root).replace(os.sep, "/")
                for prefix, role in STAGE_BY_PREFIX.items():
                    if rel.startswith(prefix):
                        roles.add(role)
        except Exception:
            continue
    return sorted(roles)


def load_prompt_entries(root: str = ".") -> List[PromptEntry]:
    entries: List[PromptEntry] = []
    for path in sorted(glob.glob(os.path.join(root, "prompts", "*.yaml"))):
        base = os.path.basename(path)
        try:
            data = yaml.safe_load(open(path, encoding="utf-8")) or {}
        except Exception:
            continue
        consumers = derive_consumers(base, root)
        for key, v in data.items():
            if not (isinstance(v, dict) and "prompt" in v):
                continue
            entries.append(PromptEntry(
                file=base, key=key,
                used_for=list(v.get("used_for") or []),
                audience=list(v.get("audience") or []),
                values=extract_values(str(v.get("prompt", ""))),
                consumers=consumers,
            ))
    return entries
```

- [ ] **Step 4: Run to verify it passes**

Run: `~/.pyenv/shims/python -m pytest tests/prompts/test_reader.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add tools/prompts/__init__.py tools/prompts/reader.py tests/prompts/__init__.py tests/prompts/test_reader.py
git commit -m "feat(prompts): reader — value extraction, consumer derivation, metadata"
```

---

### Task 4: Prompts renderer (`tools/prompts/render.py`)

**Files:**
- Create: `tools/prompts/render.py`
- Test: `tests/prompts/test_render.py`

**Interfaces:** `render_catalog(entries) -> str`

- [ ] **Step 1: Write the failing test**

```python
# tests/prompts/test_render.py
from tools.prompts.reader import PromptEntry
from tools.prompts.render import render_catalog

def test_render_catalog():
    e = PromptEntry("core_extractors.yaml", "function_type", ["classification"], ["enrichment"],
                    ["declarative", "interrogative"], ["enrichment"])
    out = render_catalog([e])
    assert "## core_extractors.yaml" in out
    assert "function_type" in out and "probabilistic" in out
    assert "classification" in out and "enrichment" in out
    assert "declarative" in out
```

- [ ] **Step 2: Run to verify it fails**

Run: `~/.pyenv/shims/python -m pytest tests/prompts/test_render.py -v`
Expected: FAIL — no module `tools.prompts.render`

- [ ] **Step 3: Implement**

```python
# tools/prompts/render.py
from __future__ import annotations

from typing import List

from tools.prompts.reader import PromptEntry


def render_catalog(entries: List[PromptEntry]) -> str:
    by_file: dict = {}
    for e in entries:
        by_file.setdefault(e.file, []).append(e)
    lines = ["# Prompt registry (probabilistic components)", ""]
    for file in sorted(by_file):
        lines.append(f"## {file}")
        lines.append("")
        lines.append("| key | classification | used_for | audience | consumers | values |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for e in sorted(by_file[file], key=lambda e: e.key):
            vals = ", ".join(e.values) if e.values else ""
            lines.append(f"| {e.key} | probabilistic | {', '.join(e.used_for)} | "
                         f"{', '.join(e.audience)} | {', '.join(e.consumers)} | {vals} |")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
```

- [ ] **Step 4: Run to verify it passes**

Run: `~/.pyenv/shims/python -m pytest tests/prompts/test_render.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tools/prompts/render.py tests/prompts/test_render.py
git commit -m "feat(prompts): catalog renderer (prompts as probabilistic components)"
```

---

### Task 5: Prompts guard (`tools/prompts/check.py`)

**Files:**
- Create: `tools/prompts/check.py`
- Test: `tests/prompts/test_check.py`

**Interfaces:**
- Consumes: `tools.prompts.reader`, `tools.prompts.render.render_catalog`, `tools.glossary.model.load_glossary`
- Produces: `@dataclass Finding`; `check_values_vs_glossary`, `check_audience_vs_consumers`, `check_orphan`, `check_catalog_in_sync`, `run_all(root=".")`; `KEY_TO_TERM` map; `INTERNAL_ROLES`

- [ ] **Step 1: Write the failing test**

```python
# tests/prompts/test_check.py
from tools.prompts.reader import PromptEntry
from tools.glossary.model import Term
from tools.prompts.check import (
    check_values_vs_glossary, check_audience_vs_consumers, check_orphan, Finding,
)

def test_values_vs_glossary_names_glossary_as_fix():
    entry = PromptEntry("core_extractors.yaml", "purpose", ["classification"], ["enrichment"],
                        ["Statement", "Query"], ["enrichment"])
    terms = [Term("purpose", "dimension", "prompts/core_extractors.yaml", ["Statement"], "", "p")]  # missing Query
    msgs = " ".join(f.message for f in check_values_vs_glossary([entry], terms))
    assert "purpose" in msgs and "glossary" in msgs.lower() and "Query" in msgs

def test_audience_vs_consumers():
    # declared internal role with no consumer -> flagged
    e1 = PromptEntry("x.yaml", "k", ["extraction"], ["api"], [], [])
    m1 = " ".join(f.message for f in check_audience_vs_consumers([e1]))
    assert "api" in m1 and "no code consumes" in m1
    # consumed by a role not declared -> flagged
    e2 = PromptEntry("x.yaml", "k", ["extraction"], [], [], ["enrichment"])
    m2 = " ".join(f.message for f in check_audience_vs_consumers([e2]))
    assert "enrichment" in m2 and "not" in m2.lower()
    # external role declared -> not reconciled (no finding for cli alone)
    e3 = PromptEntry("x.yaml", "k", ["extraction"], ["cli"], [], ["enrichment"])
    m3 = " ".join(f.message for f in check_audience_vs_consumers([e3]))
    assert "cli" not in m3

def test_orphan_flags_no_consumer():
    e = PromptEntry("task_prompts.yaml", "sentence_purpose", [], [], ["Statement"], [])
    msgs = " ".join(f.message for f in check_orphan([e]))
    assert "task_prompts.yaml" in msgs and "unused" in msgs
```

- [ ] **Step 2: Run to verify it fails**

Run: `~/.pyenv/shims/python -m pytest tests/prompts/test_check.py -v`
Expected: FAIL — no module `tools.prompts.check`

- [ ] **Step 3: Implement**

```python
# tools/prompts/check.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List

from tools.prompts.reader import PromptEntry, load_prompt_entries
from tools.prompts.render import render_catalog
from tools.glossary.model import load_glossary

# prompt key -> glossary term
KEY_TO_TERM = {
    "function_type": "function_type", "structure_type": "structure_type",
    "purpose": "purpose", "topic_level_1": "topic_level_1", "topic_level_3": "topic_level_3",
    "entity_mentions": "entity-type", "claims": "claim-kind",
}
INTERNAL_ROLES = {"enrichment", "ingestion", "ask", "lens", "api", "agent"}


@dataclass
class Finding:
    message: str


def check_values_vs_glossary(entries: List[PromptEntry], terms: List) -> List[Finding]:
    by_term = {t.term: t for t in terms}
    findings: List[Finding] = []
    for e in entries:
        term_name = KEY_TO_TERM.get(e.key)
        if not term_name or not e.values:
            continue
        t = by_term.get(term_name)
        if t is None:
            findings.append(Finding(f"prompt {e.file}:{e.key} enumerates values but glossary has no term {term_name}"))
            continue
        if set(e.values) != set(t.values):
            missing = sorted(set(e.values) - set(t.values))
            extra = sorted(set(t.values) - set(e.values))
            findings.append(Finding(
                f"glossary term {term_name} out of sync with the registry ({e.file}:{e.key}) — "
                f"missing: {missing}, extra: {extra} — update the glossary"))
    return findings


def check_audience_vs_consumers(entries: List[PromptEntry]) -> List[Finding]:
    findings: List[Finding] = []
    for e in entries:
        declared_internal = {a for a in e.audience if a in INTERNAL_ROLES}
        for role in sorted(declared_internal):
            if role not in e.consumers:
                findings.append(Finding(f"{e.file}:{e.key} declares audience {role} but no code consumes it"))
        for role in sorted(set(e.consumers) - set(e.audience)):
            findings.append(Finding(f"{e.file}:{e.key} is consumed by {role} but audience does not list it"))
    return findings


def check_orphan(entries: List[PromptEntry]) -> List[Finding]:
    findings: List[Finding] = []
    seen_files = set()
    for e in entries:
        external = [a for a in e.audience if a not in INTERNAL_ROLES]
        if not e.consumers and not external and e.file not in seen_files:
            findings.append(Finding(f"{e.file} appears unused (no code consumer)"))
            seen_files.add(e.file)
    return findings


def check_missing_metadata(entries: List[PromptEntry]) -> List[Finding]:
    return [Finding(f"{e.file}:{e.key} has no used_for/audience metadata")
            for e in entries if e.consumers and not (e.used_for or e.audience)]


def check_catalog_in_sync(catalog_path: str, entries: List[PromptEntry]) -> List[Finding]:
    want = render_catalog(entries)
    have = open(catalog_path, encoding="utf-8").read() if os.path.exists(catalog_path) else ""
    if want != have:
        return [Finding("docs/prompts/index.md out of sync — run make prompt-index")]
    return []


def run_all(root: str = ".") -> List[Finding]:
    entries = load_prompt_entries(root)
    terms = load_glossary(os.path.join(root, "docs/glossary"))
    findings: List[Finding] = []
    findings += check_values_vs_glossary(entries, terms)
    findings += check_audience_vs_consumers(entries)
    findings += check_orphan(entries)
    findings += check_missing_metadata(entries)
    findings += check_catalog_in_sync(os.path.join(root, "docs/prompts/index.md"), entries)
    return findings
```

- [ ] **Step 4: Run to verify it passes**

Run: `~/.pyenv/shims/python -m pytest tests/prompts/test_check.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add tools/prompts/check.py tests/prompts/test_check.py
git commit -m "feat(prompts): guard — values<->glossary, audience<->consumers, orphan, catalog-sync"
```

---

### Task 6: Prompts CLI + Makefile targets

**Files:**
- Create: `tools/prompts/__main__.py`
- Modify: `Makefile` (add `prompt-index`, `prompt-check`, each with a `##` doc)
- Test: `tests/prompts/test_cli.py`

**Interfaces:** `python -m tools.prompts {index|check}` (exit 0)

- [ ] **Step 1: Write the failing test**

```python
# tests/prompts/test_cli.py
import subprocess, sys

def test_cli_check_exits_zero():
    proc = subprocess.run([sys.executable, "-m", "tools.prompts", "check"], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "prompt-check" in proc.stdout
```

- [ ] **Step 2: Run to verify it fails**

Run: `~/.pyenv/shims/python -m pytest tests/prompts/test_cli.py -v`
Expected: FAIL — no `tools.prompts.__main__`

- [ ] **Step 3: Implement**

```python
# tools/prompts/__main__.py
from __future__ import annotations

import argparse
import os
import sys

from tools.prompts.check import run_all
from tools.prompts.reader import load_prompt_entries
from tools.prompts.render import render_catalog

CATALOG = "docs/prompts/index.md"


def cmd_index(args) -> int:
    os.makedirs(os.path.dirname(CATALOG), exist_ok=True)
    with open(CATALOG, "w", encoding="utf-8") as fh:
        fh.write(render_catalog(load_prompt_entries()))
    print(f"wrote {CATALOG}")
    return 0


def cmd_check(args) -> int:
    findings = run_all()
    if findings:
        print(f"prompt-check: {len(findings)} warning(s):")
        for f in findings:
            print(f"  - {f.message}")
    else:
        print("prompt-check: clean")
    return 0  # NON-BLOCKING


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="tools.prompts")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("index")
    sub.add_parser("check")
    args = parser.parse_args(argv)
    return {"index": cmd_index, "check": cmd_check}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
```

Add to `Makefile` (near `glossary-check`):

```makefile
.PHONY: prompt-index
prompt-index: ## Regenerate docs/prompts/index.md (probabilistic-components catalog)
	@$(PYTHON) -m tools.prompts index

.PHONY: prompt-check
prompt-check: ## Reconcile the prompt registry vs glossary + code consumers (non-blocking)
	@$(PYTHON) -m tools.prompts check
```

- [ ] **Step 4: Run test + smoke**

Run: `~/.pyenv/shims/python -m pytest tests/prompts/test_cli.py -v` → PASS
Run: `~/.pyenv/shims/python -m tools.prompts check` → exit 0; findings expected (missing metadata + catalog out of sync + orphan task/domain) until Task 7.

- [ ] **Step 5: Commit**

```bash
git add tools/prompts/__main__.py Makefile tests/prompts/test_cli.py
git commit -m "feat(prompts): CLI (index/check) + make targets"
```

---

### Task 7: Metadata backfill + catalog generation

**Files:**
- Modify: `prompts/core_extractors.yaml`, `prompts/ingestion_prompts.yaml`, `prompts/ask_prompts.yaml`, `prompts/lens_meeting_minutes.yaml`, `prompts/lens_persona.yaml` (add `used_for` + `audience` to each entry)
- Create (generated): `docs/prompts/index.md`

- [ ] **Step 1: Add `used_for` + `audience` to each live prompt entry**

For every prompt entry in the **live** files, add the two keys (leave `prompt:` untouched). Guidance:

| file | keys | `used_for` | `audience` |
|---|---|---|---|
| `core_extractors.yaml` | function_type, structure_type, purpose | `[classification]` | `[enrichment]` |
| | topic_level_1, topic_level_3 | `[classification]` | `[enrichment]` |
| | overall_keywords, domain_keywords | `[extraction]` | `[enrichment]` |
| | entity_mentions, claims | `[extraction]` | `[enrichment]` |
| | topic_segments | `[segmentation]` | `[enrichment]` |
| `ingestion_prompts.yaml` | speaker_window, stitch_window | `[ingestion]` | `[ingestion]` |
| `ask_prompts.yaml` | ask_synthesis | `[synthesis]` | `[ask]` |
| `lens_meeting_minutes.yaml`, `lens_persona.yaml` | all | `[lens]` | `[lens]` |

`audience` here lists the **derived internal role** so `check_audience_vs_consumers` is satisfied (declared == consumed). Leave `task_prompts.yaml` / `domain_prompts.yaml` **without** metadata — the orphan check flags them as unused (intended).

- [ ] **Step 2: Generate the catalog + reconcile**

```bash
make prompt-index          # writes docs/prompts/index.md
make prompt-check          # iterate until clean except the intended orphan findings
```
Expected end state: **no values-drift** (glossary was fixed in Task 2), **no audience mismatch** (declared == derived), **no missing-metadata**; the only remaining finding is the informational **orphan** `task_prompts.yaml` (legacy — leave it; do not delete). Note: `domain_prompts.yaml` has no `prompt:` entries (just a `domain_keywords` list), so it produces no registry entries and is simply absent from the catalog — not flagged. Note the task_prompts orphan in the commit.

- [ ] **Step 3: Commit**

```bash
git add prompts/ docs/prompts/index.md
git commit -m "feat(prompts): backfill used_for/audience metadata + generated catalog (task/domain flagged legacy)"
```

---

## Final verification

- [ ] `~/.pyenv/shims/python -m pytest tests/glossary/ tests/prompts/ -v` — all green.
- [ ] `make glossary-check` — clean (claim-kind reconciles vs the Literal).
- [ ] `make prompt-check` — clean except the intended `task_prompts.yaml` orphan warning.
- [ ] `make prompt-index` / `make glossary-index` then `git status` — both generated files regenerate identically.
- [ ] `make cli-index` — regenerate the CLI catalog to include `prompt-*` (and confirm `make cli-check`, `adr-check` clean).
