# Phase L0 — Corpus substrate (type-primary intake + misfiled check) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the corpus substrate's first slice — a single, type-primary, repo-wide intake that discovers every OKF document by its *own* frontmatter `type:` (not a body match), plus a non-blocking check that flags records living outside their home. This is the foundation the domains later project over.

**Architecture:** A new `tools/corpus/` tool in the established `model → reader → check → CLI` shape. `okf_records()` walks the whole repo (minus an ignore list), parses each `.md`'s top frontmatter via the shared `parse_front_matter`, and emits a `Record` for any file whose parsed `type:` is an OKF document type. `check_misfiled` flags a record whose path is outside its type's home directory. Non-blocking, self-registered like every other domain tool.

**Tech Stack:** Python 3 (stdlib + `src.ingestion.front_matter.parse_front_matter`), pytest, GNU Make. No new deps.

**Spec:** `docs/superpowers/specs/2026-08-15-first-class-knowledge-graph-program-design.md` (Phase L0). **ADRs:** 0024 (corpus substrate primary; type-primary intake) and 0025 (first-class ephemeral graph) — this plan builds the intake half of ADR-0024.

## Global Constraints

- **Discovery keys on the file's OWN top-of-file frontmatter**, parsed by `parse_front_matter` — never a body/line grep. A plan or spec that contains `type: Capability` inside a fenced example MUST NOT be discovered as a record. This is the load-bearing correctness property.
- **Type-primary, repo-wide.** A record is found by what it *is*, anywhere in the repo (minus the ignore list) — not by globbing a home folder. The home folder is only used to judge *misfiled*, never to *find*.
- **Non-blocking.** `corpus-check` prints findings and returns 0, always (ADR-0016/0023 visibility-not-gates).
- **Scope of THIS plan:** OKF *document* intake (the five frontmatter'd types) + the misfiled check. Explicitly **deferred to later plans:** code-derived intake via `# okf:` markers (Test, GraphQuery, Prompt — an explicit-tagging migration, per ADR-0024), migrating the existing domain readers to consume `okf_records` (domains-as-projections), and orphan/reachability checks (L2).
- **OKF document types + homes (verbatim):** `ADR → docs/adr`, `Capability → docs/capabilities`, `UseCase → docs/use-cases`, `CodeUnit → docs/code`, `Term → docs/glossary`. These are the five types that carry `type:` frontmatter today (verified 100% present). Code-derived nodes (Test/GraphQuery/Prompt) are **not** in this plan — they self-declare via `# okf:` markers in a later phase.
- **Names verbatim:** module `tools/corpus`; `Record` (fields `type`, `id`, `path`, `frontmatter`, `body`); `OKF_HOMES`; `okf_records(root=".", ignore=_IGNORE_DIRS)`; `check_misfiled(records)`; `run_all(root=".")`; CLI subcommands `check` and `list`.

---

### Task 1: `Record` model + OKF home registry

**Files:**
- Create: `tools/corpus/__init__.py` (empty), `tools/corpus/model.py`
- Test: `tests/corpus/__init__.py` (empty), `tests/corpus/test_model.py`

**Interfaces:**
- Produces: `Record` dataclass (`type`, `id`, `path`, `frontmatter`, `body`); `OKF_HOMES: Dict[str, str]`.

- [ ] **Step 1: Write the failing test** — `tests/corpus/test_model.py`:

```python
from tools.corpus.model import OKF_HOMES, Record


def test_okf_homes_cover_the_five_document_types():
    assert OKF_HOMES == {
        "ADR": "docs/adr",
        "Capability": "docs/capabilities",
        "UseCase": "docs/use-cases",
        "CodeUnit": "docs/code",
        "Term": "docs/glossary",
    }


def test_record_is_a_plain_value():
    r = Record(type="Capability", id="import-transcripts",
               path="docs/capabilities/import-transcripts.md", frontmatter={"type": "Capability"}, body="…")
    assert r.type == "Capability" and r.id == "import-transcripts"
    assert r.path.endswith("import-transcripts.md")
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/corpus/test_model.py -v --no-cov`
Expected: FAIL — `ModuleNotFoundError: No module named 'tools.corpus'`.

- [ ] **Step 3: Create `tools/corpus/__init__.py`** (empty) and **`tools/corpus/model.py`:**

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

# OKF document types → their expected home directory (repo-relative). A record of type X
# found outside its home is "misfiled". These five all carry `type:` frontmatter today.
# Code-DERIVED nodes (Test, GraphQuery, Prompt) are NOT documents — they self-declare via
# `# okf:` markers in code / YAML keys, handled in a later phase (ADR-0024), not here.
OKF_HOMES: Dict[str, str] = {
    "ADR": "docs/adr",
    "Capability": "docs/capabilities",
    "UseCase": "docs/use-cases",
    "CodeUnit": "docs/code",
    "Term": "docs/glossary",
}


@dataclass
class Record:
    type: str            # the file's OWN frontmatter `type:` — an OKF document type
    id: str              # local id: the file stem
    path: str            # provenance: repo-relative path the record was found at
    frontmatter: dict    # parsed top-of-file frontmatter
    body: str            # content after the frontmatter (the record's claim + context)
```

- [ ] **Step 4: Create `tests/corpus/__init__.py`** (empty) and run tests

Run: `python -m pytest tests/corpus/test_model.py -v --no-cov`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add tools/corpus/__init__.py tools/corpus/model.py tests/corpus/__init__.py tests/corpus/test_model.py
git commit -m "feat(corpus): Record model + OKF home registry"
```

---

### Task 2: Type-primary, repo-wide intake (`okf_records`)

**Files:**
- Create: `tools/corpus/reader.py`
- Test: `tests/corpus/test_reader.py`

**Interfaces:**
- Consumes: `Record`, `OKF_HOMES` (Task 1); `src.ingestion.front_matter.parse_front_matter`.
- Produces: `okf_records(root=".", ignore=_IGNORE_DIRS) -> List[Record]`; `_IGNORE_DIRS: set`.

- [ ] **Step 1: Write the failing tests** — `tests/corpus/test_reader.py`:

```python
import os

from tools.corpus.reader import okf_records


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w", encoding="utf-8").write(text)


def test_discovers_by_frontmatter_not_body(tmp_path):
    root = str(tmp_path)
    # a real capability record (its OWN frontmatter is type: Capability)
    _write(os.path.join(root, "docs/capabilities/import.md"),
           "---\ntype: Capability\n---\nImport transcripts.\n")
    # a PLAN that merely EMBEDS `type: Capability` in a fenced example — NOT a record
    _write(os.path.join(root, "docs/superpowers/plans/p.md"),
           "# A plan\n\nExample frontmatter:\n\n```\ntype: Capability\n```\n")
    recs = okf_records(root)
    assert [(r.type, r.id) for r in recs] == [("Capability", "import")]


def test_finds_misfiled_record_anywhere(tmp_path):
    root = str(tmp_path)
    # a Capability record sitting in the ADR folder — must still be discovered (type-primary)
    _write(os.path.join(root, "docs/adr/stray.md"),
           "---\ntype: Capability\n---\nStray.\n")
    recs = okf_records(root)
    assert [(r.type, r.path.replace(os.sep, "/")) for r in recs] == [
        ("Capability", "docs/adr/stray.md")]


def test_ignore_dirs_and_no_frontmatter(tmp_path):
    root = str(tmp_path)
    _write(os.path.join(root, "node_modules/pkg/readme.md"),
           "---\ntype: Capability\n---\nvendored, must be ignored.\n")
    _write(os.path.join(root, "docs/notes.md"), "# just a note, no frontmatter\n")
    assert okf_records(root) == []
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/corpus/test_reader.py -v --no-cov`
Expected: FAIL — `ModuleNotFoundError: No module named 'tools.corpus.reader'`.

- [ ] **Step 3: Create `tools/corpus/reader.py`:**

```python
from __future__ import annotations

import os
from typing import Iterable, List

from src.ingestion.front_matter import parse_front_matter
from tools.corpus.model import OKF_HOMES, Record

# Directories never scanned for records (vendored, build, VCS, caches, worktrees).
_IGNORE_DIRS = {".git", "node_modules", "__pycache__", ".worktrees", "htmlcov",
                ".pytest_cache", ".mypy_cache", "venv", ".venv", "build", "dist", ".next"}


def _iter_markdown(root: str, ignore) -> Iterable[str]:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in ignore]   # prune in place
        for fn in filenames:
            if fn.endswith(".md"):
                yield os.path.join(dirpath, fn)


def okf_records(root: str = ".", ignore=_IGNORE_DIRS) -> List[Record]:
    """Every OKF document in the repo, discovered by its OWN top-of-file `type:` frontmatter
    (never a body match) and classified by type. This is the type-primary, repo-wide intake:
    a record is found by what it IS, anywhere; its home folder is only used later to judge
    whether it is misfiled."""
    out: List[Record] = []
    for path in sorted(_iter_markdown(root, ignore)):
        try:
            text = open(path, encoding="utf-8", errors="ignore").read()
        except OSError:
            continue
        fm, offset = parse_front_matter(text)
        if not fm or fm.get("type") not in OKF_HOMES:
            continue
        out.append(Record(
            type=fm["type"],
            id=os.path.splitext(os.path.basename(path))[0],
            path=os.path.relpath(path, root),
            frontmatter=fm,
            body=text[offset:],
        ))
    return out
```

- [ ] **Step 4: Run tests to verify pass**

Run: `python -m pytest tests/corpus/test_reader.py -v --no-cov`
Expected: PASS (3 passed) — the plan-with-a-fenced-example is not discovered; the misfiled capability is; ignored dirs and frontmatter-less files yield nothing.

- [ ] **Step 5: Smoke against the real repo**

Run: `python -c "from tools.corpus.reader import okf_records; rs=okf_records(); import collections; print(collections.Counter(r.type for r in rs))"`
Expected: a Counter with all five types present — roughly `Capability` ~54, `ADR` ~25, `UseCase` ~20, `CodeUnit` ~47, `Term` ~111 — and **no** records sourced from `docs/superpowers/` (verify: `python -c "from tools.corpus.reader import okf_records; print([r.path for r in okf_records() if 'superpowers' in r.path])"` prints `[]`).

- [ ] **Step 6: Commit**

```bash
git add tools/corpus/reader.py tests/corpus/test_reader.py
git commit -m "feat(corpus): type-primary, repo-wide OKF intake (frontmatter-keyed, not body)"
```

---

### Task 3: Misfiled check + CLI

**Files:**
- Create: `tools/corpus/check.py`, `tools/corpus/__main__.py`
- Test: `tests/corpus/test_check.py`

**Interfaces:**
- Consumes: `okf_records`, `OKF_HOMES`, `Record`.
- Produces: `Finding` (with `.message`); `check_misfiled(records) -> List[Finding]`; `run_all(root=".") -> List[Finding]`; CLI `python -m tools.corpus check|list`.

- [ ] **Step 1: Write the failing test** — `tests/corpus/test_check.py`:

```python
from tools.corpus.check import check_misfiled
from tools.corpus.model import Record


def _rec(type_, path):
    return Record(type=type_, id="x", path=path, frontmatter={"type": type_}, body="")


def test_clean_when_in_home():
    recs = [_rec("Capability", "docs/capabilities/x.md"), _rec("ADR", "docs/adr/x.md")]
    assert check_misfiled(recs) == []


def test_flags_record_outside_its_home():
    findings = check_misfiled([_rec("Capability", "docs/adr/x.md")])
    assert len(findings) == 1
    assert "misfiled" in findings[0].message and "docs/capabilities" in findings[0].message
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/corpus/test_check.py -v --no-cov`
Expected: FAIL — `ModuleNotFoundError: No module named 'tools.corpus.check'`.

- [ ] **Step 3: Create `tools/corpus/check.py`:**

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from tools.corpus.model import OKF_HOMES
from tools.corpus.reader import okf_records


@dataclass
class Finding:
    message: str


def check_misfiled(records) -> List[Finding]:
    """A record whose path is outside its type's home directory is misfiled — the blind spot
    type-primary intake exists to catch (found by type, judged by home)."""
    out: List[Finding] = []
    for r in records:
        home = OKF_HOMES[r.type]
        p = r.path.replace("\\", "/")
        if not p.startswith(home.rstrip("/") + "/"):
            out.append(Finding(
                f"corpus: {r.type} '{r.id}' is at {p} — outside its home {home}/ (misfiled)"))
    return out


def run_all(root: str = ".") -> List[Finding]:
    return check_misfiled(okf_records(root))
```

- [ ] **Step 4: Create `tools/corpus/__main__.py`:**

```python
from __future__ import annotations

import argparse
import sys

from tools.corpus.check import run_all
from tools.corpus.reader import okf_records


def cmd_check(args) -> int:
    findings = run_all()
    if findings:
        print(f"corpus-check: {len(findings)} warning(s):")
        for f in findings:
            print(f"  - {f.message}")
    else:
        print("corpus-check: clean")
    return 0  # NON-BLOCKING


def cmd_list(args) -> int:
    for r in okf_records():
        print(f"{r.type}\t{r.id}\t{r.path}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="tools.corpus")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    sub.add_parser("list")
    args = p.parse_args(argv)
    return {"check": cmd_check, "list": cmd_list}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Run tests + smoke the CLI on the real repo**

Run: `python -m pytest tests/corpus/test_check.py -v --no-cov`
Expected: PASS (2 passed).
Run: `python -m tools.corpus check`
Expected: `corpus-check: clean` (no records are misfiled today).
Run: `python -m tools.corpus list | wc -l`
Expected: a count in the dozens (all real ADR/Capability/UseCase records).

- [ ] **Step 6: Commit**

```bash
git add tools/corpus/check.py tools/corpus/__main__.py tests/corpus/test_check.py
git commit -m "feat(corpus): misfiled check + CLI (check/list), non-blocking"
```

---

### Task 4: Self-register the tool (code map, capability, Makefile, health)

**Files:**
- Create: `docs/code/tools.corpus.md` (code-map unit)
- Modify: an operations capability's `implemented_by` (or add a capability) so `tools.corpus` is claimed
- Modify: `Makefile` (`corpus-check` target + add `corpus` to the `health` loop)
- Regenerate: `docs/code/index.md`, `docs/capabilities/index.md`, `docs/cli/index.md`, `docs/graph/*`

**Interfaces:** none (wiring only). This is the self-registration every domain tool does; `code-check` and `graph-check` will flag `tools.corpus` as an unclaimed code unit until it is done.

- [ ] **Step 1: Establish the baseline gap**

Run: `python -m tools.code check`
Expected: a finding that `tools.corpus` (a new top-level tool module) is undocumented / unclaimed — confirming the self-registration is required.

- [ ] **Step 2: Add the code-map unit** `docs/code/tools.corpus.md` (match the frontmatter shape of an existing `docs/code/tools.*.md`, e.g. `docs/code/tools.knowledge.md`):

```markdown
---
type: CodeUnit
unit: tools.corpus
role: knowledge-tooling
key_modules: [reader, check, model]
---
The corpus substrate: a single type-primary, repo-wide intake that discovers every OKF
document by its own frontmatter `type:` (ADR-0024), plus a non-blocking misfiled check.
The foundation the domains project over.
```

(Confirm `role:` and `key_modules:` keys against `docs/code/tools.knowledge.md` and adjust to the exact schema that file uses.)

- [ ] **Step 3: Claim it with a capability.** Find the operations capability that covers the knowledge tooling (grep `docs/capabilities` for one whose `implemented_by` lists tool units like `tools.knowledge`/`tools.graph`) and add `tools.corpus` to its `implemented_by` list. If none fits, add a small `type: Capability` file under `docs/capabilities/` in the same shape (category: operations) implemented_by `[tools.corpus]`.

Run to find the target: `grep -rl "tools.knowledge" docs/capabilities/`

- [ ] **Step 4: Add the Makefile target + health loop entry**

Add near the other `*-check` targets:

```makefile
.PHONY: corpus-check
corpus-check: ## Reconcile the OKF corpus: every record discoverable by type, none misfiled (non-blocking)
	@$(PYTHON) -m tools.corpus check
```

And add `corpus` to the `health` loop list (Makefile ~line 95):

```makefile
	@for d in adr cli api glossary prompts graphq code capability knowledge graph usecase testmap corpus; do $(PYTHON) -m tools.$$d check || true; done
```

- [ ] **Step 5: Regenerate indexes and confirm all checks clean-or-advisory**

```bash
make code-index capability-index cli-index graph-index
python -m tools.code check      # tools.corpus now documented — clean (or only pre-existing)
python -m tools.graph check     # clean — no dangling endpoints from the new unit
python -m tools.corpus check    # corpus-check: clean
make cli-check                  # corpus-check catalogued
```

Expected: `tools.corpus` appears in the code map and is claimed by a capability; `code-check`/`graph-check` no longer flag it; `corpus-check` clean; `cli-check` clean.

- [ ] **Step 6: Commit**

```bash
git add docs/code/tools.corpus.md docs/capabilities/ Makefile docs/code/index.md docs/capabilities/index.md docs/cli/index.md docs/graph/
git commit -m "feat(corpus): self-register tool (code unit + capability + corpus-check + health)"
```

---

## After all tasks

Run the full unit suite (`make test-unit`) and confirm green. The decisions are already captured in ADR-0024/0025, so no new ADR this plan. Run the final whole-branch review on the most capable model, then use **superpowers:finishing-a-development-branch**. The next L0 plan migrates an existing domain reader (start with use-cases — smallest, most isolated) to consume `okf_records` instead of globbing its folder, proving domains-as-projections with an equal-node-set regression test.

## Knowledge-graph check

Reviewed against `docs/index.md` on 2026-08-15.

| domain | touched? | consulted / reconciled | notes |
| --- | --- | --- | --- |
| code | yes | new `tools.corpus` unit registered in the code map; `code-check` clean | self-registration |
| capabilities | yes | `tools.corpus` claimed by an operations capability | — |
| cli | yes | `corpus-check` target catalogued → `cli-index`; `cli-check` clean | — |
| graph | yes (read-only) | `graph-check` clean — new unit has no dangling edges | — |
| adr | yes | builds ADR-0024's intake half; no new ADR | — |
| corpus (new) | yes | the subject — type-primary intake + misfiled check | not yet a graph node domain; a substrate tool |
| use-cases / tests / glossary / prompts / graph-queries / api | no | — | migrated in later L0 plans |

**Verdict:** reconciled — `corpus` is the new subject tool; code/capabilities/cli reconciled via self-registration; graph clean. Domain migration onto the substrate is explicitly deferred to later L0 plans.
