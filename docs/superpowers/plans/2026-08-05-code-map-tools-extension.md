# Code-Map Extension to `tools/` — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the code map (`docs/code/`, `tools/code/`) to cover the 9 `tools/` packages, so the map spans all the repo's Python and Round B can link operations capabilities to their tool code.

**Architecture:** A focused change to `tools/code/reader.py` (teach `packages()`, `_files_of()`, and the import scan to see `tools/` alongside `src/`, slugging tool units `tools.<pkg>`), plus 9 authored `docs/code/tools.<pkg>.md` nodes (`role: tooling`). The code domain's guard is unchanged — the wider unit set flows through the existing checks.

**Tech Stack:** Python 3 stdlib, pytest, Make.

## Global Constraints

- **Non-blocking, always:** no reader function raises; `make code-check` stays exit 0.
- **Interpreter:** `~/.pyenv/shims/python`. **Run tests:** `~/.pyenv/shims/python -m pytest <path> -p no:cacheprovider -q -o addopts=""`.
- **Slug convention:** `src` units stay **bare** (`api`, `enrichment`); `tools/` units are **`tools.<pkg>`** (`tools.adr`). No collision; existing product-capability `implemented_by` links are untouched.
- **`tools.` prefix disambiguates** a tools *package* from a src *key-module* (both contain a `.`): `tools.adr` → the `tools/adr/` package dir; `ingestion.orchestrator` → the one `src/ingestion/orchestrator.py` file.
- **Package-level only** for tools (no tool key-module nodes this round).
- DRY, YAGNI, TDD, frequent commits.

---

### Task 1: Extend the reader to `tools/`

**Files:** Modify `tools/code/reader.py`; Test `tests/code/test_reader_tools.py` (new)

**Interfaces:**
- Consumes: nothing new.
- Produces (changed behaviour): `packages(root)` now returns bare `src` packages + `tools.<pkg>`; `_files_of` resolves `tools.*` to the tools package dir; `dep_edges`/`dep_edges_for_module` derive edges across both trees via a new `_dep_slug` helper.

- [ ] **Step 1: Write the failing test**

```python
# tests/code/test_reader_tools.py
import os
from tools.code.reader import packages, dep_edges, _files_of


def _w(p, text=""):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "w", encoding="utf-8").write(text)


def _fixture(tmp):
    _w(str(tmp / "src/a/__init__.py"))
    _w(str(tmp / "tools/y/__init__.py"))
    _w(str(tmp / "tools/x/reader.py"), "from tools.y import z\nfrom src.a import q\n")
    _w(str(tmp / "tools/x/__init__.py"))


def test_packages_includes_tools_prefixed(tmp_path):
    _fixture(tmp_path)
    pkgs = set(packages(str(tmp_path)))
    assert "a" in pkgs and "tools.x" in pkgs and "tools.y" in pkgs


def test_files_of_resolves_tools_package(tmp_path):
    _fixture(tmp_path)
    files = _files_of("tools.x", str(tmp_path))
    assert any(f.endswith(os.path.join("tools", "x", "reader.py")) for f in files)


def test_dep_edges_tool_to_tool_and_tool_to_src(tmp_path):
    _fixture(tmp_path)
    edges = dep_edges(str(tmp_path))
    assert edges["tools.x"] == ["a", "tools.y"]  # tool->src (a) + tool->tool (tools.y)


def test_bare_src_unit_edges_unchanged(tmp_path):
    # a src package importing another src package still yields a bare edge
    _w(str(tmp_path / "src/a/__init__.py"), "from src.b import q\n")
    _w(str(tmp_path / "src/b/__init__.py"))
    assert dep_edges(str(tmp_path))["a"] == ["b"]
```

- [ ] **Step 2: Run to verify fail** — `test_packages_includes_tools_prefixed` fails (`tools.x` absent); `_files_of` import may fail if not exported (it's module-level, importable).

- [ ] **Step 3: Implement** — edit `tools/code/reader.py`:

Replace the `_IMPORT` line and add a helper:

```python
_IMPORT = re.compile(r"(?:from|import)\s+(src|tools)\.(\w+)")


def _dep_slug(match) -> str:
    tree, name = match.group(1), match.group(2)
    return name if tree == "src" else f"tools.{name}"
```

Replace `packages`:

```python
def packages(root: str = ".") -> List[str]:
    out = []
    for tree, prefix in (("src", ""), ("tools", "tools.")):
        base = os.path.join(root, tree)
        if os.path.isdir(base):
            for name in sorted(os.listdir(base)):
                if os.path.isdir(os.path.join(base, name)) and name != "__pycache__":
                    out.append(f"{prefix}{name}")
    return out
```

Replace `_files_of`:

```python
def _files_of(unit: str, root: str) -> List[str]:
    # tools package -> all its .py; src package -> all its .py; src dotted module -> that one file
    if unit.startswith("tools."):
        pkg = unit.split(".", 1)[1]
        return glob.glob(os.path.join(root, "tools", pkg, "**", "*.py"), recursive=True)
    if "." in unit:
        return [os.path.join(root, "src", *unit.split(".")) + ".py"]
    return glob.glob(os.path.join(root, "src", unit, "**", "*.py"), recursive=True)
```

In `dep_edges`, change the match handling from `dep = m.group(1)` to:

```python
            for m in _IMPORT.finditer(text):
                dep = _dep_slug(m)
                if dep != pkg and dep in edges:
                    edges[pkg].add(dep)
```

Replace `dep_edges_for_module` (skip tools packages — they're handled by `dep_edges`; use `_dep_slug`):

```python
def dep_edges_for_module(unit: str, root: str) -> List[str]:
    if "." not in unit or unit.startswith("tools."):
        return []
    valid = set(packages(root))
    parent = unit.split(".")[0]
    deps = set()
    for f in _files_of(unit, root):
        try:
            t = open(f, encoding="utf-8", errors="ignore").read()
        except Exception:
            continue
        for m in _IMPORT.finditer(t):
            dep = _dep_slug(m)
            if dep != parent and dep in valid:
                deps.add(dep)
    return sorted(deps)
```

(`io_of` and `load_units` need no change — `io_of` already scans `_files_of`, which now resolves tools units.)

- [ ] **Step 4: Run tests** — new file 4 passed; then the whole existing suite to confirm no regression:

Run: `~/.pyenv/shims/python -m pytest tests/code/ -p no:cacheprovider -q -o addopts=""`
Expected: all green (existing tmp-fixture tests have no `tools/` dir, so `packages()` is unchanged for them).

- [ ] **Step 5: Commit**

```bash
git add tools/code/reader.py tests/code/test_reader_tools.py
git commit -m "feat(code): extend the map reader to tools/ (tools.<pkg> units, cross-tree edges)"
```

---

### Task 2: Backfill the 9 `tools.*` nodes + regenerate

**Files:** Create `docs/code/tools.<pkg>.md` × 9 + regenerated `docs/code/index.md`, `docs/code/pipeline.md`

- [ ] **Step 1: List what needs nodes**

```bash
~/.pyenv/shims/python -c "from tools.code.reader import packages, dep_edges, io_of; \
tp=[p for p in packages('.') if p.startswith('tools.')]; \
[print(p, 'deps=', dep_edges('.').get(p), 'io=', io_of(p,'.')) for p in tp]"
```

Expect the 9: `tools.adr, tools.api, tools.capability, tools.cli, tools.code, tools.glossary, tools.graphq, tools.knowledge, tools.prompts` — with derived deps (most depend on `tools.code` and/or `ingestion` via `src.ingestion.front_matter`).

- [ ] **Step 2: Author one node per tools package** — `role: tooling`, real `key_modules`, a one-line description. e.g.:

```markdown
<!-- docs/code/tools.capability.md -->
---
type: CodeUnit
unit: tools.capability
role: tooling
key_modules: [reader, render, check]
---
The capabilities domain: reads Capability nodes, renders the catalogue, and reconciles implemented_by links + coverage against the code map.
```

```markdown
<!-- docs/code/tools.knowledge.md -->
---
type: CodeUnit
unit: tools.knowledge
role: tooling
key_modules: [check]
---
The knowledge-graph disclosure layer: the spec/plan honesty-check nudge + the cascade-coverage / addendum-presence guard.
```

The 9 nodes and what each domain does (mine `tools/<pkg>/` + the matching `docs/<domain>/` bundle for accurate one-liners): `tools.adr` (ADR corpus + code-links), `tools.api` (HTTP-surface catalogue vs openapi.json), `tools.capability` (capability map), `tools.cli` (CLI-surface catalogue), `tools.code` (the code map itself — reader/render/check), `tools.glossary` (vocabulary pinned to enums), `tools.graphq` (Neo4j read-query registry), `tools.knowledge` (cascade + honesty check), `tools.prompts` (probabilistic-components registry). Set `key_modules` from each package's real module files (e.g. `reader, render, check`; `tools.adr` adds `model, index, code_links, intent, scaffold`).

- [ ] **Step 3: Generate + reconcile**

```bash
make code-index                              # writes docs/code/index.md + pipeline.md (now incl. tools)
~/.pyenv/shims/python -m tools.code check     # iterate until: code-check: clean
```

`clean` = every package (src + the 9 tools) has a node, every node classified, index + pipeline in sync, no stale nodes.

- [ ] **Step 4: Verify the operations architecture rendered**

```bash
~/.pyenv/shims/python -c "t=open('docs/code/pipeline.md').read(); \
print('tools.capability --> tools.code' in t, '| tooling group in index:', '## tooling' in open('docs/code/index.md').read())"
```

Expect a `## tooling` group in the index and at least one `tools.* --> tools.code` edge in the pipeline.

- [ ] **Step 5: Commit**

```bash
git add docs/code/
git commit -m "docs(code): backfill 9 tools.* nodes (role tooling) + regenerated map incl. operations architecture"
```

---

## Final verification

- [ ] `~/.pyenv/shims/python -m pytest tests/code/ -p no:cacheprovider -q -o addopts=""` — all green.
- [ ] `make code-check` — clean (25 units now: 16 src packages + 9 tools + the 14 src key modules all covered/classified/in-sync).
- [ ] `make code-index` then `git status` — `docs/code/index.md` + `pipeline.md` regenerate identically.
- [ ] `make knowledge-check` — clean (this spec + plan carry `## Knowledge-graph check` addenda).
- [ ] Open `docs/code/pipeline.md` — the `tools.*` nodes and their tool→tool / tool→`src` edges render (the operations architecture).

## Knowledge-graph check

Reviewed against `docs/index.md` on 2026-08-05.

| domain | touched? | consulted / reconciled | notes |
| --- | --- | --- | --- |
| code | yes | the domain being extended — reader change + 9 tools nodes; `code-check` clean | `src/` untouched |
| capabilities | no | — | operations caps consuming these nodes are Round B |
| cli / adr / glossary / api / prompts / graph-queries | no | — | no new targets, decision, vocabulary, surface, prompt, or query |

**Verdict:** reconciled — only the code domain is touched (it is the subject); Round B carries the capability + ADR reconciliation.
