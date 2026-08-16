# Hierarchical code intake — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Derive the real code (packages + modules) into the graph from `src/`/`tools/`, retire the hand-authored `docs/code/*.md` overlay, add `contains`/`contained_by` hierarchy and module-level `depends_on`, and surface derived `category`/`determinism` axes — so any code node can `walk` up to the architecture that governs it.

**Architecture:** `tools/code/reader.py` stops globbing `docs/code/` and instead walks the source trees, emitting one `CodeUnit` per package and per module with a `level` axis, docstring context, and module-granular imports. Hierarchy is a new derived `contains` edge computed from the node-id set. Cross-domain `category`/`determinism` are computed post-harvest in a new `tools/graph/classify.py`. Downstream consumers (`tools/graph/reader.py` resolvers, `tools/capability/reader.py::real_code_units`, `tools/corpus`) resolve against the derived node set. The 48 overlay files are deleted; `docs/code/index.md` + `pipeline.md` remain as generated catalogs.

**Tech Stack:** Python 3 (stdlib: `ast`, `os`, `re`), pytest. No new deps.

**Spec:** `docs/superpowers/specs/2026-08-16-hierarchical-code-intake-design.md`.
**ADRs:** new ADR (Task 5) refines ADR-0019 + ADR-0024, extends ADR-0020; ADR-0016/0023 govern non-blocking checks.

## Global Constraints

- **Node ids (verbatim scheme):** a package = a directory under `src/`/`tools/` that *directly contains at least one `.py` file*; a module = a non-`__init__` `.py` file. Ids are dotted paths with the `src/` prefix stripped and the `tools.` prefix kept: package `api`, sub-package `api.routers`, module `api.routers.segments`, tool package `tools.graph`, tool module `tools.graph.traverse`. A top-level `src/*.py` (e.g. `config`, `main`, `tasks`, `celery_app`, `run_projection_service`) is a **module** with a bare id (`config`). The `src/` and `tools/` roots are **not** nodes.
- **`level` axis:** every `CodeUnit` carries `level ∈ {"package", "module"}`. `symbol` is reserved (NOT this phase).
- **`depends_on` is module-granular:** derived from AST-style import lines matching `(?:from|import)\s+((?:src|tools)\.[\w.]+)`, each resolved to the **longest prefix of the dotted target that is an existing node id** (module preferred, else the enclosing package). Only `level == "module"` nodes carry `depends_on`.
- **Context = docstring only:** `ast.get_docstring` of the module (or the package's `__init__.py`). No authored descriptions.
- **Additive to existing edges:** after the flip, every pre-existing `implements` / `governs` / `verifies` / `consumed_by` endpoint MUST still resolve — `python -m tools.graph check` reports no `does not resolve` finding.
- **Non-blocking:** every `check_*` returns `List[Finding]`; every domain CLI still `return 0` (ADR-0016/0023).
- **Overlay retirement:** the `docs/code/*.md` **unit** files are deleted; `docs/code/index.md` + `docs/code/pipeline.md` remain and are GENERATED from the derived nodes.
- **Freshness:** after Task 2 onward, `make regen-derived && git diff --exit-code` is CLEAN (regenerate + commit the catalogs in the task that changes their render).
- **Names verbatim:** `discover_units(root=".")`, `contains_edges(root=".")`, `dep_edges(root=".")`, `load_units(root=".")`, `derive_axes(root=".")`, `check_missing_docstring(units)`.

---

### Task 1: New derivation primitives (additive) in `tools/code/reader.py`

Add the source-walking discovery, module-level import resolution, and hierarchy — **without** changing `load_units` yet, so every existing consumer keeps its current behavior and the suite stays green.

**Files:**
- Modify: `tools/code/reader.py` (add `level` to `CodeUnit`; add `discover_units`, `contains_edges`, and their helpers)
- Test: `tests/code/test_discover.py` (new)

**Interfaces:**
- Consumes: `ast`, `os`, `re` (stdlib).
- Produces:
  - `CodeUnit` gains `level: str = ""` (defaulted, so the existing `load_units` still constructs it).
  - `discover_units(root=".") -> List[CodeUnit]` — package + module nodes, each with `unit`, `level`, `description` (docstring), `path`, and (modules only) `depends_on`.
  - `contains_edges(root=".") -> List[Tuple[str, str]]` — `(parent_id, child_id)` pairs for the hierarchy.

- [ ] **Step 1: Write the failing test** — `tests/code/test_discover.py`:

```python
import os

from tools.code.reader import CodeUnit, contains_edges, discover_units


def _w(path, text=""):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w", encoding="utf-8").write(text)


def _fixture(tmp):
    # a package with a sub-package and modules; a tools package; a top-level src module
    _w(str(tmp / "src/api/__init__.py"), '"""The API surface."""\n')
    _w(str(tmp / "src/api/main.py"), "from src.api.routers import segments\n")
    _w(str(tmp / "src/api/routers/__init__.py"))
    _w(str(tmp / "src/api/routers/segments.py"),
       '"""Segment routes."""\nfrom src.events import store\n')
    _w(str(tmp / "src/events/__init__.py"))
    _w(str(tmp / "src/events/store.py"), "x = 1\n")
    _w(str(tmp / "src/config.py"), '"""Settings."""\n')
    _w(str(tmp / "tools/graph/__init__.py"))
    _w(str(tmp / "tools/graph/traverse.py"), "from src.config import settings\n")


def test_discovers_packages_and_modules_with_level(tmp_path):
    _fixture(tmp_path)
    by_id = {u.unit: u for u in discover_units(str(tmp_path))}
    assert by_id["api"].level == "package"
    assert by_id["api.routers"].level == "package"
    assert by_id["api.routers.segments"].level == "module"
    assert by_id["config"].level == "module"          # top-level src/*.py is a module
    assert by_id["tools.graph"].level == "package"
    assert by_id["tools.graph.traverse"].level == "module"
    assert "src" not in by_id and "tools" not in by_id  # roots are not nodes


def test_context_comes_from_docstring(tmp_path):
    _fixture(tmp_path)
    by_id = {u.unit: u for u in discover_units(str(tmp_path))}
    assert by_id["api"].description == "The API surface."           # package __init__ docstring
    assert by_id["api.routers.segments"].description == "Segment routes."
    assert by_id["events.store"].description == ""                  # no docstring


def test_module_depends_on_is_dotted_and_longest_prefix(tmp_path):
    _fixture(tmp_path)
    by_id = {u.unit: u for u in discover_units(str(tmp_path))}
    # full dotted resolution to the module, not just the top package:
    assert by_id["api.routers.segments"].depends_on == ["events.store"]
    # 'from src.api.routers import segments' resolves to the sub-package (segments is a name here):
    assert by_id["api.main"].depends_on == ["api.routers"]
    # tools module importing a src top-level module resolves to that module:
    assert by_id["tools.graph.traverse"].depends_on == ["config"]
    # packages carry no depends_on
    assert by_id["api"].depends_on == []


def test_contains_edges_form_the_hierarchy(tmp_path):
    _fixture(tmp_path)
    edges = set(contains_edges(str(tmp_path)))
    assert ("api", "api.routers") in edges
    assert ("api.routers", "api.routers.segments") in edges
    assert ("api", "api.main") in edges
    assert ("tools.graph", "tools.graph.traverse") in edges
    # roots (no parent node) appear as no child: 'config' and 'api' are never a child
    assert not any(child == "config" for _, child in edges)
    assert not any(child == "api" for _, child in edges)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/code/test_discover.py -q --no-cov`
Expected: FAIL — `cannot import name 'discover_units'`.

- [ ] **Step 3: Implement.** In `tools/code/reader.py`, add `import ast` at the top (keep the existing imports). Add `level` to the dataclass — insert the field right after `unit`:

```python
@dataclass
class CodeUnit:
    unit: str
    role: str = ""
    key_modules: List[str] = field(default_factory=list)
    depends_on: List[str] = field(default_factory=list)
    io: List[str] = field(default_factory=list)
    description: str = ""
    path: str = ""
    level: str = ""
```

(Keep the old fields for now — Task 2 removes `role`/`key_modules`/`io`. Adding `level` last keeps the existing positional constructors in `load_units` valid.)

Add the new derivation code (place after the dataclass):

```python
_SRC_TREES = (("src", ""), ("tools", "tools."))
_IMPORT_DOTTED = re.compile(r"^\s*(?:from|import)\s+((?:src|tools)\.[\w.]+)", re.M)


def _docstring(path: str) -> str:
    try:
        tree = ast.parse(open(path, encoding="utf-8", errors="ignore").read())
    except (OSError, SyntaxError):
        return ""
    return (ast.get_docstring(tree) or "").strip()


def _dotted(prefix: str, parts: List[str]) -> str:
    return prefix + ".".join(parts)


def _longest_node_prefix(cand: str, ids: set) -> str:
    parts = cand.split(".")
    for i in range(len(parts), 0, -1):
        pref = ".".join(parts[:i])
        if pref in ids:
            return pref
    return ""


def _module_deps(path: str, self_id: str, ids: set) -> List[str]:
    try:
        text = open(path, encoding="utf-8", errors="ignore").read()
    except OSError:
        return []
    deps = set()
    for m in _IMPORT_DOTTED.finditer(text):
        dotted = m.group(1)
        cand = dotted[len("src."):] if dotted.startswith("src.") else dotted  # keep 'tools.'
        dep = _longest_node_prefix(cand, ids)
        # skip self and own ancestors — the parent chain is the `contains` edge, not a dependency
        if dep and dep != self_id and not self_id.startswith(dep + "."):
            deps.add(dep)
    return sorted(deps)


def discover_units(root: str = ".") -> List[CodeUnit]:
    """Derive package + module CodeUnits from source (src/, tools/).

    A directory that directly contains a .py file is a package; every non-__init__ .py is a
    module. Ids are dotted paths (src/ stripped, tools. kept). Context = the docstring."""
    units: Dict[str, CodeUnit] = {}
    for tree, prefix in _SRC_TREES:
        base = os.path.join(root, tree)
        if not os.path.isdir(base):
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
            pyfiles = sorted(f for f in filenames if f.endswith(".py"))
            if not pyfiles:
                continue
            rel = os.path.relpath(dirpath, base)
            dir_parts = [] if rel == "." else rel.split(os.sep)
            if dir_parts:                                   # package node (root itself is not one)
                pid = _dotted(prefix, dir_parts)
                init = os.path.join(dirpath, "__init__.py")
                units[pid] = CodeUnit(
                    unit=pid, level="package",
                    description=_docstring(init) if os.path.exists(init) else "",
                    path=dirpath + os.sep)
            for f in pyfiles:                               # module nodes
                if f == "__init__.py":
                    continue
                mpath = os.path.join(dirpath, f)
                mid = _dotted(prefix, dir_parts + [f[:-3]])
                units[mid] = CodeUnit(
                    unit=mid, level="module",
                    description=_docstring(mpath), path=mpath)
    ids = set(units)
    for u in units.values():
        if u.level == "module":
            u.depends_on = _module_deps(u.path, u.unit, ids)
    return [units[k] for k in sorted(units)]


def contains_edges(root: str = ".") -> List["tuple"]:
    """(parent, child) hierarchy pairs — a package contains its sub-packages and modules."""
    ids = {u.unit for u in discover_units(root)}
    out = []
    for uid in sorted(ids):
        parent = uid.rsplit(".", 1)[0] if "." in uid else ""
        if parent and parent in ids:
            out.append((parent, uid))
    return out
```

- [ ] **Step 4: Run tests to verify pass**

Run: `python -m pytest tests/code/test_discover.py -q --no-cov`
Expected: PASS (4 passed).

- [ ] **Step 5: Sanity-check on the real repo**

Run: `python -c "from tools.code.reader import discover_units, contains_edges; u=discover_units(); print(len(u),'nodes', sum(x.level=='package' for x in u),'pkg', sum(x.level=='module' for x in u),'mod'); print(len(contains_edges()),'contains')"`
Expected: ~240 nodes (~48 packages + ~194 modules), a few hundred `contains` edges. Note the counts.
Run: `python -m pytest tests/code -q --no-cov` — the existing code tests still PASS (nothing else changed).

- [ ] **Step 6: Commit**

```bash
git add tools/code/reader.py tests/code/test_discover.py
git commit -m "feat(code): derive package+module CodeUnits from source (discover_units, contains_edges)"
```

---

### Task 2: Flip the code seam — derive everywhere, retire the obsolete checks

Repoint `load_units`/`dep_edges` at the derived nodes, add the `contains` edge to the graph, fix the two path resolvers, update `real_code_units`, rewrite the code render + code-check, retire the now-broken role-based capability coverage check, and update every affected test. This is the atomic core: after it, the graph is the ~240-node derived model with hierarchy and module deps, and the 48 overlays are dead weight (deleted in Task 4).

**Files:**
- Modify: `tools/code/reader.py` (rewrite `load_units`; rewrite `dep_edges`; delete obsolete machinery)
- Modify: `tools/code/render.py` (render by `level`)
- Modify: `tools/code/check.py` (drop obsolete checks; add `check_missing_docstring`; adapt `check_map_in_sync`)
- Modify: `tools/graph/registry.py` (add the `contains` edge)
- Modify: `tools/graph/reader.py` (add `contains` derivation; rewrite `_unit_dir`, `_unit_of_file`)
- Modify: `tools/capability/reader.py` (`real_code_units` → derived id set; drop `code_nodes`; drop `packages`/`KEY_MODULES` import)
- Modify: `tools/capability/check.py` (re-express `check_coverage` on `level` + `_INFRA_PACKAGES`; drop `code_nodes` usage + `_MANDATORY_ROLES`)
- Rewrite tests: `tests/code/test_reader.py`, `tests/code/test_check.py`, `tests/code/test_render.py`
- Delete test: `tests/code/test_reader_tools.py` (its cases moved to `test_discover.py`)
- Modify tests: `tests/capability/test_reader.py`, `tests/graph/test_verifies_edge.py`, `tests/testmap/test_reader.py`

**Interfaces:**
- Produces: `load_units(root=".") -> List[CodeUnit]` (== `discover_units`); `dep_edges(root=".") -> Dict[str, List[str]]` (module-id → module-granular deps); `contains` edge type in `EDGES`; `check_missing_docstring(units) -> List[Finding]`.
- Consumes (unchanged): `tools.graph.reader._addr`, `Edge`; `tools.code.reader.contains_edges`.

- [ ] **Step 1: Rewrite the code reader.** In `tools/code/reader.py`, **replace** `load_units`, `dep_edges`, and delete the obsolete machinery. The file should keep: imports (`ast`, `os`, `re`, `dataclass`/`field`, `Dict`/`List`, `parse_front_matter` may be dropped), `CodeUnit` (trim fields), the Task-1 helpers + `discover_units` + `contains_edges`, and the two functions below. **Delete** `KEY_MODULES`, `_IMPORT`, `_dep_slug`, `packages`, `_files_of`, `io_of`, `_dep_targets`, `dep_edges_for_module`.

Trim the dataclass to the derived shape:

```python
@dataclass
class CodeUnit:
    unit: str
    level: str = ""                                   # "package" | "module"
    depends_on: List[str] = field(default_factory=list)
    description: str = ""                             # docstring
    path: str = ""
```

Replace `load_units` and `dep_edges`:

```python
def load_units(root: str = ".") -> List[CodeUnit]:
    """The code node registry — derived from source (packages + modules)."""
    return discover_units(root)


def dep_edges(root: str = ".") -> Dict[str, List[str]]:
    """unit id -> module-granular depends_on targets (modules only carry deps)."""
    return {u.unit: u.depends_on for u in load_units(root) if u.depends_on}
```

Remove the now-unused `from src.ingestion.front_matter import parse_front_matter` and `import glob` if nothing else references them.

- [ ] **Step 2: Rewrite the code render.** Replace `tools/code/render.py` entirely:

```python
from __future__ import annotations

from typing import List

from tools.code.reader import CodeUnit


def render_index(units: List[CodeUnit]) -> str:
    lines = ["# Code map", "",
             "Derived from `src/` and `tools/`. See `pipeline.md` for the dependency graph.", ""]
    for level in ("package", "module"):
        rows = sorted((u for u in units if u.level == level), key=lambda u: u.unit)
        if not rows:
            continue
        lines += [f"## {level.capitalize()}s", "", "| unit | depends_on |", "| --- | --- |"]
        for u in rows:
            lines.append(f"| {u.unit} | {', '.join(u.depends_on)} |")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_pipeline(units: List[CodeUnit]) -> str:
    lines = ["# Dependency / pipeline map", "", "```mermaid", "graph LR"]
    for u in sorted(units, key=lambda u: u.unit):
        for dep in u.depends_on:                         # modules only carry deps
            lines.append(f"    {u.unit} --> {dep}")
    lines.append("```")
    return "\n".join(lines) + "\n"
```

- [ ] **Step 3: Rewrite the code check.** Replace `tools/code/check.py` entirely:

```python
# tools/code/check.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List

from tools.code.reader import CodeUnit, load_units
from tools.code.render import render_index, render_pipeline


@dataclass
class Finding:
    message: str


def check_missing_docstring(units: List[CodeUnit]) -> List[Finding]:
    """A module with no docstring has no derivable context — the completeness signal that
    replaces the retired authored `role`/description overlay."""
    return [Finding(f"code: module {u.unit} has no docstring (no derivable context)")
            for u in units if u.level == "module" and not u.description]


def check_map_in_sync(index_path: str, pipeline_path: str, units: List[CodeUnit]) -> List[Finding]:
    findings: List[Finding] = []
    for path, render in ((index_path, render_index), (pipeline_path, render_pipeline)):
        want = render(units)
        have = open(path, encoding="utf-8", errors="ignore").read() if os.path.exists(path) else ""
        if want != have:
            findings.append(Finding(
                f"code: {os.path.basename(path)} out of sync — run make code-index"))
    return findings


def run_all(root: str = ".") -> List[Finding]:
    units = load_units(root)
    findings: List[Finding] = []
    findings += check_missing_docstring(units)
    findings += check_map_in_sync(os.path.join(root, "docs/code/index.md"),
                                  os.path.join(root, "docs/code/pipeline.md"), units)
    return findings
```

(`tools/code/__main__.py` needs no change — it imports `load_units`, `render_index`, `render_pipeline`, `run_all`, all of which still exist with the same names.)

- [ ] **Step 4: Add the `contains` edge to the registry.** In `tools/graph/registry.py`, append one entry to the `EDGES` list (after the `depends_on` entry is natural):

```python
    EdgeType("contains", "contained_by", "CodeUnit", "CodeUnit", "derived",
             field="contains_edges", resolve="id",
             description="Hierarchy: a package contains its sub-packages and modules."),
```

- [ ] **Step 5: Wire `contains` derivation + fix resolvers in the graph reader.** In `tools/graph/reader.py`:

Add to the imports from the code reader:

```python
from tools.code.reader import contains_edges, dep_edges, load_units
```

Add the derivation builder (beside `_derived_deps`):

```python
def _derived_contains(edge: EdgeType, root: str) -> List[Edge]:
    return [Edge(edge.name, _addr("CodeUnit", p), _addr("CodeUnit", c))
            for p, c in contains_edges(root)]
```

Register it in `_DERIVED`:

```python
_DERIVED = {
    "dep_edges": _derived_deps,
    "contains_edges": _derived_contains,
    "verifies_edges": _derived_verifies,
    "gq_consumed_by": _derived_consumers("GraphQuery", "graph_id", load_queries),
    "prompt_consumed_by": _derived_consumers("Prompt", "graph_id", load_prompt_entries),
}
```

Replace `_unit_dir` (uniform dotted → path prefix, works for package or module containment):

```python
def _unit_dir(unit: str) -> str:
    if unit.startswith("tools."):
        return "tools/" + "/".join(unit.split(".")[1:]) + "/"
    return "src/" + "/".join(unit.split(".")) + "/"
```

Replace `_unit_of_file` (a src/tools file path → its **module** node, else its package):

```python
def _unit_of_file(path: str, code_ids: Set[str]) -> List[str]:
    """The code node that owns a src/tools file path (src/events/store.py -> 'events.store')."""
    p = (path or "").replace("\\", "/")
    parts = p.split("/")
    if len(parts) < 2 or parts[0] not in ("src", "tools") or not parts[-1].endswith(".py"):
        return []
    prefix = "tools." if parts[0] == "tools" else ""
    stem = parts[-1][:-3]
    mid_parts = parts[1:-1] + ([stem] if stem != "__init__" else [])
    module_id = prefix + ".".join(mid_parts)
    if module_id in code_ids:
        return [module_id]
    pkg_id = prefix + ".".join(parts[1:-1])
    return [pkg_id] if pkg_id in code_ids else []
```

- [ ] **Step 6: Update the capability domain.** In `tools/capability/reader.py`:
  - Change the import at the top from `from tools.code.reader import KEY_MODULES, load_units, packages` to `from tools.code.reader import load_units`.
  - Replace `real_code_units`:

```python
def real_code_units(root: str = ".") -> set:
    """Valid implemented_by / verifies targets — the derived code node registry (packages + modules)."""
    return {u.unit for u in load_units(root)}
```

  - Delete `code_nodes` (its only caller, `check_coverage`, now takes `load_units` directly).

In `tools/capability/check.py`, **re-express `check_coverage` on the derived `level` axis** (the source-derived replacement for the retired `role`). `role` used `_MANDATORY_ROLES = ("pipeline-layer", "surface", "tooling")` — i.e. every `tools.*` package (tooling) and every top-level `src` package that is *not* infrastructure/model/agent must be claimed by some capability. Without `role`, the exclusion becomes: all `tools.*` top-level packages are mandatory; all top-level `src` packages are mandatory except a small curated infrastructure set. This is one constant replacing 48 per-unit overlays, and it reproduces today's behavior (zero findings — every mandatory package is already claimed).

  - Change the import to drop `code_nodes` and add `load_units`:

```python
from tools.capability.reader import CATEGORIES, load_capabilities, real_code_units
from tools.code.reader import load_units
```

  - Replace the `_MANDATORY_ROLES` constant with the infrastructure denylist:

```python
# Top-level src packages that are infrastructure / model / agent — not expected to trace to a
# capability. The source-derived replacement for the retired per-unit `role` exclusion; add a
# package here (or give it a capability) when the coverage check flags a new infra area.
_INFRA_PACKAGES = frozenset({
    "agents", "models", "events", "persistence", "utils", "io", "commands",
})
```

  - Replace `check_coverage` (it now reads `.level`/`.unit`, and only checks top-level packages — modules and sub-packages inherit their top-level package's coverage):

```python
def check_coverage(caps, units) -> List[Finding]:
    """A product/tooling package that no capability claims. Mandatory scope = every top-level
    tools.* package and every top-level src package except infrastructure (_INFRA_PACKAGES).
    A package is covered if it, or any module/sub-package under it, is implemented_by a capability."""
    claimed = set()
    for c in caps:
        claimed.update(c.implemented_by)
    findings: List[Finding] = []
    for u in units:
        if u.level != "package":
            continue
        is_tool = u.unit.startswith("tools.")
        segs = u.unit.count(".")
        if is_tool and segs != 1:
            continue                              # only top-level tools.<name> packages
        if not is_tool and segs != 0:
            continue                              # only top-level src packages
        if not is_tool and u.unit in _INFRA_PACKAGES:
            continue                              # infrastructure — not expected to trace to a capability
        covered = u.unit in claimed or any(t.startswith(u.unit + ".") for t in claimed)
        if not covered:
            findings.append(Finding(f"capability: package {u.unit} is claimed by no capability"))
    return findings
```

  - In `run_all`, change the coverage line to pass the derived units:

```python
    findings += check_coverage(caps, load_units(root))
```

Rationale to record for the reviewer (owner decision, 2026-08-16): the capability coverage signal is *preserved* but re-expressed on `level` + `_INFRA_PACKAGES` instead of the retired authored `role`. `check_links` (implemented_by targets must resolve) is unaffected and stays.

- [ ] **Step 7: Rewrite the code tests.** Replace `tests/code/test_reader.py` entirely:

```python
import os

from tools.code.reader import CodeUnit, dep_edges, load_units


def _w(path, text=""):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w", encoding="utf-8").write(text)


def test_load_units_is_the_derived_registry(tmp_path):
    _w(str(tmp_path / "src/api/__init__.py"))
    _w(str(tmp_path / "src/api/main.py"), "from src.events import store\n")
    _w(str(tmp_path / "src/events/__init__.py"))
    _w(str(tmp_path / "src/events/store.py"), "x = 1\n")
    by_id = {u.unit: u for u in load_units(str(tmp_path))}
    assert by_id["api"].level == "package"
    assert by_id["api.main"].level == "module"
    assert by_id["api.main"].depends_on == ["events"]   # 'events' is the longest node prefix


def test_dep_edges_are_module_granular(tmp_path):
    _w(str(tmp_path / "src/a/__init__.py"))
    _w(str(tmp_path / "src/a/m.py"), "from src.b import x\n")
    _w(str(tmp_path / "src/b/__init__.py"))
    edges = dep_edges(str(tmp_path))
    assert edges["a.m"] == ["b"]        # keyed by the importing MODULE, not the package
    assert "a" not in edges            # packages carry no depends_on
```

Replace `tests/code/test_render.py` entirely:

```python
from tools.code.reader import CodeUnit
from tools.code.render import render_index, render_pipeline

UNITS = [
    CodeUnit("api", level="package"),
    CodeUnit("api.main", level="module", depends_on=["events", "api.routers"]),
    CodeUnit("events", level="package"),
]


def test_render_index_groups_by_level():
    out = render_index(UNITS)
    assert "## Packages" in out and "## Modules" in out
    assert "api.main" in out and "events, api.routers" in out


def test_render_pipeline_is_mermaid():
    out = render_pipeline(UNITS)
    assert "graph LR" in out
    assert "api.main --> events" in out and "api.main --> api.routers" in out
```

Replace `tests/code/test_check.py` entirely:

```python
from tools.code.reader import CodeUnit
from tools.code.check import check_map_in_sync, check_missing_docstring, run_all
from tools.code.render import render_index, render_pipeline


def test_missing_docstring_flags_modules_only():
    units = [
        CodeUnit("api", level="package", description=""),            # package: not flagged
        CodeUnit("api.main", level="module", description=""),        # module, no docstring: flagged
        CodeUnit("api.ok", level="module", description="Has one."),  # module with docstring: not
    ]
    msgs = " ".join(f.message for f in check_missing_docstring(units))
    assert "api.main" in msgs and "api.ok" not in msgs and "code: module api " not in msgs


def test_map_in_sync_clean_then_drift(tmp_path):
    units = [CodeUnit("api", level="package"),
             CodeUnit("api.main", level="module", depends_on=["events"])]
    idx, pipe = tmp_path / "index.md", tmp_path / "pipeline.md"
    idx.write_text(render_index(units), encoding="utf-8")
    pipe.write_text(render_pipeline(units), encoding="utf-8")
    assert check_map_in_sync(str(idx), str(pipe), units) == []
    drifted = units + [CodeUnit("api.new", level="module", depends_on=["events"])]
    msgs = " ".join(f.message for f in check_map_in_sync(str(idx), str(pipe), drifted))
    assert "index.md" in msgs and "pipeline.md" in msgs


def test_run_all_never_raises_on_empty_root(tmp_path):
    findings = run_all(str(tmp_path))
    assert isinstance(findings, list)
```

Delete the obsolete tools test:

```bash
git rm tests/code/test_reader_tools.py
```

- [ ] **Step 8: Fix the cross-domain tests.** These seed a bare `tools/capability/` directory (no `.py`); the derived model requires a `.py` for it to be a package node.

In `tests/testmap/test_reader.py`, in `_seed`, change the line that creates the tools dir so it contains a file:

```python
    # a real tools package (with a .py) so real_code_units() resolves the target
    (tmp_path / "tools" / "capability").mkdir(parents=True)
    (tmp_path / "tools" / "capability" / "__init__.py").write_text("", encoding="utf-8")
```

In `tests/graph/test_verifies_edge.py`, in `_seed`, do the same — after `(tmp_path / "tools" / "capability").mkdir(parents=True)` add:

```python
    (tmp_path / "tools" / "capability" / "__init__.py").write_text("", encoding="utf-8")
```

(The `docs/code/tools.capability.md` overlay in that fixture is now ignored — the node comes from the source dir. Leave it or delete it; the assertions on `code:tools.capability` still hold because the package node exists.)

In `tests/capability/test_reader.py`:
  - Remove the `code_nodes` import and delete `test_code_nodes_carry_roles` entirely (roles are retired).
  - Replace `test_real_code_units_includes_packages_and_key_modules` with:

```python
def test_real_code_units_covers_packages_and_modules():
    units = real_code_units(".")
    assert "enrichment" in units          # a package
    assert "lens.engine" in units         # a module (src/lens/engine.py)
    assert "ask.reader" in units          # a module
```

In `tests/capability/test_check.py`, the three coverage tests key on the retired `role` — re-express them on `level` + the new scope. Replace `test_coverage_flags_unclaimed_pipeline_unit_but_not_infra`, `test_coverage_parent_package_covers_key_module`, and `test_coverage_now_flags_unclaimed_tooling` with:

```python
def test_coverage_flags_unclaimed_src_package_but_not_infra():
    nodes = [NS(unit="lens", level="package"), NS(unit="utils", level="package")]
    caps = [_cap("x", impl=["ingestion"])]              # claims neither
    msgs = " ".join(f.message for f in check_coverage(caps, nodes))
    assert "lens" in msgs and "utils" not in msgs       # utils is infrastructure (_INFRA_PACKAGES)


def test_coverage_package_covered_by_a_module_claim():
    nodes = [NS(unit="lens", level="package")]
    caps = [_cap("x", impl=["lens.engine"])]            # a module under lens covers the package
    assert check_coverage(caps, nodes) == []


def test_coverage_flags_unclaimed_tooling_package():
    nodes = [NS(unit="tools.adr", level="package"), NS(unit="utils", level="package")]
    caps = [_cap("x", impl=["tools.code"])]
    msgs = " ".join(f.message for f in check_coverage(caps, nodes))
    assert "tools.adr" in msgs and "utils" not in msgs  # tooling mandatory; infra advisory


def test_coverage_ignores_modules_and_subpackages():
    nodes = [NS(unit="lens.engine", level="module"), NS(unit="api.routers", level="package")]
    caps = []                                           # nothing claimed
    # modules and sub-packages are never flagged directly — they inherit the top-level package
    assert check_coverage(caps, nodes) == []
```

(`NS` is the `SimpleNamespace` already imported in that file; the re-expressed check reads only `.level` and `.unit`.)

- [ ] **Step 9: Run the affected suites**

Run: `python -m pytest tests/code tests/capability tests/testmap tests/graph -q --no-cov`
Expected: PASS. If a graph test asserts an old `defined_in` endpoint (`code:events` where the term's source is a specific file), it now resolves to the finer module id (`code:events.store`) — update that assertion to the module id. If any other test encodes the old 48-node model, update it to the derived model. Do not weaken assertions; correct them to the new expected values.

- [ ] **Step 10: Verify no dangling edges on the real graph**

Run: `python -m tools.graph check`
Expected: exit 0, and **no** `edge ... does not resolve` finding (the additive constraint). Reachability findings are advisory. If an `implements`/`governs` endpoint dangles, an existing capability/ADR targets an id that is no longer a node — investigate before proceeding (it should not happen; every old id is a subset of the derived set).

- [ ] **Step 11: Regenerate + commit the catalogs**

Run: `make code-index && make graph-index`
Run: `python -m pytest tests/code -q --no-cov` (map-in-sync clean on the real tree)
Then commit everything:

```bash
git add tools/code tools/graph tools/capability tests/code tests/capability tests/testmap tests/graph docs/code/index.md docs/code/pipeline.md docs/graph/index.md docs/graph/graph.md
git commit -m "feat(code): flip to derived nodes — module deps, contains hierarchy, resolvers; retire role-based checks"
```

---

### Task 3: Derived classification axes (`category` / `determinism`)

Compute the two cross-domain axes post-harvest and surface them in the code catalog.

**Files:**
- Create: `tools/graph/classify.py`
- Modify: `tools/code/render.py` (add axis columns to `render_index`)
- Modify: `tools/code/check.py` (`check_map_in_sync` + `run_all` pass axes)
- Modify: `tools/code/__main__.py` (`cmd_index` passes axes)
- Test: `tests/graph/test_classify.py` (new)
- Modify test: `tests/code/test_render.py`, `tests/code/test_check.py` (axes-aware render signature)

**Interfaces:**
- Produces: `derive_axes(root=".") -> Dict[str, Tuple[str, str]]` — code unit id → `(category, determinism)`.
- Consumes: `tools.graph.reader.harvest`, `tools.code.reader.load_units`, `tools.capability.reader.load_capabilities`.

- [ ] **Step 1: Write the failing test** — `tests/graph/test_classify.py`:

```python
import os

from tools.graph.classify import derive_axes


def _w(path, text=""):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w", encoding="utf-8").write(text)


def _seed(tmp):
    # capability (category=product) implements the 'ask' package
    _w(str(tmp / "docs/capabilities/answer.md"),
       "---\ntype: Capability\nkind: primary\ntier: core\ncategory: product\n"
       "implemented_by: [ask]\n---\nAnswer questions.\n")
    _w(str(tmp / "src/ask/__init__.py"))
    _w(str(tmp / "src/ask/engine.py"), "from src.agents import factory\n")  # depends_on agents
    _w(str(tmp / "src/agents/__init__.py"))
    _w(str(tmp / "src/agents/factory.py"), "x = 1\n")
    _w(str(tmp / "src/events/__init__.py"))
    _w(str(tmp / "src/events/store.py"), "x = 1\n")


def test_category_from_implementing_capability(tmp_path):
    _seed(tmp_path)
    axes = derive_axes(str(tmp_path))
    assert axes["ask"][0] == "product"          # implemented by a product capability
    assert axes["events"][0] == ""              # implemented by nobody -> no category (the signal)


def test_determinism_from_agents_dependency(tmp_path):
    _seed(tmp_path)
    axes = derive_axes(str(tmp_path))
    assert axes["ask.engine"][1] == "probabilistic"   # depends_on agents
    assert axes["events.store"][1] == "deterministic"
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/graph/test_classify.py -q --no-cov`
Expected: FAIL — `No module named 'tools.graph.classify'`.

- [ ] **Step 3: Implement** — `tools/graph/classify.py`:

```python
# tools/graph/classify.py
from __future__ import annotations

from typing import Dict, Tuple

from tools.capability.reader import load_capabilities
from tools.code.reader import load_units
from tools.graph.reader import harvest


def _id(addr: str) -> str:
    return addr.split(":", 1)[1]


def derive_axes(root: str = ".") -> Dict[str, Tuple[str, str]]:
    """code unit id -> (category, determinism), computed from the assembled cross-domain edges.

    category: the category of a capability that `implements` the unit (direct only; a unit no
              capability implements has no category — the reachability signal, not a gap).
    determinism: probabilistic if the unit is consumed_by a Prompt, or depends_on the `agents`
                 package/module; else deterministic."""
    edges = harvest(root)
    cap_category = {c.slug: c.category for c in load_capabilities(root)}

    category: Dict[str, str] = {}
    probabilistic = set()
    for e in edges:
        if e.type == "implements":
            cat = cap_category.get(_id(e.src))
            if cat:
                category.setdefault(_id(e.dst), cat)
        elif e.type == "consumed_by" and e.src.startswith("prompts:"):
            probabilistic.add(_id(e.dst))
        elif e.type == "depends_on" and _id(e.dst).split(".")[0] == "agents":
            probabilistic.add(_id(e.src))

    axes: Dict[str, Tuple[str, str]] = {}
    for u in load_units(root):
        det = "probabilistic" if u.unit in probabilistic else "deterministic"
        axes[u.unit] = (category.get(u.unit, ""), det)
    return axes
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/graph/test_classify.py -q --no-cov`
Expected: PASS (2 passed).

- [ ] **Step 5: Surface the axes in the code catalog.** In `tools/code/render.py`, change `render_index` to accept and print the axes:

```python
def render_index(units, axes=None):
    axes = axes or {}
    lines = ["# Code map", "",
             "Derived from `src/` and `tools/`. See `pipeline.md` for the dependency graph.", ""]
    for level in ("package", "module"):
        rows = sorted((u for u in units if u.level == level), key=lambda u: u.unit)
        if not rows:
            continue
        lines += [f"## {level.capitalize()}s", "",
                  "| unit | category | determinism | depends_on |", "| --- | --- | --- | --- |"]
        for u in rows:
            cat, det = axes.get(u.unit, ("", ""))
            lines.append(f"| {u.unit} | {cat} | {det} | {', '.join(u.depends_on)} |")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
```

- [ ] **Step 6: Pass axes through check + CLI.** In `tools/code/check.py`:
  - Add `from tools.graph.classify import derive_axes` to the imports.
  - Change `check_map_in_sync` to take `axes` and hand it to `render_index` (pipeline render is unchanged):

```python
def check_map_in_sync(index_path, pipeline_path, units, axes=None) -> List[Finding]:
    findings: List[Finding] = []
    renders = ((index_path, lambda u: render_index(u, axes)), (pipeline_path, render_pipeline))
    for path, render in renders:
        want = render(units)
        have = open(path, encoding="utf-8", errors="ignore").read() if os.path.exists(path) else ""
        if want != have:
            findings.append(Finding(
                f"code: {os.path.basename(path)} out of sync — run make code-index"))
    return findings
```

  - In `run_all`, compute axes and pass them:

```python
def run_all(root: str = ".") -> List[Finding]:
    units = load_units(root)
    axes = derive_axes(root)
    findings: List[Finding] = []
    findings += check_missing_docstring(units)
    findings += check_map_in_sync(os.path.join(root, "docs/code/index.md"),
                                  os.path.join(root, "docs/code/pipeline.md"), units, axes)
    return findings
```

In `tools/code/__main__.py`, `cmd_index`: add `from tools.graph.classify import derive_axes` and change the render call to `render_index(units, derive_axes())`.

- [ ] **Step 7: Update the two code tests for the axes signature.**

In `tests/code/test_render.py`, update `test_render_index_groups_by_level` to pass axes and assert a column value:

```python
def test_render_index_groups_by_level():
    axes = {"api": ("product", "deterministic")}
    out = render_index(UNITS, axes)
    assert "## Packages" in out and "## Modules" in out
    assert "api.main" in out and "events, api.routers" in out
    assert "| api | product | deterministic |" in out
```

In `tests/code/test_check.py`, `test_map_in_sync_clean_then_drift` still works (axes defaults to `None` → empty). Add one line asserting axes drift is caught:

```python
    # an axis change alone (same units) also drifts the index
    assert any("index.md" in f.message
               for f in check_map_in_sync(str(idx), str(pipe), units, {"api.main": ("product", "deterministic")}))
```

- [ ] **Step 8: Run suites + regenerate + commit**

Run: `python -m pytest tests/code tests/graph -q --no-cov` → PASS.
Run: `make code-index` (now writes the axis columns), then `python -m tools.code check` → exit 0, map-in-sync clean.

```bash
git add tools/graph/classify.py tools/code/render.py tools/code/check.py tools/code/__main__.py tests/graph/test_classify.py tests/code/test_render.py tests/code/test_check.py docs/code/index.md
git commit -m "feat(code): derived category/determinism axes surfaced in the code catalog"
```

---

### Task 4: Retire the overlay + drop CodeUnit from the corpus

Delete the 48 hand-authored unit files (the source is now truth) and stop treating `CodeUnit` as an OKF document type.

**Files:**
- Delete: `docs/code/*.md` **except** `index.md` and `pipeline.md`
- Modify: `tools/corpus/model.py` (drop `CodeUnit` from `OKF_HOMES` + fix the comment)
- Modify test: `tests/corpus/test_model.py` (`OKF_HOMES` no longer has `CodeUnit`)

**Interfaces:** none new. `OKF_HOMES` drops one key.

- [ ] **Step 1: Update the corpus model.** In `tools/corpus/model.py`, remove the `"CodeUnit": "docs/code",` line from `OKF_HOMES`, and update the comment block above it to read that code is now **derived from source** (not an OKF document type):

```python
# OKF document types → their expected home directory (repo-relative). A record of type X
# found outside its home is "misfiled". Code is NOT a document type — it is derived from
# source (src/, tools/) by tools/code/reader.py, not authored as markdown (see the
# hierarchical-code-intake ADR). These four all carry `type:` frontmatter today.
OKF_HOMES: Dict[str, str] = {
    "ADR": "docs/adr",
    "Capability": "docs/capabilities",
    "UseCase": "docs/use-cases",
    "Term": "docs/glossary",
}
```

- [ ] **Step 2: Update the corpus test.** In `tests/corpus/test_model.py`, update `test_okf_homes_cover_the_five_document_types` — rename it and drop the `CodeUnit` line:

```python
def test_okf_homes_cover_the_document_types():
    assert OKF_HOMES == {
        "ADR": "docs/adr",
        "Capability": "docs/capabilities",
        "UseCase": "docs/use-cases",
        "Term": "docs/glossary",
    }
```

- [ ] **Step 3: Run to verify it fails, then delete the overlays.**

Run: `python -m pytest tests/corpus/test_model.py -q --no-cov` → PASS (model + test now agree).

Delete every `docs/code/*.md` except the two generated catalogs:

```bash
/usr/bin/find docs/code -maxdepth 1 -name '*.md' ! -name 'index.md' ! -name 'pipeline.md' -print
/usr/bin/find docs/code -maxdepth 1 -name '*.md' ! -name 'index.md' ! -name 'pipeline.md' -exec git rm {} +
```

Expected: ~48 files removed; `index.md` + `pipeline.md` remain.

- [ ] **Step 4: Verify nothing regressed.**

Run: `python -m tools.corpus check` → exit 0 (`corpus-check: clean`; no `CodeUnit`/misfiled findings, no unregistered-type findings since no `type: CodeUnit` files remain).
Run: `python -m tools.code check` and `python -m tools.graph check` → both exit 0; the code catalog is unchanged (render already reads derived nodes, so deleting overlays does not change `index.md`).
Run: `python -m pytest tests/corpus tests/code tests/graph -q --no-cov` → PASS.
Run: `make regen-derived && git diff --exit-code` → CLEAN (no generated index changed by the deletion).

- [ ] **Step 5: Commit**

```bash
git add tools/corpus/model.py tests/corpus/test_model.py
git commit -m "feat(corpus): retire the docs/code overlay — code is derived from source, not an OKF document type"
```

---

### Task 5: ADR + freshness + final review

Capture the decision, confirm the freshness gate, and run the whole-branch review.

**Files:**
- Create: a new ADR under `docs/adr/` (via the ADR tool)
- Modify: `docs/adr/index.md`, `docs/adr/log.md` (regenerated by `make adr-index`)

- [ ] **Step 1: Scaffold the ADR.**

```bash
python -m tools.adr new "Code map derived from source, hierarchically; overlay retired"
```

Fill the scaffold:
- **Decision:** the code domain is derived from `src/`/`tools/` as package + module `CodeUnit` nodes (`level` axis), with `contains`/`contained_by` hierarchy and module-granular `depends_on`; `category`/`determinism` are derived post-harvest; the hand-authored `docs/code/*.md` overlay is retired; context comes from docstrings.
- **`refines:`** ADR-0019 (implementation is now genuinely derived, not a hand-authored link) and ADR-0024 (the deferred code-side intake, realized and made hierarchical).
- **`extends:`** ADR-0020 (adds the `contains` edge, the `level` axis, and derived `category`/`determinism` node axes).
- No `supersedes` (no dedicated ADR ever established the overlay — its retirement is a consequence recorded here).
- **`source:`** `docs/superpowers/specs/2026-08-16-hierarchical-code-intake-design.md`.

- [ ] **Step 2: Regenerate the ADR index + confirm no drift.**

Run: `make adr-index`
Run: `make adr-check` → reports the new ADR is consistent (schema, bidirectional refine/extend edges). Fix any drift it reports.

- [ ] **Step 3: Full freshness + test gate.**

Run: `make regen-derived && git diff --exit-code` → CLEAN.
Run: `make test-unit` → green.
Run: `python -m tools.code check && python -m tools.graph check && python -m tools.corpus check && python -m tools.capability check` → all exit 0.

- [ ] **Step 4: Commit the ADR.**

```bash
git add docs/adr
git commit -m "docs(adr): code map derived from source, hierarchically; overlay retired (refines 0019/0024, extends 0020)"
```

- [ ] **Step 5: Final whole-branch review.** Dispatch the code-reviewer subagent (superpowers:requesting-code-review) on the most capable model, with a review package for the full branch (`scripts/review-package "$(git merge-base main HEAD)" HEAD`). Then use **superpowers:finishing-a-development-branch**.

## After all tasks

- `make test-unit` green; `make regen-derived && git diff --exit-code` clean.
- `python -m tools.graph check` shows no dangling-endpoint findings (additive constraint held); reachability findings are advisory.
- The graph is ~240 code nodes (packages + modules) with `contains` hierarchy and module `depends_on`; walk-up from any module reaches its package → the ADR that governs it and the capability that implements it.
- The `docs/code/*.md` overlay is gone; `index.md` + `pipeline.md` are generated catalogs carrying the derived `category`/`determinism` axes.

## Knowledge-graph check

Reviewed against `docs/index.md` on 2026-08-16.

| domain | touched? | consulted / reconciled | notes |
| --- | --- | --- | --- |
| code | yes | `load_units` derives packages+modules from source; `contains_edges`; module `dep_edges`; docstring context; render/check rewritten; overlay retired | the subject |
| graph | yes | `contains` edge in registry; `_derived_contains`; `_unit_dir`/`_unit_of_file` rewritten; `classify.derive_axes` | co-subject |
| capabilities | yes | `real_code_units` → derived id set; `check_coverage` re-expressed on `level` + `_INFRA_PACKAGES` (replaces retired `role`); `code_nodes` dropped | consequence of retiring the overlay |
| corpus | yes | `CodeUnit` dropped from `OKF_HOMES` — code is derived, not a document type | — |
| adr | yes | new ADR (refines 0019/0024, extends 0020) | — |
| tests / use-cases / glossary / prompts / graph-queries | no (logic) | edges into code still resolve at package+module grain; `defined_in` now resolves to the finer module node | — |

**Verdict:** reconciled — code + graph are the subjects (code derived hierarchically from source, overlay retired, classifications derived); the capability domain's coverage check is preserved but re-expressed on the derived `level` axis (plus a small `_INFRA_PACKAGES` denylist) since `role` is retired; corpus drops `CodeUnit` as a document type; a new ADR captures the decision.
