# Graph-Links Model — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** A new `tools/graph/` meta-domain: an extensible edge registry + a registry-driven harvester that assembles every domain's typed edges into one cross-domain graph, rendered (catalog + meta-schema + full instance graph) and guarded (`graph-check`, wired into pre-commit).

**Architecture:** `registry.py` (the extensible `EDGES`/`NODE_DOMAINS` data) → `reader.py` (`harvest` → `list[Edge]`, node addressing `<domain>:<id>`) → `render.py` → `check.py` → CLI. Harvests from existing frontmatter/derivations — no new authoring surface. Complements per-domain checks.

**Tech Stack:** Python 3 stdlib + reuse of `tools.code` / `tools.capability` / `tools.adr` readers.

## Global Constraints

- **Non-blocking:** every check returns `list[Finding]`; none raises; CLI `check` returns 0; `graph-check` in pre-commit must not fail a commit.
- **Interpreter:** `~/.pyenv/shims/python`. **Tests:** `~/.pyenv/shims/python -m pytest <path> -p no:cacheprovider -q -o addopts=""`.
- **Registry-driven / extensible:** adding an authored edge on existing node types = one `EdgeType` in `EDGES`, no reader change; `harvest(root, edges=EDGES)` takes the registry as a param so tests prove this. A genuinely new node type adds one adapter entry.
- **Node address = `<domain>:<id>`** via `NODE_DOMAINS`.
- **No new authoring surface**; edges come from existing frontmatter (`implemented_by`, `parent`, `governs`, `supersedes`) + derivations (`dep_edges`).
- DRY, YAGNI.

---

### Task 1: `registry.py` — the extensible edge registry

**Files:** Create `tools/graph/__init__.py` (empty), `tools/graph/registry.py`; Test `tests/graph/test_registry.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/graph/test_registry.py
from tools.graph.registry import EDGES, NODE_DOMAINS, EdgeType, PropSpec


def test_registry_well_formed():
    names = {e.name for e in EDGES}
    assert {"implements", "child_of", "depends_on", "governs", "supersedes"} <= names
    for e in EDGES:
        assert e.from_type in NODE_DOMAINS and e.to_type in NODE_DOMAINS
        assert e.source in ("authored", "derived")


def test_edge_properties_are_supported():
    # the extensibility capacity: an edge type can carry typed properties (e.g. tests)
    e = EdgeType("verifies", "verified_by", "Capability", "CodeUnit", "authored",
                 field="verifies", properties=[PropSpec("test_type", enum=["unit", "integration"])])
    assert e.properties[0].name == "test_type" and "unit" in e.properties[0].enum
```

- [ ] **Step 2: Run to verify fail** — `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

```python
# tools/graph/registry.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class PropSpec:
    name: str
    enum: List[str] = field(default_factory=list)


@dataclass
class EdgeType:
    name: str                       # verb (traceability vocabulary)
    inverse: str                    # "" if none
    from_type: str
    to_type: str
    source: str                     # "authored" | "derived"
    field: str = ""                 # authored: frontmatter key on the from-node; derived: origin tag
    resolve: str = "id"             # id | path  (how a target string maps to a to-node id)
    properties: List[PropSpec] = field(default_factory=list)
    description: str = ""


# Node type -> cascade domain slug (for <domain>:<id> addressing).
# Add a row (+ a reader adapter in reader.py) for a genuinely new node type.
NODE_DOMAINS = {
    "CodeUnit": "code",
    "Capability": "capabilities",
    "ADR": "adr",
    # reserved: GlossaryTerm→glossary, Prompt→prompts, GraphQuery→graph-queries,
    # Spec→spec, Test→test, UseCase→usecase
}

# The extensible edge registry. Adding an authored edge on existing node types is a
# one-entry change here (harvest is registry-driven). Reserved edges (verifies/fulfills)
# are added the same way in their rounds.
EDGES: List[EdgeType] = [
    EdgeType("implements", "implemented_by", "Capability", "CodeUnit", "authored",
             field="implemented_by", resolve="id",
             description="A capability's current implementation reaches toward its intent."),
    EdgeType("child_of", "parent_of", "Capability", "Capability", "authored",
             field="parent", resolve="id",
             description="Decomposition: a narrower intent sits under a broader one."),
    EdgeType("depends_on", "depended_on_by", "CodeUnit", "CodeUnit", "derived",
             field="dep_edges", resolve="id",
             description="Static import dependency between code units."),
    EdgeType("governs", "governed_by", "ADR", "CodeUnit", "authored",
             field="governs", resolve="path",
             description="An architectural decision constrains code under a path."),
    EdgeType("supersedes", "superseded_by", "ADR", "ADR", "authored",
             field="supersedes", resolve="id",
             description="A decision replaces an earlier one."),
]
```

- [ ] **Step 4: Run tests** → PASS (2 passed).
- [ ] **Step 5: Commit** — `git add tools/graph/__init__.py tools/graph/registry.py tests/graph/test_registry.py && git commit -m "feat(graph): extensible edge registry (EdgeType/PropSpec/EDGES, node addressing)"`

---

### Task 2: `reader.py` — registry-driven harvest (the core)

**Files:** Create `tools/graph/reader.py`; Test `tests/graph/test_reader.py`

**Interfaces:**
- Produces: `@dataclass Edge(type, src, dst, props)`; `nodes(root) -> dict[str, set[str]]`; `harvest(root=".", edges=EDGES) -> list[Edge]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/graph/test_reader.py
from tools.graph.reader import harvest, nodes, Edge
from tools.graph.registry import EdgeType


def test_harvest_real_edges_typed_and_addressed():
    edges = harvest(".")
    kinds = {e.type for e in edges}
    assert {"implements", "child_of", "depends_on", "governs", "supersedes"} <= kinds
    impl = [e for e in edges if e.type == "implements"][0]
    assert impl.src.startswith("capabilities:") and impl.dst.startswith("code:")


def test_child_of_direction():
    # the field `parent` lives on the child; the edge points child -> parent
    edges = harvest(".")
    co = [e for e in edges if e.type == "child_of"]
    assert co and all(e.src.startswith("capabilities:") and e.dst.startswith("capabilities:") for e in co)


def test_governs_resolves_path_to_units():
    edges = harvest(".")
    gov = [e for e in edges if e.type == "governs"]
    assert gov and all(e.src.startswith("adr:") and e.dst.startswith("code:") for e in gov)


def test_harvest_is_registry_driven_extensible():
    # a NEW authored edge on existing node types, added only to the passed registry,
    # is harvested with NO reader change (proves extensibility)
    extra = [EdgeType("supersedes", "superseded_by", "ADR", "ADR", "authored",
                      field="supersedes", resolve="id")]
    out = harvest(".", edges=extra)
    assert out and all(e.type == "supersedes" for e in out)


def test_nodes_addressable():
    n = nodes(".")
    assert "CodeUnit" in n and "Capability" in n and "ADR" in n
```

- [ ] **Step 2: Run to verify fail** — no `tools.graph.reader`.

- [ ] **Step 3: Implement**

```python
# tools/graph/reader.py
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Set

from tools.graph.registry import EDGES, NODE_DOMAINS, EdgeType
from tools.capability.reader import load_capabilities
from tools.code.reader import dep_edges, load_units
from tools.adr.index import load_bundle


@dataclass
class Edge:
    type: str
    src: str                 # "<domain>:<id>"
    dst: str                 # "<domain>:<id>"
    props: dict = field(default_factory=dict)


def _addr(node_type: str, node_id) -> str:
    return f"{NODE_DOMAINS[node_type]}:{node_id}"


# --- per-node-type adapter: (load callable, id attribute). Add one for a new node type. ---
_ADAPTERS = {
    "Capability": (load_capabilities, "slug"),
    "CodeUnit": (load_units, "unit"),
    "ADR": (lambda root: load_bundle(os.path.join(root, "docs/adr")), "id"),
}


def nodes(root: str = ".") -> Dict[str, Set[str]]:
    out: Dict[str, Set[str]] = {}
    for t, (load, idattr) in _ADAPTERS.items():
        out[t] = {str(getattr(n, idattr)) for n in load(root)}
    return out


def _unit_dir(unit: str) -> str:
    if unit.startswith("tools."):
        return f"tools/{unit.split('.', 1)[1]}/"
    if "." in unit:                                   # src key module a.b -> its package dir
        return "src/" + "/".join(unit.split(".")[:-1]) + "/"
    return f"src/{unit}/"


def _units_under(path: str, code_ids: Set[str]) -> Set[str]:
    p = path if path.endswith("/") else path + "/"
    return {u for u in code_ids if _unit_dir(u).startswith(p)}


def _authored(edge: EdgeType, root: str, node_ids: Dict[str, Set[str]]) -> List[Edge]:
    load, idattr = _ADAPTERS[edge.from_type]
    out: List[Edge] = []
    for n in load(root):
        src = _addr(edge.from_type, getattr(n, idattr))
        targets = getattr(n, edge.field, None) or []
        if isinstance(targets, (str, int)):
            targets = [targets]
        for t in targets:
            if edge.resolve == "path":
                dsts = _units_under(str(t), node_ids[edge.to_type])
            else:
                dsts = [str(t)]                        # kept even if unresolved — the guard flags it
            for d in dsts:
                out.append(Edge(edge.name, src, _addr(edge.to_type, d)))
    return out


def _derived_deps(edge: EdgeType, root: str) -> List[Edge]:
    return [Edge(edge.name, _addr("CodeUnit", u), _addr("CodeUnit", d))
            for u, deps in dep_edges(root).items() for d in deps]


_DERIVED = {"dep_edges": _derived_deps}                # add a handler for a new derived edge


def harvest(root: str = ".", edges: List[EdgeType] = EDGES) -> List[Edge]:
    node_ids = nodes(root)
    out: List[Edge] = []
    for e in edges:
        if e.source == "authored":
            out += _authored(e, root, node_ids)
        else:
            out += _DERIVED[e.field](e, root)
    return out
```

- [ ] **Step 4: Run tests** → PASS (5 passed). Real-repo smoke: `~/.pyenv/shims/python -c "from tools.graph.reader import harvest; es=harvest('.'); from collections import Counter; print(Counter(e.type for e in es))"` — expect all 5 types with non-trivial counts, no exception.
- [ ] **Step 5: Commit** — `git add tools/graph/reader.py tests/graph/test_reader.py && git commit -m "feat(graph): registry-driven harvest — typed cross-domain edges, <domain>:<id> addressing"`

---

### Task 3: `render.py` — catalog + meta-schema + full instance graph

**Files:** Create `tools/graph/render.py`; Test `tests/graph/test_render.py`

**Interfaces:** `render_catalog(edges, node_ids) -> str` (edge-type table with live counts + node inventory + a Mermaid meta-schema of node-types↔edge-types); `render_graph(edges) -> str` (one Mermaid section per edge type — every instance, the full graph digestibly).

- [ ] **Step 1: Write the failing test**

```python
# tests/graph/test_render.py
from tools.graph.reader import Edge
from tools.graph.render import render_catalog, render_graph

EDGES = [Edge("implements", "capabilities:x", "code:api"),
         Edge("depends_on", "code:api", "code:ui")]
NODES = {"Capability": {"x"}, "CodeUnit": {"api", "ui"}}


def test_catalog_lists_edge_types_and_counts():
    out = render_catalog(EDGES, NODES)
    assert "implements" in out and "depends_on" in out
    assert "```mermaid" in out            # the meta-schema diagram


def test_graph_has_per_edge_type_sections():
    out = render_graph(EDGES)
    assert "## implements" in out and "## depends_on" in out
    assert "capabilities:x" in out and "code:api" in out
    assert out.count("```mermaid") >= 2   # one diagram per edge type
```

- [ ] **Step 2: Run to verify fail.**
- [ ] **Step 3: Implement** — `render.py`. `render_catalog`: a `# Graph` header; a table `| edge | inverse | from → to | source | count |` (count = live instances of that type in `edges`); a node inventory (`type: N`); and a **meta-schema** Mermaid built from the registry (`Capability -->|implements| CodeUnit`, one line per `EdgeType`). `render_graph`: for each edge type present (registry order), a `## <type>` heading + a ```mermaid `graph LR` block with one `src --> dst` line per instance (Mermaid-sanitise `:`/`.` in ids to a safe node id with a label). Both deterministic (sorted). Mirror the string-building style of `tools/code/render.py`.
- [ ] **Step 4: Run tests** → PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat(graph): render — edge catalog + meta-schema + full per-edge-type instance graph"`

---

### Task 4: `check.py` — cross-domain endpoint integrity

**Files:** Create `tools/graph/check.py`; Test `tests/graph/test_check.py`

**Interfaces:** `Finding`; `check_endpoints(edges, node_ids)`; `check_registry(edges_registry, node_domains)`; `check_index_sync(paths, edges, node_ids)`; `run_all(root=".")`.

- [ ] **Step 1: Write the failing test**

```python
# tests/graph/test_check.py
from tools.graph.reader import Edge
from tools.graph.check import check_endpoints, check_registry, run_all
from tools.graph.registry import EdgeType


def test_endpoints_flag_dangling():
    edges = [Edge("implements", "capabilities:x", "code:gone")]
    node_ids = {"Capability": {"x"}, "CodeUnit": {"api"}, "ADR": set()}
    msgs = " ".join(f.message for f in check_endpoints(edges, node_ids))
    assert "code:gone" in msgs


def test_endpoints_clean_when_resolvable():
    edges = [Edge("implements", "capabilities:x", "code:api")]
    node_ids = {"Capability": {"x"}, "CodeUnit": {"api"}, "ADR": set()}
    assert check_endpoints(edges, node_ids) == []


def test_registry_flags_unknown_node_type():
    bad = [EdgeType("weird", "", "Nope", "CodeUnit", "authored")]
    msgs = " ".join(f.message for f in check_registry(bad, {"CodeUnit": "code"}))
    assert "Nope" in msgs


def test_run_all_returns_list_never_raises(tmp_path):
    assert isinstance(run_all(str(tmp_path)), list)
```

- [ ] **Step 2: Run to verify fail.**
- [ ] **Step 3: Implement** — `check.py`:
  - `check_endpoints(edges, node_ids)`: for each edge, split `src`/`dst` on `:` into `(domain, id)`; map domain→node-type via `NODE_DOMAINS` inverse; if the id isn't in `node_ids[type]` → `Finding("graph: edge <type> endpoint <addr> does not resolve")`.
  - `check_registry(edges, node_domains)`: an `EdgeType` whose `from_type`/`to_type` ∉ `node_domains` → finding; an authored edge with no `field` → finding.
  - `check_index_sync(index_path, graph_path, edges, node_ids)`: committed `docs/graph/index.md` + `graph.md` differ from a fresh render → finding (treat missing file as empty).
  - `run_all(root=".")`: `harvest` + `nodes`, run all three (index/graph paths under `docs/graph/`). Non-blocking; guard the `:`-split (an id containing `:` splits once via `split(":", 1)`).
- [ ] **Step 4: Run tests** → PASS. Real-repo smoke: `~/.pyenv/shims/python -c "from tools.graph.check import run_all; print(len(run_all('.')),'findings')"` — expect only index-sync findings pre-generate (docs/graph not written yet), NO exception.
- [ ] **Step 5: Commit** — `git commit -m "feat(graph): guard — cross-domain endpoint integrity + registry + index-sync (non-blocking)"`

---

### Task 5: CLI + Makefile + pre-commit + cascade wiring

**Files:** Create `tools/graph/__main__.py`; Modify `Makefile`, `.githooks/pre-commit`, `tools/knowledge/check.py`, `docs/index.md`; Test `tests/graph/test_cli.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/graph/test_cli.py
import subprocess, sys


def test_check_exits_zero():
    p = subprocess.run([sys.executable, "-m", "tools.graph", "check"], capture_output=True, text=True)
    assert p.returncode == 0 and "graph-check" in p.stdout


def test_neighbors_reports_edges():
    p = subprocess.run([sys.executable, "-m", "tools.graph", "neighbors", "code:api"],
                       capture_output=True, text=True)
    assert p.returncode == 0


def test_graph_in_knowledge_registry():
    from tools.knowledge.check import DOMAINS
    assert "graph" in {slug for slug, _ in DOMAINS}
```

- [ ] **Step 2: Run to verify fail.**
- [ ] **Step 3: Implement**
  - `tools/graph/__main__.py` — `index` (writes `docs/graph/index.md` = `render_catalog`, `docs/graph/graph.md` = `render_graph`; `os.makedirs`), `check` (prints `graph-check: N warning(s)` / `clean`, `return 0`), `neighbors <addr>` (harvest; print inbound + outbound edges of `<addr>`, `return 0`). Mirror `tools/code/__main__.py`.
  - `Makefile` after `code-check`: `graph-index`, `graph-check` (self-documented), and:
    ```makefile
    .PHONY: health
    health: ## Run every domain check + the cross-domain graph check (full sweep)
    	@for d in adr cli api glossary prompts graphq code capability knowledge graph; do $(PYTHON) -m tools.$$d check || true; done
    ```
    (Use the real module names: `graphq` for graph-queries, `capability`, `graph`.)
  - `.githooks/pre-commit` — add a non-blocking line after the adr-check line: `bash scripts/with-project-py.sh tools.graph check || true`.
  - `tools/knowledge/check.py` `DOMAINS` — add `("graph", "graph")`.
  - `docs/index.md` — add a row: `| [graph/](graph/index.md) | cross-domain edge graph (typed links between all domains) | \`make graph-check\` |`.
- [ ] **Step 4: Run test + smoke** — test PASS; `~/.pyenv/shims/python -m tools.graph check` exit 0; `~/.pyenv/shims/python -m tools.knowledge check` → clean (cascade row present).
- [ ] **Step 5: Commit** — `git commit -m "feat(graph): CLI (index/check/neighbors) + make targets + health + pre-commit + cascade wiring"`

---

### Task 6: Self-register the graph domain + generate artifacts + reconcile

**Files:** Create `docs/code/tools.graph.md`, one operations capability child under `docs/capabilities/`, `docs/graph/index.md`, `docs/graph/graph.md`; regenerate `docs/code/index.md`+`pipeline.md`, `docs/capabilities/index.md`, `docs/cli/index.md`

**Why this step exists (found during Task 4):** creating `tools/graph/` makes it a new tool package. Round A's code map now flags `tools.graph` as an uncovered package; once it has a code node, Round B's tooling-coverage flags it as unclaimed by any operations capability; and `graph-check` flags the `code:tools.graph` `depends_on` endpoints as dangling. The graph domain must **self-register** across code + capabilities — dogfooding the very graph it introduces.

- [ ] **Step 1: Register `tools.graph` in the code map** — author `docs/code/tools.graph.md`:
  ```markdown
  ---
  type: CodeUnit
  unit: tools.graph
  role: tooling
  key_modules: [registry, reader, render, check]
  ---
  The cross-domain graph layer: an extensible edge registry + a registry-driven harvester that assembles every domain's typed links into one traversable graph, rendered and guarded.
  ```
  Then `~/.pyenv/shims/python -m tools.code index`; `~/.pyenv/shims/python -m tools.code check` → clean (`tools.graph` covered; `pipeline.md` now shows `tools.graph → tools.adr/capability/code`).
- [ ] **Step 2: Claim it with an operations capability** — author one child under the `maintain-a-guarded-knowledge-graph` primary (mirroring the other operations children — `kind: child`, `parent: maintain-a-guarded-knowledge-graph`, no tier/category), e.g. `docs/capabilities/link-the-domains.md` with `implemented_by: [tools.graph]` and a terse value statement ("Assemble every domain's typed links into one cross-domain graph — traverse and guard it."). Then `~/.pyenv/shims/python -m tools.capability index`; `~/.pyenv/shims/python -m tools.capability check` → clean (`tools.graph` now claimed).
- [ ] **Step 3: Generate the graph artifacts** — `~/.pyenv/shims/python -m tools.graph index` (writes `docs/graph/index.md` + `graph.md`). Note: Steps 1–2 also add a new `implements` edge (`link-the-domains → tools.graph`) and `depends_on` edges from `tools.graph` — that's expected; they now resolve.
- [ ] **Step 4: Reconcile everything clean** — `~/.pyenv/shims/python -m tools.graph check` → `graph-check: clean` (the `code:tools.graph` endpoints now resolve; index in sync). `make cli-index` (catalog `graph-*`/`health`) → `tools.cli check` clean. `tools.knowledge check` clean. Re-run `tools.code check` + `tools.capability check` → both still clean.
- [ ] **Step 5: Verify idempotent** — `docs/graph/index.md` has the catalog + meta-schema; `docs/graph/graph.md` has `## implements` (incl. `link-the-domains → tools.graph`) + `## depends_on` + `## governs` + `## supersedes` + `## child_of`. Re-run all three `index` commands → `git status --short docs/` shows no diff.
- [ ] **Step 6: Commit** — `git add docs/code/ docs/capabilities/ docs/graph/ docs/cli/ && git commit -m "docs(graph): self-register tools.graph (code node + operations capability) + generate graph artifacts + catalog targets"`

---

### Task 7: Capture ADR-0020

- [ ] **Step 1: Scaffold** — `~/.pyenv/shims/python -m tools.adr new "Adopt an OKF-extension typed-edge graph model"`.
- [ ] **Step 2: Fill** — `status: accepted`; `date: 2026-08-06`; `source:` = the spec; `supersedes: []`. Body: base OKF links are untyped/undirected, so typed edges are a sanctioned extension; we adopt a property-graph-shaped, extensible **edge registry** (verb names from the traceability vocabulary; edges carry properties) + `<domain>:<id>` addressing + a registry-driven harvest/render/guard; `graph-check` is a non-blocking cross-domain integrity sweep that **complements** (does not replace) per-domain checks and runs in pre-commit; new edges (tests `verifies`, use-case `fulfills`) are registry additions. Refines the domain-family ADRs; supersedes nothing.
- [ ] **Step 3: Regenerate + verify** — `make adr-index`; `~/.pyenv/shims/python -m tools.adr check` → clean apart from the 3 known warnings.
- [ ] **Step 4: Commit** — `git add docs/adr/ && git commit -m "docs(adr): ADR-0020 — OKF-extension typed-edge graph model"`

---

## Final verification

- [ ] `~/.pyenv/shims/python -m pytest tests/graph/ tests/knowledge/ -p no:cacheprovider -q -o addopts=""` — all green.
- [ ] `make graph-check` — clean; `make graph-index` then `git status` — `docs/graph/*` regenerate identically.
- [ ] `make code-check` + `make capability-check` — clean (`tools.graph` self-registered: a code node + an operations capability claim).
- [ ] `make knowledge-check` + `make cli-check` — clean; `make adr-check` — clean apart from 3 known.
- [ ] `make health` — runs all domain checks + graph-check, exits 0.
- [ ] `.githooks/pre-commit` includes a non-blocking `tools.graph check`.
- [ ] `docs/graph/index.md` (catalog + meta-schema) + `docs/graph/graph.md` (per-edge-type sections incl. the capability→code `implements` view) render on GitHub.
- [ ] `~/.pyenv/shims/python -m tools.graph neighbors code:api` lists the capabilities that implement it, the ADRs that govern it, and the code that depends on it.

## Knowledge-graph check

Reviewed against `docs/index.md` on 2026-08-06.

| domain | touched? | consulted / reconciled | notes |
| --- | --- | --- | --- |
| graph | yes | the new meta-domain | — |
| code / capabilities / adr | yes (read-only) | edges harvested from their readers/frontmatter; no change | reuse existing readers |
| cli | yes | `graph-*` + `health` targets → `cli-index`; `cli-check` clean | — |
| adr | yes | ADR-0020 | — |
| knowledge | yes | cascade row + `DOMAINS` entry; `graph-check` added to `.githooks/pre-commit` | — |
| glossary / api / prompts / graph-queries | no | — | their edges are reserved registry entries |

**Verdict:** reconciled — graph (subject) + code/capabilities/adr (read-only) consulted; cli/adr/knowledge reconciled in the plan.
