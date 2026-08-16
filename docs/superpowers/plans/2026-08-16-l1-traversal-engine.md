# L1 — Traversal engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the knowledge graph a first-class `walk(entry, direction, depth) → Subgraph` primitive (any node/selector, in/out/both, bounded depth or to exhaustion, rebuilt from source, subgraph carrying each node's claim+context), and complete the known node set by adding `GlossaryTerm`, `GraphQuery`, `Prompt`.

**Architecture:** A new `tools/graph/traverse.py` (`walk`, selectors, adjacency, context resolver) over the existing `harvest()` edge list. Three node types are added as registry + `_ADAPTERS` + edge entries in `tools/graph/registry.py` and `tools/graph/reader.py`. CLI gains a `walk` subcommand; `neighbors` becomes its depth-1 shorthand.

**Tech Stack:** Python 3 (stdlib), pytest, GNU Make. No new deps.

**Spec:** `docs/superpowers/specs/2026-08-16-l1-traversal-engine-design.md`. **ADRs:** realizes 0025 (traversal substrate), extends 0020 (reserved node types) — no new ADR.

## Global Constraints

- **Rebuilt from source every call** — `walk` calls `harvest()` fresh; no cache (ADR-0025).
- **Node addresses are `<slug>:<id>`** — slugs from `NODE_DOMAINS` (`code`, `capabilities`, `adr`, `use-cases`, `tests`, and new: `glossary`, `graph-queries`, `prompts`).
- **`walk` never mutates** the graph; it reads `harvest()` and returns a `Subgraph`.
- **All new edges must resolve** (no dangling) — `graph-check` stays clean after each node-type addition.
- **Names verbatim:** `walk(entry, direction="both", depth=None, root=".")`; `Subgraph` (`nodes: Dict[str, Node]`, `edges: List[Edge]`); `Node` (`address`, `type`, `context`); selectors `type:<T>` and `under:<path>`; edges `defined_in` (GlossaryTerm→CodeUnit), `consumed_by` (GraphQuery→CodeUnit, Prompt→CodeUnit).
- **Non-blocking checks stay non-blocking.**

---

### Task 1: `walk` core — Subgraph model + adjacency + BFS

**Files:**
- Create: `tools/graph/traverse.py`
- Test: `tests/graph/test_traverse.py`

**Interfaces:**
- Consumes: `tools.graph.reader.harvest` → `List[Edge]` (`Edge.type`, `.src`, `.dst`, `.props`).
- Produces: `Node`, `Subgraph`, `walk(entry, direction="both", depth=None, root=".") -> Subgraph` (entry = a node address string in this task; selectors added in Task 3; `context` is `""` for now — filled in Task 2).

- [ ] **Step 1: Write the failing test** — `tests/graph/test_traverse.py`:

```python
from tools.graph.traverse import walk, Subgraph
from tools.graph.reader import Edge


def _fake_harvest(edges):
    return lambda root=".": list(edges)


def test_walk_out_depth_1(monkeypatch):
    import tools.graph.traverse as tr
    edges = [Edge("implements", "capabilities:a", "code:x"),
             Edge("depends_on", "code:x", "code:y")]
    monkeypatch.setattr(tr, "harvest", _fake_harvest(edges))
    sg = walk("capabilities:a", direction="out", depth=1)
    assert set(sg.nodes) == {"capabilities:a", "code:x"}          # 1 hop only
    assert [e.dst for e in sg.edges] == ["code:x"]


def test_walk_out_to_exhaustion(monkeypatch):
    import tools.graph.traverse as tr
    edges = [Edge("implements", "capabilities:a", "code:x"),
             Edge("depends_on", "code:x", "code:y")]
    monkeypatch.setattr(tr, "harvest", _fake_harvest(edges))
    sg = walk("capabilities:a", direction="out", depth=None)
    assert set(sg.nodes) == {"capabilities:a", "code:x", "code:y"}  # full chain


def test_walk_in_uses_reverse_edges(monkeypatch):
    import tools.graph.traverse as tr
    edges = [Edge("implements", "capabilities:a", "code:x")]
    monkeypatch.setattr(tr, "harvest", _fake_harvest(edges))
    sg = walk("code:x", direction="in", depth=1)
    assert set(sg.nodes) == {"code:x", "capabilities:a"}


def test_walk_cycle_terminates(monkeypatch):
    import tools.graph.traverse as tr
    edges = [Edge("depends_on", "code:x", "code:y"),
             Edge("depends_on", "code:y", "code:x")]
    monkeypatch.setattr(tr, "harvest", _fake_harvest(edges))
    sg = walk("code:x", direction="out", depth=None)
    assert set(sg.nodes) == {"code:x", "code:y"}


def test_walk_unknown_entry_is_singleton(monkeypatch):
    import tools.graph.traverse as tr
    monkeypatch.setattr(tr, "harvest", _fake_harvest([]))
    sg = walk("code:nope", depth=1)
    assert set(sg.nodes) == {"code:nope"} and sg.edges == []
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/graph/test_traverse.py -q --no-cov`
Expected: FAIL — `ModuleNotFoundError: No module named 'tools.graph.traverse'`.

- [ ] **Step 3: Implement `tools/graph/traverse.py`:**

```python
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from tools.graph.reader import Edge, harvest


@dataclass
class Node:
    address: str        # "<slug>:<id>"
    type: str           # node type name ("Capability", ...); filled by the context pass
    context: str = ""   # claim + context body (Task 2)


@dataclass
class Subgraph:
    nodes: Dict[str, Node] = field(default_factory=dict)
    edges: List[Edge] = field(default_factory=list)


def _adjacency(edges: List[Edge]):
    out = defaultdict(list)   # addr -> list[(neighbor, edge)] following edge direction
    inc = defaultdict(list)   # addr -> list[(neighbor, edge)] against edge direction
    for e in edges:
        out[e.src].append((e.dst, e))
        inc[e.dst].append((e.src, e))
    return out, inc


def walk(entry, direction: str = "both", depth: Optional[int] = None, root: str = ".") -> Subgraph:
    """Materialize the subgraph reachable from `entry` — a node address (selectors: Task 3) —
    following edges `out` | `in` | `both`, to `depth` hops (None = to exhaustion). Rebuilt from
    source each call (harvest())."""
    edges = harvest(root)
    out, inc = _adjacency(edges)
    starts = [entry] if isinstance(entry, str) else list(entry)

    visited = set(starts)
    frontier = deque((s, 0) for s in starts)
    used_edges: List[Edge] = []
    seen_edge = set()

    def _neighbors(addr):
        pairs = []
        if direction in ("out", "both"):
            pairs += out.get(addr, [])
        if direction in ("in", "both"):
            pairs += inc.get(addr, [])
        return pairs

    while frontier:
        addr, d = frontier.popleft()
        if depth is not None and d >= depth:
            continue
        for nbr, e in _neighbors(addr):
            key = (e.src, e.dst, e.type)
            if key not in seen_edge:
                seen_edge.add(key)
                used_edges.append(e)
            if nbr not in visited:
                visited.add(nbr)
                frontier.append((nbr, d + 1))

    # induced edges: only those whose BOTH endpoints are in the visited set
    induced = [e for e in used_edges if e.src in visited and e.dst in visited]
    return Subgraph(nodes={a: Node(address=a, type="") for a in visited}, edges=induced)
```

- [ ] **Step 4: Run tests to verify pass**

Run: `python -m pytest tests/graph/test_traverse.py -q --no-cov`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add tools/graph/traverse.py tests/graph/test_traverse.py
git commit -m "feat(graph): walk() traversal core — adjacency + bounded/exhaustive BFS"
```

---

### Task 2: Claim + context resolution

**Files:**
- Modify: `tools/graph/traverse.py` (add `resolve_context`, wire into `walk`)
- Test: `tests/graph/test_traverse_context.py`

**Interfaces:**
- Produces: `resolve_context(addresses, root=".") -> Dict[str, tuple]` mapping address → `(type, context)`; `walk` fills each `Node.type` and `Node.context`.

- [ ] **Step 1: Write the failing test** — `tests/graph/test_traverse_context.py`:

```python
from tools.graph.traverse import walk


def test_walk_fills_context_for_real_nodes():
    # a real capability node exists in the repo; its context should be non-empty
    sg = walk("capabilities:ask-the-corpus", direction="out", depth=1)
    n = sg.nodes["capabilities:ask-the-corpus"]
    assert n.type == "Capability"
    assert n.context.strip() != ""
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/graph/test_traverse_context.py -q --no-cov`
Expected: FAIL — `n.type` is `""` (context not yet resolved).

- [ ] **Step 3: Add the context resolver** to `tools/graph/traverse.py`. Import the loaders and `NODE_DOMAINS`, build a per-type `(loader, id_attr, body_fn)` table, and resolve addresses in one batched pass:

```python
import os

from tools.graph.registry import NODE_DOMAINS
from tools.capability.reader import load_capabilities
from tools.usecase.reader import load_use_cases
from tools.code.reader import load_units
from tools.adr.index import load_bundle
from tools.testmap.reader import load_tests

# slug -> node type name (inverse of NODE_DOMAINS)
_SLUG_TYPE = {slug: t for t, slug in NODE_DOMAINS.items()}

# node type -> (loader(root)->objs, id attribute, body function(obj)->str)
_CONTEXT = {
    "Capability": (load_capabilities, "slug", lambda o: o.statement),
    "UseCase": (load_use_cases, "slug", lambda o: o.statement),
    "CodeUnit": (load_units, "unit", lambda o: o.description),
    "ADR": (lambda root: load_bundle(os.path.join(root, "docs/adr")), "id",
            lambda o: f"{o.title}\n{o.body}"),
    "Test": (load_tests, "slug",
             lambda o: f"{o.slug} ({o.test_type}) verifies {o.target or o.verifies}"),
}


def resolve_context(addresses, root: str = "."):
    """address -> (type, context body). Loads each needed type's objects once."""
    want = defaultdict(set)                     # type -> {id}
    for a in addresses:
        slug, _, nid = a.partition(":")
        t = _SLUG_TYPE.get(slug)
        if t:
            want[t].add(nid)
    out = {}
    for t, ids in want.items():
        spec = _CONTEXT.get(t)
        if not spec:
            continue
        load, idattr, body = spec
        by_id = {str(getattr(o, idattr)): o for o in load(root)}
        for nid in ids:
            o = by_id.get(nid)
            out[f"{NODE_DOMAINS[t]}:{nid}"] = (t, (body(o) or "").strip() if o else "")
    return out
```

Then, at the end of `walk`, replace the `nodes=` construction with a context-filled version:

```python
    ctx = resolve_context(visited, root)
    nodes = {}
    for a in visited:
        t, body = ctx.get(a, ("", ""))
        nodes[a] = Node(address=a, type=t, context=body)
    return Subgraph(nodes=nodes, edges=induced)
```

- [ ] **Step 4: Run tests to verify pass**

Run: `python -m pytest tests/graph/test_traverse_context.py tests/graph/test_traverse.py -q --no-cov`
Expected: PASS — the capability node carries type `Capability` and a non-empty statement; Task 1 tests still pass (they use a fake harvest and unknown ids → empty context, still fine).

- [ ] **Step 5: Commit**

```bash
git add tools/graph/traverse.py tests/graph/test_traverse_context.py
git commit -m "feat(graph): resolve claim+context per node in walk (subgraph = model context)"
```

---

### Task 3: Entry selectors + CLI `walk` subcommand

**Files:**
- Modify: `tools/graph/traverse.py` (selector resolution in `walk`)
- Modify: `tools/graph/__main__.py` (`walk` subcommand)
- Test: `tests/graph/test_traverse_selectors.py`, `tests/graph/test_walk_cli.py`

**Interfaces:**
- `walk` accepts `entry` as an address OR a selector `"type:<T>"` / `"under:<path>"`; CLI `python -m tools.graph walk <entry> --dir out|in|both --depth N|full`.

- [ ] **Step 1: Write the failing tests** — `tests/graph/test_traverse_selectors.py`:

```python
from tools.graph.traverse import _entry_addresses


def test_type_selector_returns_all_of_a_type():
    addrs = _entry_addresses("type:Capability", root=".")
    assert addrs and all(a.startswith("capabilities:") for a in addrs)


def test_under_selector_returns_code_units_below_path():
    addrs = _entry_addresses("under:src/api/", root=".")
    assert all(a.startswith("code:") for a in addrs)


def test_plain_address_passes_through():
    assert _entry_addresses("code:api", root=".") == ["code:api"]
```

And `tests/graph/test_walk_cli.py`:

```python
import subprocess
import sys


def test_walk_cli_runs():
    p = subprocess.run([sys.executable, "-m", "tools.graph", "walk", "capabilities:ask-the-corpus",
                        "--dir", "out", "--depth", "1"], capture_output=True, text=True)
    assert p.returncode == 0
    assert "capabilities:ask-the-corpus" in p.stdout
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/graph/test_traverse_selectors.py -q --no-cov`
Expected: FAIL — `_entry_addresses` does not exist.

- [ ] **Step 3: Add `_entry_addresses`** to `tools/graph/traverse.py` and call it at the top of `walk` (replace `starts = [entry] ...`):

```python
from tools.graph.reader import nodes as _all_nodes, _unit_dir


def _entry_addresses(entry: str, root: str = ".") -> List[str]:
    if entry.startswith("type:"):
        t = entry[len("type:"):]
        ids = _all_nodes(root).get(t, set())
        slug = NODE_DOMAINS.get(t)
        return sorted(f"{slug}:{i}" for i in ids) if slug else []
    if entry.startswith("under:"):
        path = entry[len("under:"):]
        p = path if path.endswith("/") else path + "/"
        return sorted(f"code:{u}" for u in _all_nodes(root).get("CodeUnit", set())
                      if _unit_dir(u).startswith(p))
    return [entry]
```

In `walk`, change `starts`:

```python
    starts = _entry_addresses(entry, root) if isinstance(entry, str) else list(entry)
```

- [ ] **Step 4: Add the `walk` CLI subcommand** in `tools/graph/__main__.py`. Add the handler and register it:

```python
def cmd_walk(args) -> int:
    from tools.graph.traverse import walk
    depth = None if args.depth == "full" else int(args.depth)
    sg = walk(args.entry, direction=args.dir, depth=depth)
    print(f"subgraph from {args.entry} (dir={args.dir}, depth={args.depth}): "
          f"{len(sg.nodes)} nodes, {len(sg.edges)} edges")
    for addr in sorted(sg.nodes):
        n = sg.nodes[addr]
        head = n.context.splitlines()[0] if n.context else ""
        print(f"  {addr}  [{n.type}]  {head[:80]}")
    for e in sg.edges:
        print(f"    {e.src} --{e.type}--> {e.dst}")
    return 0
```

Register in `main()`: `wp = sub.add_parser("walk"); wp.add_argument("entry"); wp.add_argument("--dir", default="both", choices=["out", "in", "both"]); wp.add_argument("--depth", default="full")`, and add `"walk": cmd_walk` to the dispatch dict.

- [ ] **Step 5: Run tests + smoke**

Run: `python -m pytest tests/graph/test_traverse_selectors.py tests/graph/test_walk_cli.py -q --no-cov`
Expected: PASS.
Run: `python -m tools.graph walk type:UseCase --dir out --depth 2 | head`
Expected: a subgraph summary listing use-case nodes and their outbound reach.

- [ ] **Step 6: Commit**

```bash
git add tools/graph/traverse.py tools/graph/__main__.py tests/graph/test_traverse_selectors.py tests/graph/test_walk_cli.py
git commit -m "feat(graph): walk entry selectors (type:/under:) + walk CLI subcommand"
```

---

### Task 4: Add the `GlossaryTerm` node type (+ `defined_in` edge)

**Files:**
- Modify: `tools/graph/registry.py` (`NODE_DOMAINS`, `EDGES`)
- Modify: `tools/graph/reader.py` (`_ADAPTERS`, a `_unit_of_file` resolve for authored file-paths)
- Modify: `tools/graph/traverse.py` (`_CONTEXT` gains GlossaryTerm)
- Test: `tests/graph/test_nodeset_glossary.py`

**Interfaces:** adds node type `GlossaryTerm` (slug `glossary`, id = `term`) and edge `defined_in: GlossaryTerm → CodeUnit`, authored from the term's `source:` file path resolved to its owning top-level code unit.

- [ ] **Step 1: Write the failing test** — `tests/graph/test_nodeset_glossary.py`:

```python
from tools.graph.reader import nodes, harvest


def test_glossary_terms_are_nodes():
    ns = nodes()
    assert "GlossaryTerm" in ns and len(ns["GlossaryTerm"]) > 50   # ~111 terms


def test_a_defined_in_edge_resolves_to_a_code_unit():
    edges = harvest()
    di = [e for e in edges if e.type == "defined_in"]
    assert di, "expected at least one defined_in edge"
    code_ids = nodes()["CodeUnit"]
    # at least one term's source maps to a real code unit (e.g. events/, projections/)
    assert any(e.dst.split(":", 1)[1] in code_ids for e in di)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/graph/test_nodeset_glossary.py -q --no-cov`
Expected: FAIL — `GlossaryTerm` not in `nodes()`.

- [ ] **Step 3: Register the node type + edge.** In `tools/graph/registry.py` add to `NODE_DOMAINS` (remove `GlossaryTerm` from the reserved comment):

```python
    "GlossaryTerm": "glossary",
```

and to `EDGES`:

```python
    EdgeType("defined_in", "defines", "GlossaryTerm", "CodeUnit", "authored",
             field="source", resolve="file",
             description="A glossary term is defined in the code unit that owns its source file."),
```

In `tools/graph/reader.py`, add the adapter (import `load_glossary` from `tools.glossary.model`):

```python
    "GlossaryTerm": (lambda root: load_glossary(os.path.join(root, "docs/glossary")), "term"),
```

and support the `resolve == "file"` case in `_authored` (a source *file* → its owning top-level unit). Add near `_units_under`:

```python
def _unit_of_file(path: str, code_ids: Set[str]) -> List[str]:
    """The top-level code unit that owns a src/tools file path (src/events/x.py -> 'events')."""
    p = (path or "").replace("\\", "/")
    parts = p.split("/")
    if len(parts) >= 2 and parts[0] in ("src", "tools"):
        unit = parts[1] if parts[0] == "src" else f"tools.{parts[1]}"
        return [unit] if unit in code_ids else []
    return []
```

and in `_authored`, extend the resolve dispatch:

```python
            if edge.resolve == "path":
                dsts = _units_under(str(t), node_ids[edge.to_type])
            elif edge.resolve == "file":
                dsts = _unit_of_file(str(t), node_ids[edge.to_type])
            else:
                dsts = [str(t)]
```

In `tools/graph/traverse.py` add to `_CONTEXT`:

```python
    "GlossaryTerm": (lambda root: load_glossary(os.path.join(root, "docs/glossary")), "term",
                     lambda o: o.definition),
```

(import `load_glossary` from `tools.glossary.model`).

- [ ] **Step 4: Run tests + graph-check**

Run: `python -m pytest tests/graph/test_nodeset_glossary.py -q --no-cov`
Expected: PASS.
Run: `python -m tools.graph check`
Expected: `graph-check: clean` (any Term whose source maps to an undocumented unit would surface as advisory drift — if so, note it; it is informative, not a failure).

- [ ] **Step 5: Commit**

```bash
git add tools/graph/registry.py tools/graph/reader.py tools/graph/traverse.py tests/graph/test_nodeset_glossary.py
git commit -m "feat(graph): GlossaryTerm node type + defined_in edge (source file -> owning unit)"
```

---

### Task 5: Add `GraphQuery` + `Prompt` node types (+ `consumed_by` edges)

**Files:**
- Modify: `tools/graph/registry.py` (`NODE_DOMAINS`, `EDGES`)
- Modify: `tools/graph/reader.py` (`_ADAPTERS`, `_DERIVED` derivations)
- Modify: `tools/graph/traverse.py` (`_CONTEXT`)
- Test: `tests/graph/test_nodeset_query_prompt.py`

**Interfaces:** adds node types `GraphQuery` (slug `graph-queries`, id = `name`) and `Prompt` (slug `prompts`, id = `key`), each with a derived edge `consumed_by → CodeUnit` from its `consumers` (already top-level unit ids).

- [ ] **Step 1: Write the failing test** — `tests/graph/test_nodeset_query_prompt.py`:

```python
from tools.graph.reader import nodes, harvest


def test_query_and_prompt_are_nodes():
    ns = nodes()
    assert ns.get("GraphQuery") and ns.get("Prompt")


def test_consumed_by_edges_resolve():
    edges = harvest()
    code_ids = nodes()["CodeUnit"]
    cb = [e for e in edges if e.type == "consumed_by"]
    assert cb, "expected consumed_by edges"
    assert all(e.dst.split(":", 1)[1] in code_ids for e in cb)   # all resolve to real units
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/graph/test_nodeset_query_prompt.py -q --no-cov`
Expected: FAIL — `GraphQuery`/`Prompt` not in `nodes()`.

- [ ] **Step 3: Register the node types + derived edges.** In `tools/graph/registry.py` add to `NODE_DOMAINS`:

```python
    "GraphQuery": "graph-queries",
    "Prompt": "prompts",
```

and to `EDGES` (two derived edges; the `field` is the `_DERIVED` key, not an object attribute):

```python
    EdgeType("consumed_by", "consumes", "GraphQuery", "CodeUnit", "derived",
             field="gq_consumed_by", resolve="id",
             description="A graph query is consumed by the code units that call it."),
    EdgeType("consumed_by", "consumes", "Prompt", "CodeUnit", "derived",
             field="prompt_consumed_by", resolve="id",
             description="A prompt is consumed by the code units that use it."),
```

In `tools/graph/reader.py` add the adapters (imports: `load_queries` from `tools.graphq.reader`, `load_prompt_entries` from `tools.prompts.reader`):

```python
    "GraphQuery": (load_queries, "name"),
    "Prompt": (load_prompt_entries, "key"),
```

and add the two `_DERIVED` derivations + register them in the `_DERIVED` dict:

```python
def _derived_consumers(from_type, id_attr, load):
    def build(edge: EdgeType, root: str) -> List[Edge]:
        out: List[Edge] = []
        for o in load(root):
            src = _addr(from_type, getattr(o, id_attr))
            for c in getattr(o, "consumers", []):
                out.append(Edge(edge.name, src, _addr("CodeUnit", c)))
        return out
    return build

# in the _DERIVED mapping:
    "gq_consumed_by": _derived_consumers("GraphQuery", "name", load_queries),
    "prompt_consumed_by": _derived_consumers("Prompt", "key", load_prompt_entries),
```

In `tools/graph/traverse.py` add to `_CONTEXT` (imports as above):

```python
    "GraphQuery": (load_queries, "name",
                   lambda o: f"{o.purpose or ''} returns={o.returns} labels={o.labels}".strip()),
    "Prompt": (load_prompt_entries, "key",
               lambda o: f"used_for={o.used_for} audience={o.audience}"),
```

- [ ] **Step 4: Run tests + graph-check**

Run: `python -m pytest tests/graph/test_nodeset_query_prompt.py -q --no-cov`
Expected: PASS.
Run: `python -m tools.graph check`
Expected: `graph-check: clean` (a consumer that is not a documented unit would surface as advisory — note if so).

- [ ] **Step 5: Commit**

```bash
git add tools/graph/registry.py tools/graph/reader.py tools/graph/traverse.py tests/graph/test_nodeset_query_prompt.py
git commit -m "feat(graph): GraphQuery + Prompt node types + consumed_by edges"
```

---

### Task 6: Wire-in — regenerate + reconcile

**Files:**
- Modify: `docs/code/tools.graph.md` (`key_modules` gains `traverse`)
- Regenerate: `docs/code/index.md`, `docs/graph/index.md` + `graph.md`, `docs/cli/index.md`

- [ ] **Step 1: Add `traverse` to the graph code unit's key_modules.** Edit `docs/code/tools.graph.md` frontmatter `key_modules:` to include `traverse` (keep existing entries).

- [ ] **Step 2: Regenerate + reconcile**

```bash
make code-index graph-index cli-index
python -m tools.code check      # clean
python -m tools.graph check     # clean
python -m tools.cli check       # walk subcommand catalogued (or note if cli-check ignores subcommands)
python -m tools.corpus check    # clean (unaffected)
```

Expected: the graph index/meta-schema now show 8 node types (was 5) and the new edge types (`defined_in`, `consumed_by`); all checks clean-or-advisory.

- [ ] **Step 3: Run the freshness gate locally (CI parity)**

```bash
make regen-derived && git diff --exit-code && echo "gate CLEAN" || echo "commit regenerated indexes"
```

Stage and commit any regenerated indexes so the gate is clean.

- [ ] **Step 4: Commit**

```bash
git add docs/code/tools.graph.md docs/code/index.md docs/graph/index.md docs/graph/graph.md docs/cli/index.md
git commit -m "feat(graph): wire L1 — traverse in code map + regenerate indexes (8 node types)"
```

---

## After all tasks

Run the full unit suite (`make test-unit`) and confirm green. Smoke a real deep walk end to end (`python -m tools.graph walk type:UseCase --dir out --depth full | head -40`) to see intents reach code through capabilities. No new ADR (0025 + 0020 govern). Run the final whole-branch review on the most capable model, then use **superpowers:finishing-a-development-branch**.

## Knowledge-graph check

Reviewed against `docs/index.md` on 2026-08-16.

| domain | touched? | consulted / reconciled | notes |
| --- | --- | --- | --- |
| graph | yes | new `traverse.py` (`walk`), 3 node types + edges in registry/reader; code unit + indexes regenerated | the subject |
| glossary / graph-queries / prompts | yes | readers reused as graph adapters; become node types | first-class nodes |
| code | yes (read-only) | target of `defined_in`/`consumed_by`; code map regenerated for the new `traverse` module | — |
| cli | yes | `walk` subcommand → `cli-index` | — |
| adr | yes | no new ADR — realizes 0025, extends 0020 | — |
| corpus / capabilities / use-cases / tests | no (logic) | unaffected | Test already a node |

**Verdict:** reconciled — graph is the subject (traversal + node-set completion); three domains become nodes via their existing readers; no new ADR.
