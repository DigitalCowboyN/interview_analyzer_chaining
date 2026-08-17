# Symbols + lazy frontier-expanding walk — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite `walk` to expand the graph lazily from the frontier (parsing symbol bodies only where the walk actually goes), then add **symbol**-grain code nodes (functions/classes/methods) with `contains` and pragmatic `calls` edges, gated by a `level` disclosure parameter.

**Architecture:** `walk` stops calling `harvest()`. Instead it BFS-expands via `neighbors(addr, direction, ctx)`, which computes a node's incident edges from *that node's own* file/AST/id for outbound + cheap structural edges, and from a per-walk **cached reverse index** (over the small doc/test domains + a module-import scan) for inbound intent edges. Symbol nodes are derived from a module's AST **only when the walk reaches that module at `level="symbol"`**. `harvest()` is retained unchanged for the whole-graph catalogs and checks; a regression test asserts lazy `walk` == harvest-based `walk` at module grain.

**Tech Stack:** Python 3 (stdlib: `ast`, `os`, `re`), pytest. No new deps. No static-analysis library.

**Spec:** `docs/superpowers/specs/2026-08-17-symbols-lazy-walk-design.md`. **ADRs:** new ADR (Task 6) extends ADR-0025 (ephemeral → incremental/lazy) and ADR-0020 (symbol level + `calls` edge); consistent with ADR-0019.

## Global Constraints

- **Harvest-equivalence is the correctness gate.** For the existing 8 node types, lazy `walk(entry, direction, depth)` must return a `Subgraph` whose `nodes` keys and induced `edges` (by `(src,dst,type)`) are **identical** to the current harvest-based implementation. A regression test drives real entries through both.
- **`harvest()` stays and is unchanged.** It still backs `docs/graph/*` renders and the non-blocking checks. Only `walk` goes lazy.
- **Symbol bodies are parsed only on the frontier.** At `level="module"` (default) no symbol AST is parsed. At `level="symbol"`, only modules actually visited by the walk are symbol-parsed. A test asserts unvisited modules are never symbol-parsed.
- **Structure derives everything; docstrings are optional context.** Symbol nodes exist from the AST regardless of docstrings (signature is always present). No frontmatter on code.
- **Pragmatic `calls` only — documented ceiling.** Resolve local-def + imported-symbol + class-instantiation calls from the symbol's own file. Do NOT attempt inferred-type `obj.method()`, inheritance `self.foo()`, or dynamic dispatch. A `# calls: code:x.y` marker is the escape hatch.
- **Intent edges stay coarse and unchanged** — symbols inherit intent by walking up `contained_by` (ADR-0019).
- **Names verbatim:** `neighbors(addr, direction, ctx)`, `WalkContext`, `symbols_of(module_id, root)`, `render_signature(node)`, `calls_of(...)`, `walk(entry, direction="both", depth=None, root=".", level="module")`.

---

### Task 1: Lazy frontier-expanding `walk` (module grain, harvest-equivalent)

Replace `walk`'s harvest+adjacency with neighbor-on-demand expansion. **No symbols yet** — this task only re-expresses today's module-grain graph lazily and proves equivalence.

**Files:**
- Create: `tools/graph/neighbors.py` (the `neighbors` primitive + `WalkContext`)
- Modify: `tools/graph/traverse.py` (`walk` BFS over `neighbors`; keep `resolve_context`, `Node`, `Subgraph`, `_entry_addresses`)
- Test: `tests/graph/test_lazy_walk.py`

**Interfaces:**
- Consumes: the registry `EDGES`/`NODE_DOMAINS`; the existing per-node derivations in `tools.code.reader` (`contains_edges`, `dep_edges`) and the domain loaders.
- Produces: `neighbors(addr, direction, ctx) -> list[tuple[str, Edge]]`; `WalkContext(root, level)` holding lazily-built, cached indexes.

- [ ] **Step 1: Write the failing equivalence test** — `tests/graph/test_lazy_walk.py`:

```python
import pytest

from tools.graph.reader import harvest
from tools.graph.traverse import walk, Subgraph


def _harvest_walk(entry, direction, depth, root="."):
    """The OLD algorithm, inlined, to compare against: harvest -> adjacency -> BFS."""
    from collections import defaultdict, deque
    edges = harvest(root)
    out, inc = defaultdict(list), defaultdict(list)
    for e in edges:
        out[e.src].append((e.dst, e))
        inc[e.dst].append((e.src, e))
    starts = [entry] if isinstance(entry, str) else list(entry)
    visited, frontier, seen = set(starts), deque((s, 0) for s in starts), set()
    used = []
    while frontier:
        addr, d = frontier.popleft()
        if depth is not None and d >= depth:
            continue
        nbrs = (out.get(addr, []) if direction in ("out", "both") else []) + \
               (inc.get(addr, []) if direction in ("in", "both") else [])
        for nbr, e in nbrs:
            k = (e.src, e.dst, e.type)
            if k not in seen:
                seen.add(k)
                used.append(e)
            if nbr not in visited:
                visited.add(nbr)
                frontier.append((nbr, d + 1))
    return visited, {(e.src, e.dst, e.type) for e in used if e.src in visited and e.dst in visited}


CASES = [
    ("code:tools.graph.reader", "both", 2),
    ("code:tools.graph", "out", 1),
    ("capabilities:link-the-domains", "out", None),
    ("code:tools.graph.classify", "in", 2),
]


@pytest.mark.parametrize("entry,direction,depth", CASES)
def test_lazy_walk_matches_harvest(entry, direction, depth):
    got = walk(entry, direction=depth and direction or direction, depth=depth)  # level defaults to module
    want_nodes, want_edges = _harvest_walk(entry, direction, depth)
    assert set(got.nodes) == want_nodes
    assert {(e.src, e.dst, e.type) for e in got.edges} == want_edges
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/graph/test_lazy_walk.py -q --no-cov`
Expected: FAIL — `walk()` got an unexpected keyword `level`, or (once `level` is added) node/edge mismatch until `neighbors` is correct.

- [ ] **Step 3: Implement `tools/graph/neighbors.py`.** The `WalkContext` builds cheap indexes once (lazily) and caches them; `neighbors` computes a node's incident edges. At module grain this reproduces every registry edge type.

```python
# tools/graph/neighbors.py
"""Lazy neighbor expansion for walk(): a node's incident edges computed from its own
file/AST/id (outbound + structural) and from a per-walk cached reverse index (inbound intent
edges), so a traversal never builds the whole graph. Symbol expansion (level='symbol') parses a
module's bodies only when the frontier reaches it."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from tools.graph.reader import Edge


@dataclass
class WalkContext:
    root: str = "."
    level: str = "module"
    _all_edges: Optional[List[Edge]] = None          # cached full edge set (see note)
    _out: Dict[str, List[Tuple[str, Edge]]] = field(default_factory=dict)
    _inc: Dict[str, List[Tuple[str, Edge]]] = field(default_factory=dict)
    _built: bool = False

    def _ensure(self):
        # Module-grain base graph: cheap (no symbol bodies). Built once per walk, cached.
        # This IS today's harvest at module grain; Task 4 makes symbol edges lazy on top.
        if self._built:
            return
        from tools.graph.reader import harvest
        self._all_edges = harvest(self.root)
        for e in self._all_edges:
            self._out.setdefault(e.src, []).append((e.dst, e))
            self._inc.setdefault(e.dst, []).append((e.src, e))
        self._built = True


def neighbors(addr: str, direction: str, ctx: WalkContext) -> List[Tuple[str, Edge]]:
    ctx._ensure()
    pairs: List[Tuple[str, Edge]] = []
    if direction in ("out", "both"):
        pairs += ctx._out.get(addr, [])
    if direction in ("in", "both"):
        pairs += ctx._inc.get(addr, [])
    return pairs
```

> **Design note for the implementer:** Task 1 deliberately builds the module-grain base once (via `harvest`) and caches it in `WalkContext` — at module grain this is already cheap and *guarantees* harvest-equivalence. The laziness that matters (symbol bodies) is added in Task 4, where `neighbors` splices in a frontier module's symbols on demand instead of the base graph carrying them. Structuring it this way lets the equivalence test pass first with zero behavior change, then symbols layer on without touching module-grain results.

- [ ] **Step 4: Rewrite `walk` in `tools/graph/traverse.py`** to BFS over `neighbors`, and add the `level` parameter (unused until Task 4, but part of the signature now):

```python
from tools.graph.neighbors import WalkContext, neighbors

def walk(entry, direction: str = "both", depth: Optional[int] = None,
         root: str = ".", level: str = "module") -> Subgraph:
    """Materialize the subgraph reachable from `entry` by expanding neighbors on demand
    (lazy, frontier-driven). `level` gates disclosure: 'module' (default) never descends into
    symbols; 'symbol' expands symbol nodes along the frontier."""
    ctx = WalkContext(root=root, level=level)
    starts = _entry_addresses(entry, root) if isinstance(entry, str) else list(entry)

    visited = set(starts)
    frontier = deque((s, 0) for s in starts)
    used_edges: List[Edge] = []
    seen_edge = set()

    while frontier:
        addr, d = frontier.popleft()
        if depth is not None and d >= depth:
            continue
        for nbr, e in neighbors(addr, direction, ctx):
            key = (e.src, e.dst, e.type)
            if key not in seen_edge:
                seen_edge.add(key)
                used_edges.append(e)
            if nbr not in visited:
                visited.add(nbr)
                frontier.append((nbr, d + 1))

    induced = [e for e in used_edges if e.src in visited and e.dst in visited]
    ctx_map = resolve_context(visited, root)
    nodes = {a: Node(address=a, type=ctx_map.get(a, ("", ""))[0],
                     context=ctx_map.get(a, ("", ""))[1]) for a in visited}
    return Subgraph(nodes=nodes, edges=induced)
```

Remove the now-unused `harvest`/`_adjacency` imports from `traverse.py` if nothing else references them (keep `_adjacency` only if a test uses it — it does not).

- [ ] **Step 5: Run the equivalence test**

Run: `python -m pytest tests/graph/test_lazy_walk.py -q --no-cov`
Expected: PASS (4 cases) — lazy `walk` reproduces harvest-based results exactly.
Run: `python -m pytest tests/graph -q --no-cov` — the rest of the graph suite still passes (walk's observable behavior is unchanged at module grain).

- [ ] **Step 6: Commit**

```bash
python -m flake8 tools/graph/neighbors.py tools/graph/traverse.py tests/graph/test_lazy_walk.py
git add tools/graph/neighbors.py tools/graph/traverse.py tests/graph/test_lazy_walk.py
git commit -m "feat(graph): walk expands via neighbors() (lazy engine seam); harvest-equivalent at module grain"
```

---

### Task 2: Symbol discovery from AST (nodes, signature, docstring, kind)

Add the derivation of symbol nodes — **not wired into the graph yet**, so nothing changes until Task 4.

**Files:**
- Modify: `tools/code/reader.py` (add `Symbol`, `symbols_of`, `render_signature`)
- Test: `tests/code/test_symbols.py`

**Interfaces:**
- Produces: `symbols_of(module_id, root=".") -> List[Symbol]` where `Symbol` has `.id` (`code`-less dotted, e.g. `tools.graph.traverse.walk`), `.kind` (`function|class|method`), `.signature`, `.docstring`, `.parent` (module or class id), `.calls` (filled in Task 3). `render_signature(node) -> str`.

- [ ] **Step 1: Write the failing test** — `tests/code/test_symbols.py`:

```python
import os

from tools.code.reader import symbols_of


def _w(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w", encoding="utf-8").write(text)


def test_symbols_functions_classes_methods(tmp_path):
    _w(str(tmp_path / "src/api/__init__.py"), "")
    _w(str(tmp_path / "src/api/main.py"),
       'def make(x, y=1) -> int:\n    """Build."""\n    return x + y\n\n'
       'class Router:\n    """Routes."""\n    def add(self, path):\n        return path\n')
    by_id = {s.id: s for s in symbols_of("api.main", str(tmp_path))}
    assert by_id["api.main.make"].kind == "function"
    assert by_id["api.main.make"].signature == "make(x, y=1) -> int"
    assert by_id["api.main.make"].docstring == "Build."
    assert by_id["api.main.Router"].kind == "class"
    assert by_id["api.main.Router.add"].kind == "method"
    assert by_id["api.main.Router.add"].parent == "api.main.Router"
    assert by_id["api.main.make"].parent == "api.main"


def test_symbols_without_docstring_are_thin_not_absent(tmp_path):
    _w(str(tmp_path / "src/x/__init__.py"), "")
    _w(str(tmp_path / "src/x/m.py"), "def f(a):\n    return a\n")
    by_id = {s.id: s for s in symbols_of("x.m", str(tmp_path))}
    assert "x.m.f" in by_id                       # exists from the AST
    assert by_id["x.m.f"].docstring == ""         # thin, not absent
    assert by_id["x.m.f"].signature == "f(a)"
```

- [ ] **Step 2: Run to verify it fails** — `python -m pytest tests/code/test_symbols.py -q --no-cov` → FAIL (`cannot import name 'symbols_of'`).

- [ ] **Step 3: Implement** in `tools/code/reader.py`:

```python
@dataclass
class Symbol:
    id: str                                  # dotted, e.g. "api.main.Router.add"
    kind: str                                # function | class | method
    signature: str = ""
    docstring: str = ""
    parent: str = ""                         # module id, or the class id for a method
    calls: List[str] = field(default_factory=list)   # filled by Task 3


def render_signature(node) -> str:
    a = node.args
    parts = [arg.arg for arg in a.args]
    if a.vararg:
        parts.append("*" + a.vararg.arg)
    if a.kwarg:
        parts.append("**" + a.kwarg.arg)
    # attach simple defaults to the trailing positional args
    defaults = list(a.defaults)
    if defaults:
        base = len(a.args) - len(defaults)
        for i, d in enumerate(defaults):
            try:
                parts[base + i] = f"{a.args[base + i].arg}={ast.unparse(d)}"
            except Exception:
                pass
    ret = f" -> {ast.unparse(node.returns)}" if getattr(node, "returns", None) else ""
    return f"{node.name}({', '.join(parts)}){ret}"


def _module_path(module_id: str, root: str) -> str:
    if module_id.startswith("tools."):
        return os.path.join(root, "tools", *module_id.split(".")[1:]) + ".py"
    return os.path.join(root, "src", *module_id.split(".")) + ".py"


def symbols_of(module_id: str, root: str = ".") -> List[Symbol]:
    """Top-level functions/classes of a module, and a class's methods (one level of nesting)."""
    path = _module_path(module_id, root)
    try:
        tree = ast.parse(open(path, encoding="utf-8", errors="ignore").read())
    except (OSError, SyntaxError):
        return []
    out: List[Symbol] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.append(Symbol(f"{module_id}.{node.name}", "function",
                              render_signature(node), (ast.get_docstring(node) or "").strip(),
                              module_id))
        elif isinstance(node, ast.ClassDef):
            cid = f"{module_id}.{node.name}"
            out.append(Symbol(cid, "class", f"class {node.name}",
                              (ast.get_docstring(node) or "").strip(), module_id))
            for m in node.body:
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    out.append(Symbol(f"{cid}.{m.name}", "method",
                                      render_signature(m), (ast.get_docstring(m) or "").strip(), cid))
    return out
```

(`ast` is already imported in `reader.py`.)

- [ ] **Step 4: Run to verify pass** — `python -m pytest tests/code/test_symbols.py -q --no-cov` → PASS (2 passed).

- [ ] **Step 5: Sanity on the real repo**

Run: `python -c "from tools.code.reader import symbols_of; s=symbols_of('tools.graph.traverse','.'); [print(x.kind, x.id, '::', x.signature) for x in s]"`
Expected: `walk`, `resolve_context`, `_adjacency`, `_entry_addresses`, `Node`, `Subgraph`, … with correct signatures. Note the count.

- [ ] **Step 6: Commit**

```bash
python -m flake8 tools/code/reader.py tests/code/test_symbols.py
git add tools/code/reader.py tests/code/test_symbols.py
git commit -m "feat(code): symbols_of — derive function/class/method symbols (signature+docstring) from AST"
```

---

### Task 3: Pragmatic `calls` resolution

Resolve a symbol's calls from its own file. **The `calls` edge is walk-time only** — it is emitted by
`neighbors` at `level="symbol"` (Task 4), NOT registered in `registry.EDGES`. Registering it there
would make `harvest()` (which iterates `EDGES` and dispatches derived edges through
`_DERIVED[field]`) either KeyError or eagerly derive every symbol's calls on every harvest — the exact
tax this milestone avoids. So symbols/`calls` never touch `harvest`; they live only on the lazy walk path.

**Files:**
- Modify: `tools/code/reader.py` (add `calls_of` + wire into `symbols_of`)
- Test: `tests/code/test_calls.py`

**Interfaces:**
- Produces: `calls_of(func_node, module_id, name_index, node_ids) -> List[str]` — resolved callee symbol ids. `symbols_of` fills each `Symbol.calls`.

- [ ] **Step 1: Write the failing test** — `tests/code/test_calls.py`:

```python
import os

from tools.code.reader import symbols_of


def _w(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w", encoding="utf-8").write(text)


def _fixture(tmp):
    _w(str(tmp / "src/svc/__init__.py"), "")
    _w(str(tmp / "src/svc/render.py"), "def draw(x):\n    return x\n")
    _w(str(tmp / "src/svc/main.py"),
       "from src.svc.render import draw\n\n"
       "def helper():\n    return 1\n\n"
       "def run(obj):\n    helper()\n    draw(3)\n    obj.method()\n")


def test_calls_resolve_local_and_imported(tmp_path):
    _fixture(tmp_path)
    by_id = {s.id: s for s in symbols_of("svc.main", str(tmp_path))}
    calls = set(by_id["svc.main.run"].calls)
    assert "svc.main.helper" in calls          # local def
    assert "svc.render.draw" in calls          # imported symbol
    assert not any(c.endswith(".method") for c in calls)   # obj.method() unresolved -> skipped


def test_calls_marker_escape_hatch(tmp_path):
    _fixture(tmp_path)
    (tmp_path / "src/svc/main.py").write_text(
        "def run(obj):\n    # calls: code:svc.render.draw\n    obj.method()\n", encoding="utf-8")
    by_id = {s.id: s for s in symbols_of("svc.main", str(tmp_path))}
    assert "svc.render.draw" in by_id["svc.main.run"].calls   # asserted by marker
```

- [ ] **Step 2: Run to verify it fails** — FAIL (calls empty).

- [ ] **Step 3: Implement `calls_of` in `tools/code/reader.py`** and call it from `symbols_of`. Build a per-module **name index** (imports + local top-level defs) once, then resolve each `Call`:

```python
_CALLS_MARKER = re.compile(r"#\s*calls:\s*code:([\w.]+)")


def _name_index(tree, module_id: str) -> Dict[str, str]:
    """local name -> target symbol/module id, from imports + top-level defs of this module."""
    idx: Dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and \
                (node.module.startswith("src.") or node.module.startswith("tools.")):
            base = node.module[4:] if node.module.startswith("src.") else node.module
            for alias in node.names:
                idx[alias.asname or alias.name] = f"{base}.{alias.name}"
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(("src.", "tools.")):
                    tgt = alias.name[4:] if alias.name.startswith("src.") else alias.name
                    idx[alias.asname or alias.name] = tgt
    for node in tree.body:                       # local top-level defs
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            idx[node.name] = f"{module_id}.{node.name}"
    return idx


def calls_of(func_node, name_index: Dict[str, str], marker_text: str = "") -> List[str]:
    out = set(_CALLS_MARKER.findall(marker_text))         # explicit markers
    for n in ast.walk(func_node):
        if isinstance(n, ast.Call):
            f = n.func
            if isinstance(f, ast.Name) and f.id in name_index:          # foo()
                out.add(name_index[f.id])
            elif isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) \
                    and f.value.id in name_index:                       # mod.foo()
                out.add(f"{name_index[f.value.id]}.{f.attr}")
            # obj.method() on an unknown Name -> not in name_index -> skipped (ceiling)
    return sorted(out)
```

In `symbols_of`, after building the tree, compute `nidx = _name_index(tree, module_id)`, read the source text for markers, and for each function/method set `sym.calls = calls_of(node, nidx, marker_text=<module source>)`. (For the marker this milestone, scan the whole module source — a `# calls:` comment picked up by the regex; module-scope markers are accepted for now.)

The `calls` edge type is materialized by `neighbors` in Task 4 as `Edge("calls", src, dst)` — do NOT add it to `registry.EDGES` (see the task preamble).

- [ ] **Step 4: Run to verify pass** — `python -m pytest tests/code/test_calls.py -q --no-cov` → PASS.

- [ ] **Step 5: Commit**

```bash
python -m flake8 tools/code/reader.py tests/code/test_calls.py
git add tools/code/reader.py tests/code/test_calls.py
git commit -m "feat(code): pragmatic calls resolution (local+imported) + # calls: marker (walk-time calls edge)"
```

---

### Task 4: Wire symbols into `walk` — the disclosure gate + lazy symbol expansion

Make `level="symbol"` splice a frontier module's symbols (nodes + `contains` + `calls`) into the walk, parsed only when reached.

**Files:**
- Modify: `tools/graph/neighbors.py` (lazy symbol expansion for frontier modules)
- Modify: `tools/graph/traverse.py` `_CONTEXT`/`resolve_context` (symbol context = signature + docstring)
- Test: `tests/graph/test_symbol_walk.py`

- [ ] **Step 1: Write the failing test** — `tests/graph/test_symbol_walk.py`:

```python
from tools.graph.traverse import walk


def test_module_level_surfaces_no_symbols():
    sg = walk("code:tools.graph.traverse", direction="out", depth=1, level="module")
    assert not any(a.startswith("code:tools.graph.traverse.") for a in sg.nodes)


def test_symbol_level_discloses_symbols_and_contains():
    sg = walk("code:tools.graph.traverse", direction="out", depth=1, level="symbol")
    assert "code:tools.graph.traverse.walk" in sg.nodes           # symbol node present
    assert sg.nodes["code:tools.graph.traverse.walk"].context     # signature/docstring context
    assert any(e.type == "contains" and e.dst == "code:tools.graph.traverse.walk"
               for e in sg.edges)                                  # module contains symbol


def test_symbol_walk_up_reaches_module_then_intent():
    # from a symbol, walk 'in' reaches its module (contained_by), then its capability
    sg = walk("code:tools.graph.classify.derive_axes", direction="in", depth=3, level="symbol")
    assert "code:tools.graph.classify" in sg.nodes                # its module
    assert any(a.startswith("capabilities:") for a in sg.nodes)   # inherited intent via walk-up
```

- [ ] **Step 2: Run to verify it fails** — FAIL (symbols absent even at symbol level).

- [ ] **Step 3: Implement lazy symbol expansion** in `neighbors.py`. When `ctx.level == "symbol"`, augment `neighbors` for the two relevant node types:
  - A **module** code node gains outbound `contains` edges to its symbols (via `symbols_of`, memoized per module in `ctx`), built the first time the module is visited.
  - A **symbol** code node gains: `contained_by` (its parent module/class), `contains` (a class → its methods), and `calls`/`called_by`. Outbound `calls` come from the symbol's own `.calls`. Inbound `called_by` requires the reverse — scope it: only resolve `called_by` among symbols already materialized in this walk (documented limitation; a full reverse-call index is deferred).

Concretely, extend `neighbors` (after `ctx._ensure()`), guarded by `ctx.level == "symbol"`:

```python
def _symbol_edges(addr, ctx):
    dom, _, cid = addr.partition(":")
    if dom != "code":
        return []
    edges = []
    # module -> its symbols (contains); memoize symbols_of per module
    syms = ctx.symbols_for(cid)                       # returns [] if cid is not a module we can parse
    for s in syms:
        edges.append((f"code:{s.id}", Edge("contains", addr, f"code:{s.id}")))
    # if addr is itself a symbol, add contained_by (parent) + calls (from its own record)
    rec = ctx.symbol_record(cid)
    if rec:
        edges.append((f"code:{rec.parent}", Edge("contains", f"code:{rec.parent}", addr)))
        for callee in rec.calls:
            edges.append((f"code:{callee}", Edge("calls", addr, f"code:{callee}")))
    return edges
```

Add `symbols_for(id)`/`symbol_record(id)` memoized helpers to `WalkContext` that call `symbols_of` on the owning module the first time and cache. `neighbors` filters these by `direction` the same way as base edges. **No symbol work happens unless `ctx.level == "symbol"` and the node is visited** — satisfying the frontier-laziness constraint.

Add symbol context to `resolve_context` (`traverse.py`): a `code:` address that is a symbol (has more segments than its module / is in the walk's symbol set) resolves its context to `signature + docstring`. Simplest: give `WalkContext` a `symbol_record` lookup and have `walk` fill symbol node context directly from it (since `resolve_context`'s `load_units` only knows modules). Update the node-context assembly in `walk` to prefer a symbol record when present.

- [ ] **Step 4: Run to verify pass** — `python -m pytest tests/graph/test_symbol_walk.py tests/graph/test_lazy_walk.py -q --no-cov` → PASS (symbols appear only at symbol level; module-grain equivalence still holds).

- [ ] **Step 5: Assert frontier-laziness** — add a test that monkeypatches/counts `symbols_of` calls and asserts a `level="symbol"` walk bounded to one module parses symbols for only the visited module(s), never the whole repo.

- [ ] **Step 6: Commit**

```bash
python -m flake8 tools/graph/neighbors.py tools/graph/traverse.py tests/graph/test_symbol_walk.py
git add tools/graph/neighbors.py tools/graph/traverse.py tests/graph/test_symbol_walk.py
git commit -m "feat(graph): level='symbol' discloses symbols lazily along the frontier (contains + calls; walk-up to intent)"
```

---

### Task 5: Symbol docstring backlog + no-dangling + freshness

Fold symbols into the completeness signals without disturbing the default (module) catalogs.

**Files:**
- Modify: `tools/code/check.py` (extend `check_missing_docstring` to optionally include symbols — advisory, off by default)
- Test: `tests/code/test_symbol_backlog.py`

- [ ] **Step 1:** Add an opt-in symbol pass to `check_missing_docstring` (or a sibling `check_missing_symbol_docstring(root)`) that lists `symbols_of` results with empty docstrings, framed as "thin (signature-only)". Keep it **out** of the default `run_all` (symbols are opt-in; the module backlog is done) — expose it via a flag or a separate function the CLI can call, so `make code-index`/`code check` are unchanged. Test that a fixture symbol with no docstring is flagged and one with a docstring is not.

- [ ] **Step 2: No-dangling on the real graph.** Add a test / run `python -m tools.graph check` — the module-grain graph is unchanged, so `check_endpoints` stays clean. For symbol-level, add a test that every `calls`/`contains` edge emitted by a `level="symbol"` walk resolves to a node in that walk (no dangling within the materialized subgraph).

- [ ] **Step 3: Freshness.** Confirm `make regen-derived && git diff --exit-code` is CLEAN — symbols are opt-in and not in any generated catalog this milestone, so nothing regenerates differently.

- [ ] **Step 4: Full suite** — `make test-unit` green.

- [ ] **Step 5: Commit**

```bash
git add tools/code/check.py tests/code/test_symbol_backlog.py
git commit -m "feat(code): opt-in symbol docstring backlog; symbol-walk edges dangle-free (module catalogs unchanged)"
```

---

### Task 6: ADR + regen + final review

- [ ] **Step 1: Scaffold the ADR** — `python -m tools.adr new "Lazy frontier-expanding traversal and symbol-grain code nodes"`. Fill it: decision = `walk` expands neighbors on demand (symbol bodies parsed only on the frontier) and code reaches symbol grain (AST function/class/method nodes, signature+docstring context, `contains`, pragmatic `calls` with a documented ceiling + `# calls:` marker), gated by `level`. **Extends** ADR-0025 (ephemeral rebuilt-from-source matures to incremental/lazy per-node expansion) and ADR-0020 (adds the `symbol` level value and the `calls`/`called_by` edge). Consistent with ADR-0019 (symbols inherit intent by walk-up). `source:` = the spec.

- [ ] **Step 2:** `make adr-index && make adr-check` — fix any drift for the new ADR.

- [ ] **Step 3: Full gate** — `make regen-derived && git diff --exit-code` CLEAN; `make test-unit` green; `python -m tools.graph check` no dangling.

- [ ] **Step 4: Commit the ADR**, then dispatch the final whole-branch review (most capable model) with a review package (`scripts/review-package "$(git merge-base main HEAD)" HEAD`), then use **superpowers:finishing-a-development-branch**.

## After all tasks

- Lazy `walk` reproduces harvest results at module grain (equivalence test); symbols appear only at `level="symbol"` and only for visited modules (frontier-laziness test).
- `walk(symbol, "in")` reaches its module then its governing capability/ADR (walk-up).
- Pragmatic `calls` edges resolve local + imported calls; the ceiling (`obj.method()`) is documented and the `# calls:` marker covers exceptions.
- `make test-unit` green; freshness clean; module-grain catalogs unchanged; new ADR.

## Knowledge-graph check

Reviewed against `docs/index.md` on 2026-08-17.

| domain | touched? | consulted / reconciled | notes |
| --- | --- | --- | --- |
| graph | yes | `walk` rewritten lazy via `neighbors`/`WalkContext`; `level` gate; `calls` edge; `harvest` retained for catalogs | the engine subject |
| code | yes | `symbols_of`/`render_signature`/`calls_of` (AST symbol nodes + pragmatic calls + `# calls:` marker) | the grain subject |
| capabilities/adr/use-cases/tests/prompts/glossary | no (logic) | read as today; symbols inherit intent via walk-up | — |
| adr | yes | new ADR (extends 0025 + 0020, consistent with 0019) | — |

**Verdict:** reconciled — the traversal engine goes lazy/frontier-expanding (extends ADR-0025), code reaches symbol grain (structure-derived nodes + pragmatic `calls`), and disclosure is gated by `level`. Module-grain behavior, catalogs, and the freshness gate are provably unchanged (equivalence test); full semantic resolution, authored flow-nodes, and derived summaries are deferred.
