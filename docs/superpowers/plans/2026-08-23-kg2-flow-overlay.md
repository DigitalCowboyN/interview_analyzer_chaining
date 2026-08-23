# KG-2 — derived event-and-label flow overlay — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the event-sourced write path, the analysis pipeline, and the Neo4j schema-lineage traversable by deriving four edges that connect nodes/metadata already in the graph — closing the KG-1 `pipeline-*` and `deploy-neo4j-schema-blast` eval gaps.

**Architecture:** `reads` (GraphQuery→GlossaryTerm) is cheap/static → harvested (registry + `_DERIVED`). `emits`/`handled_by`/`writes` are symbol-grain and parse handler/aggregate source → walk-time only (spliced in `tools/graph/neighbors.py::_symbol_edges`, like `calls`), never in `harvest()`. `emits` derives from the existing symbol `calls` (filtered to event classes) + a `# emits:` marker. `handled_by`/`writes` derive from `registry.register(...)` and handler Cypher `MERGE`, in a new `tools/graph/flow.py`, memoized on `WalkContext`.

**Tech Stack:** Python 3 (stdlib: `ast`, `re`), pytest. No new deps.

**Spec:** `docs/superpowers/specs/2026-08-23-kg2-flow-overlay-design.md`. **ADRs:** new ADR (Task 5) extends ADR-0020; consistent with ADR-0027 (symbol-grain/lazy) and ADR-0019.

## Global Constraints

- **Reuse existing nodes — no new node types.** Event nodes = `events.*Data` class symbols; label nodes = existing `GlossaryTerm`s (`defined_in → code:projections.schema`).
- **Grain split:** `reads` = harvest-grain (registered, cheap). `emits`/`handled_by`/`writes` = symbol-grain, **walk-time only** (spliced in `_symbol_edges` under `level="symbol"`; NOT added to `registry.EDGES`, exactly like `calls`) — so a module-grain walk and `harvest()` are unchanged (harvest-equivalence preserved).
- **Derive-first; markers only for the ceiling.** Only `emits` has a marker (`# emits:`), for dynamic emission `calls` can't resolve. Everything else is purely derived.
- **Under-link, never mislink.** A label with no matching `GlossaryTerm`, or a `register()` whose event has no `…Data` class, is skipped and (Task 5) reported — never a guessed edge.
- **Names verbatim:** edge types `reads`/`read_by`, `emits`/`emitted_by`, `handled_by`/`handles`, `writes`/`written_by`; `flow.register_map(root)`, `flow.handler_labels(handler_id, root)`; `Symbol.emits`.

---

### Task 1: `reads` edge — GraphQuery → GlossaryTerm (harvest-grain)

**Files:**
- Modify: `tools/graph/registry.py` (add the `reads` EdgeType)
- Modify: `tools/graph/reader.py` (add `_derived_reads` + wire into `_DERIVED`)
- Test: `tests/graph/test_reads_edge.py`

**Interfaces:** consumes `tools.graphq.reader.load_queries` (`QueryEntry.graph_id`, `.labels`) and the glossary term set.

- [ ] **Step 1: Write the failing test** — `tests/graph/test_reads_edge.py`:

```python
from tools.graph.reader import harvest


def test_reads_edges_link_queries_to_label_terms():
    edges = harvest(".")
    reads = {(e.src, e.dst) for e in edges if e.type == "reads"}
    # a real query declares labels=['Project'] -> reads glossary:Project (Project IS a glossary term)
    assert ("graph-queries:reader.project_exists", "glossary:Project") in reads
    # every reads edge targets a real glossary term (no dangling)
    from tools.graph.reader import nodes
    terms = nodes(".").get("GlossaryTerm", set())
    assert all(dst.split(":", 1)[1] in terms for _, dst in reads)
```

- [ ] **Step 2: Run to verify it fails** — `python -m pytest tests/graph/test_reads_edge.py -q --no-cov` → FAIL (no `reads` edges).

- [ ] **Step 3: Register the edge** in `tools/graph/registry.py` (append to `EDGES`, near `consumed_by`):

```python
    EdgeType("reads", "read_by", "GraphQuery", "GlossaryTerm", "derived",
             field="reads_edges", resolve="id",
             description="A graph query reads nodes of a Neo4j label (a glossary term)."),
```

- [ ] **Step 4: Implement `_derived_reads`** in `tools/graph/reader.py` (beside `_derived_consumers`), and wire it:

```python
def _derived_reads(edge: EdgeType, root: str) -> List[Edge]:
    from tools.glossary.model import load_glossary
    terms = {t.term for t in load_glossary(os.path.join(root, "docs/glossary"))}
    out: List[Edge] = []
    for q in load_queries(root):
        for label in getattr(q, "labels", []) or []:
            if label in terms:                                   # only real glossary labels
                out.append(Edge(edge.name, _addr("GraphQuery", q.graph_id),
                                _addr("GlossaryTerm", label)))
    return out
```

Add to `_DERIVED`: `"reads_edges": _derived_reads,`.

- [ ] **Step 5: Run + real-repo sanity** — `python -m pytest tests/graph/test_reads_edge.py -q --no-cov` PASS. Then:

```bash
python -c "from tools.graph.reader import harvest; e=[x for x in harvest('.') if x.type=='reads']; print(len(e),'reads edges'); [print(' ',x.src,'->',x.dst) for x in e[:8]]"
python -m tools.graph check   # no 'does not resolve'
```

- [ ] **Step 6: Regenerate + commit**

```bash
make regen-derived            # graph catalogs pick up the reads count
python -m flake8 tools/graph/registry.py tools/graph/reader.py tests/graph/test_reads_edge.py
git add tools/graph/registry.py tools/graph/reader.py tests/graph/test_reads_edge.py docs/graph/index.md docs/graph/graph.md
git commit -m "feat(graph): reads edge (GraphQuery -> GlossaryTerm label) — derived from query labels metadata"
```
(Add any other file `make regen-derived` changed; confirm `make regen-derived && git diff --exit-code` clean.)

---

### Task 2: `emits` edge — code symbol → event-class symbol (+ `# emits:` marker)

**Files:**
- Modify: `tools/code/reader.py` (`Symbol.emits`; fill it in `symbols_of`; `# emits:` marker)
- Modify: `tools/graph/neighbors.py` (`_symbol_edges` splices `emits`)
- Test: `tests/code/test_emits.py`, `tests/graph/test_flow_walk.py` (start it here)

**Interfaces:** produces `Symbol.emits: List[str]` (dotted event-class ids). `_symbol_edges` emits `Edge("emits", symbol, event)`.

- [ ] **Step 1: Write the failing test** — `tests/code/test_emits.py`:

```python
import os
from tools.code.reader import symbols_of


def _w(p, t):
    os.makedirs(os.path.dirname(p), exist_ok=True); open(p, "w").write(t)


def test_emits_from_event_constructor_call(tmp_path):
    _w(str(tmp_path / "src/events/__init__.py"), "")
    _w(str(tmp_path / "src/events/foo_events.py"), "class FooHappenedData:\n    pass\n")
    _w(str(tmp_path / "src/events/aggregates.py"),
       "from src.events.foo_events import FooHappenedData\n\n"
       "def do_it():\n    return FooHappenedData()\n")
    by = {s.id: s for s in symbols_of("events.aggregates", str(tmp_path))}
    assert "events.foo_events.FooHappenedData" in by["events.aggregates.do_it"].emits


def test_emits_marker_for_dynamic_emission(tmp_path):
    _w(str(tmp_path / "src/events/__init__.py"), "")
    _w(str(tmp_path / "src/events/foo_events.py"), "class FooHappenedData:\n    pass\n")
    _w(str(tmp_path / "src/events/aggregates.py"),
       "def do_it(cls):\n    # emits: events.foo_events.FooHappenedData\n    return cls()\n")
    by = {s.id: s for s in symbols_of("events.aggregates", str(tmp_path))}
    assert "events.foo_events.FooHappenedData" in by["events.aggregates.do_it"].emits


def test_non_event_call_is_not_emit(tmp_path):
    _w(str(tmp_path / "src/svc/__init__.py"), "")
    _w(str(tmp_path / "src/svc/m.py"), "class Thing:\n    pass\n\ndef f():\n    return Thing()\n")
    by = {s.id: s for s in symbols_of("svc.m", str(tmp_path))}
    assert by["svc.m.f"].emits == []
```

- [ ] **Step 2: Run to verify it fails** — `cannot ... 'emits'` / attribute missing.

- [ ] **Step 3: Implement.** In `tools/code/reader.py`: add `emits: List[str] = field(default_factory=list)` to `Symbol`; add the marker regex and helpers near `_CALLS_MARKER`:

```python
_EMITS_MARKER = re.compile(r"#\s*emits:\s*([\w.]+)")


def _is_event_class(dotted_id: str) -> bool:
    parts = dotted_id.split(".")
    return len(parts) >= 3 and parts[0] == "events" and parts[-1].endswith("Data")
```

In `symbols_of`, after computing each function/method symbol's `.calls`, set its `.emits`:

```python
    sym.emits = sorted({c for c in sym.calls if _is_event_class(c)}
                       | set(_EMITS_MARKER.findall(_marker(node))))
```

(`_marker(node)` is the per-def source segment already used for `# calls:` in the symbols milestone.)

- [ ] **Step 4: Splice `emits` in `tools/graph/neighbors.py`** — in `_symbol_edges`, in the symbol `out`/`both` branch (after the `calls` loop):

```python
        for ev in getattr(rec, "emits", []):
            out.append((f"code:{ev}", Edge("emits", addr, f"code:{ev}")))
```

- [ ] **Step 5: Start the end-to-end walk test** — `tests/graph/test_flow_walk.py`:

```python
from tools.graph.traverse import walk


def test_aggregate_emits_event_at_symbol_level():
    # a real aggregate method that constructs an event payload class
    sg = walk("code:events.aggregates", direction="out", depth=2, level="symbol")
    assert any(e.type == "emits" and e.dst.endswith("Data") for e in sg.edges)


def test_module_grain_has_no_flow_edges():
    sg = walk("code:events.aggregates", direction="out", depth=1, level="module")
    assert not any(e.type in ("emits", "handled_by", "writes") for e in sg.edges)
```

- [ ] **Step 6: Run + real sanity** — `python -m pytest tests/code/test_emits.py tests/graph/test_flow_walk.py -q --no-cov` PASS. Then:

```bash
python -c "from tools.code.reader import symbols_of; s=[x for x in symbols_of('events.aggregates') if x.emits]; print(len(s),'emitting methods'); [print(' ',x.id,'->',x.emits) for x in s[:6]]"
```

- [ ] **Step 7: Commit**

```bash
python -m flake8 tools/code/reader.py tools/graph/neighbors.py tests/code/test_emits.py tests/graph/test_flow_walk.py
git add tools/code/reader.py tools/graph/neighbors.py tests/code/test_emits.py tests/graph/test_flow_walk.py
git commit -m "feat(graph): emits edge (code symbol -> event class), derived from calls + # emits: marker (symbol-lazy)"
```

---

### Task 3: `handled_by` edge — event-class → projection handler (registry parse)

**Files:**
- Create: `tools/graph/flow.py` (`register_map`)
- Modify: `tools/graph/neighbors.py` (`WalkContext` caches `register_map`; `_symbol_edges` splices `handled_by`)
- Test: `tests/graph/test_flow_registry.py`

**Interfaces:** `flow.register_map(root=".") -> Dict[str, str]` — event-class dotted id → handler-class dotted id.

- [ ] **Step 1: Write the failing test** — `tests/graph/test_flow_registry.py`:

```python
from tools.graph.flow import register_map


def test_register_map_bridges_type_to_data_class_and_handler():
    m = register_map(".")
    # registry.register("InterviewCreated", InterviewCreatedHandler(...)) in projections.bootstrap
    assert m.get("events.interview_events.InterviewCreatedData") == \
        "projections.handlers.interview_handlers.InterviewCreatedHandler"
```

- [ ] **Step 2: Implement `tools/graph/flow.py`.** Parse `register("<Type>", <Handler>(...))` in `src/projections/bootstrap.py`; resolve `"<Type>"` → the `events.*.<Type>Data` class id (scan `events.*` modules for the class) and `<Handler>` → its `projections.handlers.*.<Handler>` symbol id (scan handler modules for the class def):

```python
"""KG-2 flow derivations parsed from source: the event->handler registry map and each handler's
written Neo4j labels. Consumed by tools.graph.neighbors at level='symbol' (memoized per walk)."""
from __future__ import annotations

import os
import re
from typing import Dict, List

from tools.code.reader import load_units, symbols_of

_REGISTER = re.compile(r'register\(\s*["\'](\w+)["\']\s*,\s*(\w+)\s*\(')


def _class_index(root: str, pkg_prefix: str) -> Dict[str, str]:
    """class name -> dotted symbol id, over modules under a package prefix (e.g. 'events', 'projections.handlers')."""
    idx: Dict[str, str] = {}
    for u in load_units(root):
        if u.level == "module" and u.unit.startswith(pkg_prefix):
            for s in symbols_of(u.unit, root):
                if s.kind == "class":
                    idx[s.id.split(".")[-1]] = s.id
    return idx


def register_map(root: str = ".") -> Dict[str, str]:
    events = _class_index(root, "events")            # 'InterviewCreatedData' -> events.interview_events.InterviewCreatedData
    handlers = _class_index(root, "projections.handlers")
    path = os.path.join(root, "src", "projections", "bootstrap.py")
    try:
        text = open(path, encoding="utf-8", errors="ignore").read()
    except OSError:
        return {}
    out: Dict[str, str] = {}
    for m in _REGISTER.finditer(text):
        etype, handler = m.group(1), m.group(2)
        ev = events.get(etype + "Data")              # convention: <Type> -> <Type>Data
        hid = handlers.get(handler)
        if ev and hid:
            out[ev] = hid
    return out
```

- [ ] **Step 3: Cache + splice in `neighbors.py`.** Add to `WalkContext` a memoized `register_map` accessor:

```python
    def event_handler_map(self):
        if getattr(self, "_reg", None) is None:
            from tools.graph.flow import register_map
            self._reg = register_map(self.root)
        return self._reg
```

(add `_reg=None` field). In `_symbol_edges`, symbol branch, `out`/`both`:

```python
        hid = ctx.event_handler_map().get(cid)       # cid is an event-class symbol id
        if hid:
            out.append((f"code:{hid}", Edge("handled_by", addr, f"code:{hid}")))
```

- [ ] **Step 4: Extend the walk test** — add to `tests/graph/test_flow_walk.py`:

```python
def test_event_handled_by_handler():
    sg = walk("code:events.interview_events.InterviewCreatedData", direction="out",
              depth=1, level="symbol")
    assert any(e.type == "handled_by" and e.dst.endswith("InterviewCreatedHandler")
               for e in sg.edges)
```

- [ ] **Step 5: Run + commit**

```bash
python -m pytest tests/graph/test_flow_registry.py tests/graph/test_flow_walk.py -q --no-cov
python -m flake8 tools/graph/flow.py tools/graph/neighbors.py tests/graph/test_flow_registry.py
git add tools/graph/flow.py tools/graph/neighbors.py tests/graph/test_flow_registry.py tests/graph/test_flow_walk.py
git commit -m "feat(graph): handled_by edge (event class -> projection handler) from registry.register parse (symbol-lazy)"
```

---

### Task 4: `writes` edge — projection handler → GlossaryTerm label (Cypher MERGE parse)

**Files:**
- Modify: `tools/graph/flow.py` (`handler_labels`)
- Modify: `tools/graph/neighbors.py` (`_symbol_edges` splices `writes`)
- Test: `tests/graph/test_flow_writes.py`, extend `test_flow_walk.py`

**Interfaces:** `flow.handler_labels(handler_id, root=".") -> List[str]` — glossary-term label ids the handler writes.

- [ ] **Step 1: Write the failing test** — `tests/graph/test_flow_writes.py`:

```python
from tools.graph.flow import handler_labels


def test_handler_labels_from_merge_matching_glossary():
    # SentenceCreatedHandler writes MERGE (s:Fragment ...); Fragment IS a glossary term
    labels = handler_labels("projections.handlers.sentence_handlers.SentenceCreatedHandler", ".")
    assert "Fragment" in labels
```

- [ ] **Step 2: Implement `handler_labels` in `flow.py`.** Find the handler class's file, scan for `MERGE (<var>:<Label>` / `CREATE (<var>:<Label>`, keep labels that are glossary terms:

```python
_MERGE_LABEL = re.compile(r"(?:MERGE|CREATE)\s*\(\s*\w*\s*:\s*(\w+)")


def _module_file(module_id: str, root: str) -> str:
    from tools.code.reader import _module_path
    return _module_path(module_id, root)


def handler_labels(handler_id: str, root: str = ".") -> List[str]:
    module_id = handler_id.rsplit(".", 1)[0]         # handler class -> its module
    from tools.glossary.model import load_glossary
    terms = {t.term for t in load_glossary(os.path.join(root, "docs/glossary"))}
    try:
        text = open(_module_file(module_id, root), encoding="utf-8", errors="ignore").read()
    except OSError:
        return []
    return sorted({lbl for lbl in _MERGE_LABEL.findall(text) if lbl in terms})
```

(Note: scans the whole handler module, so a module with multiple handlers attributes all its labels to each — acceptable v1 coarseness; documented. A per-class body scan is a follow-up.)

- [ ] **Step 3: Cache + splice in `neighbors.py`.** Add a memoized per-handler label cache on `WalkContext`:

```python
    def labels_written_by(self, handler_id):
        cache = self.__dict__.setdefault("_wlabels", {})
        if handler_id not in cache:
            from tools.graph.flow import handler_labels
            cache[handler_id] = handler_labels(handler_id, self.root)
        return cache[handler_id]
```

In `_symbol_edges`, symbol branch, `out`/`both` — **only a registered handler class** writes labels
(gate on the register-map's handler set, so helper functions in a handler module don't get spurious
`writes` edges):

```python
        if cid in set(ctx.event_handler_map().values()):   # cid is a registered handler class
            for lbl in ctx.labels_written_by(cid):
                out.append((f"glossary:{lbl}", Edge("writes", addr, f"glossary:{lbl}")))
```

- [ ] **Step 4: Extend the walk test** — add to `tests/graph/test_flow_walk.py`:

```python
def test_handler_writes_label_and_label_read_by_query():
    # handler -> writes -> glossary:Fragment, and (harvest-grain) a query reads Fragment
    sg = walk("code:projections.handlers.sentence_handlers.SentenceCreatedHandler",
              direction="out", depth=1, level="symbol")
    assert any(e.type == "writes" and e.dst == "glossary:Fragment" for e in sg.edges)


def test_schema_blast_radius_from_a_label():
    # from a label, reach who WRITES it (in) and who READS it (in) — schema-gap #2 traversable
    sg = walk("glossary:Fragment", direction="in", depth=2, level="symbol")
    assert any(a.startswith("graph-queries:") for a in sg.nodes)   # read consumers via reads
```

- [ ] **Step 5: Run + real sanity + commit**

```bash
python -m pytest tests/graph/test_flow_writes.py tests/graph/test_flow_walk.py -q --no-cov
python -c "from tools.graph.flow import handler_labels; print(handler_labels('projections.handlers.resolution_handlers.EntitiesExtractedHandler','.'))"
python -m flake8 tools/graph/flow.py tools/graph/neighbors.py tests/graph/test_flow_writes.py
git add tools/graph/flow.py tools/graph/neighbors.py tests/graph/test_flow_writes.py tests/graph/test_flow_walk.py
git commit -m "feat(graph): writes edge (projection handler -> Neo4j label glossary term) from Cypher MERGE parse"
```

---

### Task 5: Non-blocking checks + eval re-run + ADR + final review

**Files:**
- Modify: `tools/graph/check.py` (two advisory checks)
- Test: `tests/graph/test_flow_checks.py`
- Modify: `evals/graph/RESULTS.md` (record the lift)
- Create: a new ADR

- [ ] **Step 1: Two non-blocking checks** in `tools/graph/check.py` (each returns `List[Finding]`, wired into `run_all` inside a guard, per ADR-0016/0023): `check_unmatched_registration` — a `register("X", ...)` whose `XData` event class isn't found; `check_handler_writes_nothing` — a registered handler whose module has a `MERGE (:Label` but resolves **no** glossary label (a candidate missing glossary term). Test both on a fixture.

- [ ] **Step 2: Full gate.** `make regen-derived && git diff --exit-code` CLEAN (the symbol-lazy edges aren't harvested, so only `reads` may have shifted the graph catalog — already committed in Task 1); `make test-unit` green; `python -m tools.graph check` no dangling; harvest-equivalence (`tests/graph/test_lazy_walk.py`) still green.

- [ ] **Step 3: Re-run the eval (the measurement).** Re-run the three gap/partial scenarios agentically (Mode B autonomous subagent + judge, per `evals/graph/AGENTIC.md`): `pipeline-write-path`, `pipeline-ingestion-flow`, `deploy-neo4j-schema-blast`. The flow is now traversable — record the new verdicts + a note on the lift in `evals/graph/RESULTS.md`. (These scenarios' `gold_context` is unchanged; the point is the agent can now *reach* the flow.) Also update the scenarios' `expected` if a `gap` is now `solvable`, and refresh the deterministic scorecard (`make eval-graph`).

- [ ] **Step 4: ADR.** `python -m tools.adr new "Event-and-label flow overlay is derived, not authored"`. Decision = the four derived edges (emits/handled_by/writes/reads) over existing event-class + glossary-label nodes; **extends ADR-0020**; consistent with **ADR-0027** (symbol-grain, lazy, walk-time) and **ADR-0019** (no authored code→intent; `# emits:` marker only for the dynamic ceiling). `source:` = the spec. `make adr-index && make adr-check`.

- [ ] **Step 5: Commit + final review.**

```bash
git add tools/graph/check.py tests/graph/test_flow_checks.py evals/graph/RESULTS.md evals/graph/scenarios docs/adr
git commit -m "feat(graph): flow-overlay checks + ADR + eval re-run (pipeline/schema gaps now traversable)"
```

Then dispatch the final whole-branch review (most capable model) with a review package (`scripts/review-package "$(git merge-base main HEAD)" HEAD`), and use **superpowers:finishing-a-development-branch**.

## After all tasks

- `walk` at `level="symbol"` traces `command → aggregate → event → handler → :Label`, and `glossary:<Label>` reaches its `written_by` handlers + `read_by` graph-queries (schema-gap #2 closed).
- Module-grain `walk` / `harvest()` unchanged (only `reads` added to harvest); harvest-equivalence green; freshness clean.
- The KG-1 `pipeline-*` / `deploy-neo4j-schema-blast` scenarios measurably lift on re-run.
- New ADR; fidelity ceilings (`emits` dynamic, `writes` module-coarse, `reads` registry-scoped) documented + two advisory checks.

## Knowledge-graph check

Reviewed against `docs/index.md` on 2026-08-23.

| domain | touched? | consulted / reconciled | notes |
| --- | --- | --- | --- |
| graph | yes | `reads` registered+harvested; `emits`/`handled_by`/`writes` walk-time in `neighbors`; `flow.py`; 2 checks | the subject |
| code | yes | `Symbol.emits` + `# emits:` marker in `reader.py` | emit derivation |
| glossary / graph-queries | no (logic) | labels reused as endpoints; query `labels` drive `reads` | reused |
| adr | yes | new ADR (extends 0020; consistent 0027/0019) | — |

**Verdict:** reconciled — four derived edges connect existing event-class symbols and glossary-label terms so the event-sourced write path, the pipeline, and the Neo4j schema-lineage become walkable, symbol-grain and lazy. Only `# emits:` authors the dynamic ceiling; module-grain behavior is provably unchanged. Stage-ordering-as-a-node and infra topology (KG-3) stay deferred; the eval re-run measures the result.
