# Symbols + lazy frontier-expanding walk (design)

**Status:** proposed (brainstorm dialogue with owner, 2026-08-17).
**Program:** the first-class knowledge graph
(`docs/superpowers/specs/2026-08-15-first-class-knowledge-graph-program-design.md`). This is the
next code-grain milestone after hierarchical code intake (PR #43) and the docstring/evals work
(PR #44). It pushes code nodes to **symbol** grain (functions/classes/methods) and, because that
would otherwise tax every traversal, makes `walk` **lazy and frontier-expanding**.

## Why (and why together)

The code graph today stops at **module** grain (~200 nodes). The real subject — the ~1,160
functions/classes — is invisible. Adding them naively is a trap: today `walk()` calls `harvest()`,
which builds the **entire** graph (parses every module) and *then* BFS-traverses. Add symbols and
every walk — even a one-module question — parses ~1,160 symbol bodies. So symbols force the
traversal engine to change. Owner decision (2026-08-17): do both in **one milestone**, but structure
the work engine-first (regression-tested to reproduce today's subgraphs) so a bug is isolable.

The research (RANGER, GraphRAG, Glean, Kythe/SCIP, Neo4j supernodes, KG summarization/coarsening,
granularity theory) converges on one answer to "small but numerous nodes": **hierarchy + on-demand
expansion, never flat materialization** — realized here as two composing axes:

- **Progressive disclosure (vertical, level-of-detail):** same region, more depth on request —
  module → its symbols → a symbol's signature → its docstring. Buys *precision*.
- **Progressive discovery (horizontal, frontier expansion — the RANGER move):** walk to a node,
  materialize only its neighbors, decide where to go, expand again. Buys *gradual reach* without
  loading the whole graph.

## Decision 1 — Lazy, frontier-expanding `walk` (the engine)

`walk` becomes **BFS over a `neighbors` seam** rather than over a pre-built full adjacency, so that
the expensive per-node work — **symbol bodies** — is done **strictly on the frontier**: a module's
symbols are parsed only when the walk actually reaches that module at `level="symbol"`. The
~1,160-symbol cost is *never* built eagerly.

The **realization is pragmatic** (owner decision, 2026-08-17): pure per-node expansion of *everything*
buys ~nothing at module grain — at module grain there are no symbols, so today's `harvest` is already
"the cheap base with zero symbol cost," and inbound edges ("who imports/implements me") can't be
computed from a node's own file anyway (they need an index). So:

- **The cheap module/doc base is memoized once per `walk`** (via the existing `harvest`, or an
  equivalent cheap index) and cached on a `WalkContext` — no symbol bodies, fast (~200 nodes).
- **Only symbol expansion is frontier-lazy** — `neighbors` splices a module's symbols (nodes +
  `contains` + `calls`) in on demand, the first time the frontier reaches that module at
  `level="symbol"`, memoized per module.

- **`neighbors(addr, direction, ctx) -> list[(neighbor_addr, Edge)]`** — the seam. At `level="module"`
  it returns the cached base edges. At `level="symbol"` it additionally returns a visited module's
  symbol edges (computed lazily from *that module's* AST).
  - **Inbound authored/intent edges** (`implements`/`governs`/`verifies`/`consumed_by`): come from
    the *doc/test* side pointing *at* code. To answer "what implements code:X" without a full scan,
    build a **cheap reverse index** of the small authored domains (capabilities, ADRs, tests,
    prompts, glossary — dozens of files) **once per walk, cached**. This is the honest nuance: we
    never parse *unvisited code/symbols* (where the ~1,160-node cost lives); the tiny doc domains
    are indexed cheaply. Cache is per-`walk` call (still ephemeral, ADR-0025).
- **`walk` becomes BFS over the `neighbors` seam** (base adjacency memoized once per walk; symbol
  edges spliced lazily on top). Same `Subgraph` output shape (`nodes`, induced `edges`), same
  `direction`/`depth` semantics.
- **`harvest()` stays and is unchanged** — it still backs the generated catalogs (`docs/graph/*`) and
  the non-blocking checks (they genuinely need the whole graph). Symbols/`calls` never enter
  `harvest` (they'd force eager derivation); they live only on the lazy `walk` path.
- **Correctness gate (the point of engine-first):** for the existing 8 node types, lazy `walk`
  must produce **identical** `Subgraph`s to today's harvest-based `walk` for the same
  entry/direction/depth. A regression test drives real entries through both and asserts equality.

## Decision 2 — Symbols as a deeper `level`

The `level` axis gains **`symbol`** (`package | module | symbol`). A symbol node is a top-level
`FunctionDef`/`AsyncFunctionDef`/`ClassDef` in a module, or a method (a `FunctionDef` inside a
`ClassDef`).

- **Addressing** (dotted, extends the scheme): module-level symbol `code:tools.graph.traverse.walk`;
  method `code:tools.graph.reader.Edge.__init__`. `kind` (`function | class | method`) is a node
  property, not part of the address.
- **Existence is structural (AST), never authored.** `ast.parse` → the def nodes. A symbol with no
  docstring still exists (its context is just its signature). This is the *same* pattern packages
  and modules already use — one level deeper.
- **Context = signature (free) + docstring (derived).** The signature is rendered from the AST
  (name, args, defaults, return annotation, decorators): `walk(entry, direction="both", depth=None,
  root=".") -> Subgraph`. The docstring is `ast.get_docstring(node)` where present; where absent it
  becomes a **symbol-level line in the same backlog** (`check_missing_docstring` extends to symbols,
  but a symbol is "thin, not empty" — the signature carries real information, so symbol docstrings
  are lower-priority than the module backlog was).

## Decision 3 — `contains` and pragmatic `calls` edges

- **`contains`/`contained_by`** extends to symbol grain: a module contains its top-level
  functions/classes; a class contains its methods. Pure AST nesting. This is the disclosure spine.
- **`calls`** (walk-time edge, CodeUnit→CodeUnit; reverse `called_by` deferred — finding callers needs
  scanning unvisited bodies, against the frontier-lazy model) — **pragmatic resolution**, done the lazy
  way: when a symbol is expanded, parse *its body* and resolve `Call` targets against *that file's*
  imports (absolute and relative) + local defs:
  - `render_index(...)` (local def or `from .render import render_index`) → `calls` →
    `code:tools.code.render.render_index`.
  - `import x; x.foo()` / `from x import foo; foo()` → `code:x.foo`.
  - `Bar()` (imported/local class) → its class node.
  - **Ceiling (not emitted):** `obj.method()` on a statically-unknown type, `self.foo()` needing the
    inheritance chain, `getattr`/dynamic dispatch. These require whole-program type inference — a
    global precompute that fights the ephemeral/lazy model — so we do **not** do them here.
  - **Escape hatch:** a `# calls: code:x.y` marker (the macro sibling of `# verifies:`) lets a human
    assert a specific dynamic edge that matters, without an analyzer.
- **Intent edges are unchanged and stay coarse.** `implements`/`governs`/`verifies` are authored at
  module/package grain (or point at code via path). Symbols inherit intent by walking **up**
  `contained_by` (symbol → module → its capability/ADR) — ADR-0019 preserved; we never hand-wire
  symbols to capabilities.

## Decision 4 — The disclosure gate

`walk` gains a **`level`** parameter (the vertical gate): `"module"` (default — today's behavior,
never descends into symbols) vs `"symbol"` (may descend to symbol grain). At `"module"` the
`contains` edges into symbols are simply not expanded; at `"symbol"` they are. Combined with lazy
expansion, a default (module) walk never parses a single symbol body. Disclosure of a symbol's own
detail is layered: **name → signature → docstring** (body is out of scope — the agent reads the file
if it needs the implementation).

## Retained / not changed

- `discover_units` (whole-graph node registry) and the catalogs still work at module grain by
  default; symbol derivation is opt-in (a `level` argument), so `make code-index`, `graph-index`,
  and the checks are unaffected unless explicitly asked for symbols.
- The freshness gate, corpus, capability coverage, docstring-backlog machinery all carry over.

## Scope

**This milestone:** lazy frontier-expanding `walk` (with harvest-equivalence regression); the
`symbol` level (AST function/class/method nodes, signature+docstring context); `contains` at symbol
grain; pragmatic `calls`/`called_by` edges + `# calls:` marker; the `level` disclosure gate on
`walk`; symbol docstrings folded into the existing backlog.

**Deferred:**
- **Full semantic resolution** (inferred-type method calls, inheritance, re-export aliasing) — a
  global static-analysis project (SCIP/Kythe/LSP-grade) that would reintroduce a precomputed index;
  out of the ephemeral model.
- **Authored, linked, guarded flow/architecture nodes** — the honest fill for the *behavioral*
  seams pragmatic resolution can't reach (e.g. the event-sourced command→event→projection→read-model
  path). These are authored intent, linked down to code, and drift-guarded like ADRs — the natural
  *next* milestone after symbols (recorded here so it's on the books; see the "supplemental docs"
  dialogue).
- **Derived subgraph summaries** (GraphRAG-style prose roll-ups of a walked region) — an amplifier of
  existing structure, worth doing once traversal is confident; separate.
- Symbol **body** in node context; symbol-grain rendering in the generated catalogs.

## ADR

New ADR: **lazy, frontier-expanding traversal + symbol-grain code nodes.** It **extends ADR-0025**
(the ephemeral rebuilt-from-source substrate matures from full-rebuild-per-call to lazy per-node
expansion — still ephemeral, now incremental) and **ADR-0020** (adds the `symbol` level value and the
`calls`/`called_by` edge type). Consistent with **ADR-0019** (symbols inherit authored intent by
walk-up; no new authored code→intent links). `source:` = this spec.

## Testing

- **Harvest-equivalence (engine):** for a set of real entries × directions × depths, lazy `walk`
  returns a `Subgraph` equal (nodes + induced edges) to today's harvest-based `walk`. The single most
  important test — it proves the rewrite before symbols pile on.
- **Laziness:** walking a module-grain entry parses only the files on the frontier (assert via a
  counter/spy on `ast.parse` / file-open that unvisited modules are never read).
- **Symbol discovery:** a fixture module yields the right symbol nodes (top-level fns/classes +
  methods) with correct dotted ids, `kind`, and signature; a dunder/nested edge case behaves.
- **Symbol context:** signature rendered correctly; docstring picked up where present; absent
  docstring → thin (signature-only) node, and a symbol-backlog entry.
- **`contains` at symbol grain:** module contains its symbols; class contains its methods;
  `walk(symbol, "in")` reaches its module then its capability/ADR (walk-up works).
- **Pragmatic `calls`:** local-def call, imported-symbol call, and class instantiation each emit the
  right `calls` edge; an `obj.method()` on unknown type emits none; a `# calls:` marker emits the
  asserted edge.
- **Disclosure gate:** `walk(module, level="module")` surfaces no symbols; `level="symbol"` surfaces
  them; no symbol body is parsed at module level.
- **No dangling / freshness:** every emitted `calls`/`contains` symbol edge resolves; the catalogs
  and freshness gate are unchanged at default (module) grain.

## Knowledge-graph check

Reviewed against `docs/index.md` on 2026-08-17.

| domain | touched? | note |
| --- | --- | --- |
| graph | yes — `walk` rewritten lazy (neighbor-on-demand + cached reverse index for intent edges); `level` param; `calls` edge; harvest kept for catalogs/checks | the engine subject |
| code | yes — symbol discovery from AST (fn/class/method nodes, signature+docstring context); `contains` + pragmatic `calls` at symbol grain; `# calls:` marker | the grain subject |
| capabilities / adr / use-cases / tests / prompts / glossary | yes (read-only) — a cheap per-walk reverse index over these small domains resolves inbound intent edges lazily | consumed, not changed |
| adr | yes — new ADR (extends 0025 + 0020, consistent with 0019) | — |

**Verdict:** reconciled — the traversal engine goes lazy/frontier-expanding (extending ADR-0025's
ephemeral substrate to incremental) and code reaches symbol grain with structure-derived nodes,
`contains`, and pragmatic `calls` (a documented fidelity ceiling with a `# calls:` marker escape
hatch). Progressive disclosure (vertical LOD) + progressive discovery (horizontal frontier) make node
count almost irrelevant — you pay only for the frontier you choose and the detail you disclose. Full
semantic resolution, authored linked flow-nodes, and derived summaries are explicitly deferred.
