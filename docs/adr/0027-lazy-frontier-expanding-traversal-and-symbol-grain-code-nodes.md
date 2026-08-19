---
type: ADR
id: 27
title: Lazy frontier-expanding traversal and symbol-grain code nodes
status: accepted
date: 2026-08-17
supersedes: []
superseded_by: []
governs:
  - tools/graph/traverse.py
  - tools/graph/neighbors.py
  - tools/code/reader.py
tags: [adr, knowledge-management, okf, graph, code, traversal, tooling]
source: docs/superpowers/specs/2026-08-17-symbols-lazy-walk-design.md
---
## Context

The code graph (ADR-0020) stopped at **module** grain (~200 nodes); the ~1,160 functions/classes —
the real subject — were invisible. Adding them naively is a trap: `walk()` built the entire graph via
`harvest()` and *then* traversed, so every walk — even a one-module question — would parse ~1,160
symbol bodies. The research (RANGER, GraphRAG, Glean, Kythe/SCIP, Neo4j supernodes, graph
coarsening/summarization, granularity theory) converges on one answer to "small but numerous nodes":
**hierarchy + on-demand expansion, never flat materialization.**

## Decision

The traversal engine becomes **lazy and frontier-expanding**, and code reaches **symbol** grain,
composed as two axes: **progressive disclosure** (vertical level-of-detail — module → symbols →
signature → docstring) and **progressive discovery** (horizontal frontier expansion — expand a node's
neighbors on demand, coarse-to-fine).

- **`walk` expands via a `neighbors` seam** rather than a pre-built full adjacency. The cheap
  module/doc base is memoized once per walk (via the existing `harvest`); **only symbol bodies are
  parsed on the frontier** — a module's symbols are derived (`ast`) the first time a walk reaches it
  at `level="symbol"`, memoized per module. The ~1,160-symbol cost is **never** built eagerly.
- **Symbols are a deeper `level`** (`package | module | symbol`) — top-level functions/classes and a
  class's methods, existence from the AST, context = signature (free) + docstring (derived). No
  frontmatter; a symbol with no docstring still exists (thin, not absent).
- **`contains` extends to symbol grain**; a **pragmatic `calls`** edge resolves a symbol's local-def +
  imported-symbol + class-instantiation calls from its own file (absolute and relative imports). The
  reverse `called_by` is deferred (see Consequences). Inferred-type
  `obj.method()`, inheritance, and dynamic dispatch are **not** resolved (they'd need whole-program
  type inference — a precomputed global index against the ephemeral model); a `# calls:` marker is the
  escape hatch. The `calls` edge is **walk-time only** — never registered in the harvested edge set,
  so it can't force eager symbol derivation.
- **A `level` disclosure gate** on `walk`: `"module"` (default) never descends into symbols — behavior
  and catalogs provably unchanged (a harvest-equivalence regression proves identical subgraphs);
  `"symbol"` discloses them along the frontier. Symbols inherit authored intent by walking **up**
  `contained_by` (ADR-0019); no new authored code→intent links.

This **extends ADR-0025** — the ephemeral, rebuilt-from-source substrate matures from
full-rebuild-per-call to **incremental, lazy per-frontier expansion** (still ephemeral; now the
per-node symbol cost is paid only where the walk goes) — and **extends ADR-0020** (adds the `symbol`
level value and the walk-time `calls` edge to the model; `called_by` reserved, deferred). Consistent
with **ADR-0019**.

## Consequences

- Node count becomes almost irrelevant to cost: a walk pays only for the frontier it chooses (discovery)
  and the detail it discloses (disclosure). A default module-grain walk parses zero symbols; a
  symbol-level walk parses only visited modules (a frontier-laziness test asserts this).
- The graph reaches the real code: an agent can walk from a symbol to its callees (`calls`), up to its
  module and governing capability/ADR (`contained_by` walk-up), and disclose signature→docstring
  on demand.
- Fidelity ceiling: symbol `calls` are the statically-decidable subset (local + imported, absolute and
  relative). The **reverse `called_by` is deferred** — finding a symbol's callers requires scanning
  bodies the walk hasn't visited, against the frontier-lazy model; a walk `in` from a symbol reaches its
  container (via `contained_by`) but not its callers this milestone. Documented, not hidden.
- Module-grain behavior, generated catalogs, checks, and the freshness gate are unchanged (harvest is
  retained for the whole-graph renders; symbols/`calls` never enter it).
- Deferred (recorded in the spec): **full semantic resolution** (inferred-type/inheritance calls — a
  global static-analysis index); **authored, linked, guarded flow/architecture nodes** for the
  behavioral seams pragmatic resolution can't reach (the event-sourced command→event→projection path)
  — the natural next milestone; **derived subgraph summaries**; symbol **body** in context.

## Alternatives considered

- **Eager symbol materialization + hierarchical summaries** (GraphRAG-style). Rejected: rebuilds
  ~1,160 nodes every walk and needs summary generation — against the ephemeral substrate.
- **Separate precomputed symbol index** (SCIP/Kythe/CodeQL model). Rejected: a second store and a
  global precompute step, against "one ephemeral graph rebuilt from source."
- **Full semantic call resolution now.** Rejected as a milestone of its own; the hard cases are a
  minority of real edges here and would reintroduce a precomputed global index. The `# calls:` marker
  covers the rare edge that matters.
- **Pure per-node lazy expansion (never call `harvest`).** Rejected as the *literal* engine: at module
  grain there are no symbols, so `harvest` is already the cheap base, and inbound edges need an index
  regardless — pure per-node buys ~nothing there while adding complexity. The realization memoizes the
  cheap base and makes only symbol expansion frontier-lazy.
