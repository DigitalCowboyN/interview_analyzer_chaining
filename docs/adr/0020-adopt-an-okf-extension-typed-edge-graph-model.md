---
type: ADR
id: 20
title: Adopt an OKF-extension typed-edge graph model
status: accepted
date: 2026-08-06
supersedes: []
superseded_by: []
governs:
  - tools/graph/
tags: [adr, knowledge-management, okf, graph, tooling]
source: docs/superpowers/specs/2026-08-06-graph-links-model-design.md
---
## Context
Every domain so far invented its own link field and its own renderer (capabilities'
`implemented_by`, ADRs' `governs`/`supersedes`, code's dependency edges, …) — no
cross-domain graph existed, so "what does changing this touch" had no single answer.
OKF's native links are untyped, undirected body links: "the specific kind (parent/child,
references, depends-on) is conveyed by the surrounding prose, not by the link itself."
OKF v0.2 defines no typed-relationship frontmatter, leaving typing an explicitly open
extension point rather than something to bolt on ad hoc per domain.

## Decision
Adopt a property-graph-shaped, **extensible edge registry** (`tools/graph/registry.py`):
edges are first-class, typed, directed, and carry **properties** (e.g. a future
`verifies` edge's `test_type: unit|integration|e2e`), named with verbs from the
software-traceability vocabulary (`implements`, `depends_on`, `governs`, `supersedes`,
and reserved `verifies`, `fulfills`, `refines`, `derives`). Nodes are addressed
cross-domain as `<domain>:<id>` (`code:api`, `capability:import-transcripts`,
`adr:0019`). A **registry-driven** harvester/renderer/guard in `tools/graph/` builds the
unified graph with **no new authoring surface** — edges are read from each domain's
existing frontmatter fields and derivations (e.g. `implemented_by`, `governs`,
`tools.code.dep_edges`), never authored twice. `make graph-check` is a **non-blocking**
cross-domain integrity sweep (dangling endpoints, registry integrity, index-sync) that
**complements, not replaces**, per-domain checks, and runs alongside `adr-check` in
`.githooks/pre-commit`; `make health` runs the full domain-check + graph-check sweep.
Adding an edge (tests' `verifies`, use-cases' `fulfills`) is a one-entry registry
addition, not a redesign. The graph domain self-registers in the code map and claims an
operations capability, like every other domain tool.

## Consequences
Cross-domain breakage from a single-domain edit is now caught: a code unit renamed in a
code-only commit passes `code-check` (which doesn't know about capabilities or ADRs) but
now leaves a capability's `implements` or an ADR's `governs` dangling — only the graph
guard sees the whole graph and flagged exactly this during this round's build. The whole
graph becomes traversable (`neighbors` CLI: inbound + outbound edges of any node) and
rendered (`docs/graph/index.md` catalog + meta-schema, `docs/graph/graph.md` full
instance graph by edge type). Extensibility is real, not aspirational: reserved edge and
node types (`verifies`, `fulfills`, `Test`, `UseCase`, …) cost nothing until their round.
Risk: the graph view is file-rendered from the domains' own sources, not a live store —
it can go stale between renders (caught by index-sync, not prevented); a full instance
graph is a hairball at scale, so it is split into one Mermaid section per edge type for
readability rather than rendered as a single diagram.

## Alternatives considered
Untyped OKF body links only (rejected: no machine-readable typing or traversal — every
domain would keep hand-rolling its own convention); an RDF/triple-store or an actual
Neo4j load of the graph (rejected: property-graph is the conceptual model we're adopting,
not a mandate for a new runtime — the graph is rendered from the files that are already
the source of truth); authoring an `implements`/edge inverse as markers in `src/`/`tools/`
(rejected: code stays untouched, the inverse is a free derivation from the authored
field, matching ADR-0019's stance on `implements`); replacing per-domain checks with the
graph check (rejected: per-domain checks are the right granularity for fast, scoped,
focused-work feedback — the graph check's unique value is the cross-domain case no
single-domain check can see). Refines the domain-family ADRs (0015 ADR corpus, 0016
knowledge cascade, 0017/0018 capabilities); supersedes nothing.
