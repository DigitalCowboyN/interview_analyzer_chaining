---
type: ADR
id: 28
title: Event-and-label flow overlay is derived, not authored
status: accepted
date: 2026-08-23
supersedes: []
superseded_by: []
governs:
  - tools/graph/flow.py
tags: [adr, knowledge-management, okf, graph, code, flow, tooling]
source: docs/superpowers/specs/2026-08-23-kg2-flow-overlay-design.md
---
## Context

The KG-1 agentic eval proved (and verification confirmed) that three behavioral seams had **no graph
edge**: the event-sourced write path (`command → event → projection → read`), the analysis pipeline
(`ingestion → enrichment → lens → export`), and the Neo4j schema lineage (`projections.schema →
read-consumers`). The graph modeled Python imports; the coupling here is **events** and **Neo4j
labels**, not imports. But that coupling was already latent as nodes/metadata: event payload classes
are symbol nodes, Neo4j labels are `GlossaryTerm`s (`defined_in` the schema), aggregates already `call`
their event constructors, the handler registry is explicit (`register("Type", Handler)`), and graph
queries already carry `labels=[...]`.

## Decision

Add a **derived** event-and-label flow overlay — four edge types over the *existing* event-class
symbols and glossary-label terms, no new node types and (almost) no authoring:

- **`emits` / `emitted_by`** (code symbol → event-class symbol) — the subset of the symbol `calls`
  whose callee is an `events.*Data` class, plus a **`# emits:` marker** for dynamic emission the
  pragmatic call resolution misses. Symbol-grain, walk-time (like `calls`).
- **`handled_by` / `handles`** (event-class symbol → projection-handler symbol) — parsed from
  `registry.register("<Type>", <Handler>(...))`, bridging `"<Type>" → "<Type>Data"`. Symbol-grain,
  walk-time.
- **`writes` / `written_by`** (projection-handler **module** → glossary label) — from Cypher
  `MERGE (:<Label>)` in the handler module, kept only where `<Label>` is a real glossary term.
  **Harvest-grain (module-level)** so the schema blast-radius is discoverable *inbound* from a label
  (symmetric with `reads`) — a build refinement over the spec's initial symbol-grain framing, made
  because symbol-lazy edges aren't inbound-reachable.
- **`reads` / `read_by`** (graph query → glossary label) — from the query's existing `labels=[...]`
  metadata. Harvest-grain, free.

Result: a `level="symbol"` walk traces `command_handler → aggregate → event → handler`, and a walk
inbound from `glossary:<Label>` reaches both its `written_by` handler modules and its `read_by`
queries (which are `consumed_by` the API/export). The three seams are traversable.

This **extends ADR-0020** (adds `emits`/`handled_by`/`writes`/`reads` edge types to the model),
is **consistent with ADR-0027** (the symbol-grain edges are lazy/walk-time; the harvest-grain ones are
cheap), and **preserves ADR-0019** (no authored code→intent links — the flow is structure-derived; the
`# emits:` marker only covers the dynamic-emit ceiling, like `# calls:`/`# verifies:`).

## Consequences

- The KG-1 `pipeline-write-path` / `pipeline-ingestion-flow` (`gap`) and `deploy-neo4j-schema-blast`
  (`partial`) scenarios become traversable; the eval re-run measures the lift (recorded in
  `evals/graph/RESULTS.md`).
- **Fidelity ceilings, documented + guarded** (`tools.graph.check.check_flow_registrations`, advisory):
  `emits` inherits `calls`'s static-resolution ceiling (dynamic emission needs `# emits:`);
  `handled_by` relies on the `register("Type", Handler)` shape + `<Type>Data` convention (an
  unresolvable registration is flagged); `writes` is module-coarse (a multi-handler module attributes
  its labels to the module) and only catches literal `MERGE (:Label)` (a writing handler with no
  glossary label is flagged); `reads` is only as complete as the query `labels` metadata (a raw-Cypher
  read outside the graph-query registry is invisible — a pre-existing blind spot).
- Module-grain `walk` and the generated catalogs are unchanged except for the cheap harvested
  `reads`/`writes` counts; the symbol-lazy `emits`/`handled_by` add no eager cost (harvest-equivalence
  preserved, ADR-0027).

## Alternatives considered

- **Authored flow domain** (`docs/flow/` prose nodes linked to code, drift-guarded). Rejected: the
  coupling is already latent in existing nodes/metadata, so an authored domain re-introduces authoring
  burden and drift for information the graph can derive.
- **Fully derive with no marker** (parse every dynamic emission). Rejected: dynamic construction /
  `_apply(cls(**data))` isn't reliably static; a missing edge in a *flow* graph reads as "no
  connection," so the `# emits:` marker is the honest fallback for the rare case.
- **`writes` at symbol (handler-class) grain.** Rejected during the build: symbol-lazy edges aren't
  discoverable inbound from a label, so the schema blast-radius couldn't find writers. Module-grain +
  harvested makes it bidirectional; per-class attribution is a later refinement if needed.
