---
type: ADR
id: 21
title: Use-cases domain as the graph source layer
status: accepted
date: 2026-08-06
supersedes: []
superseded_by: []
tags: [adr, knowledge-management, okf, use-cases, graph, tooling]
source: docs/superpowers/specs/2026-08-06-use-cases-domain-design.md
---
## Context

The guarded knowledge graph had capabilities (durable intent), code, ADRs, and a
cross-domain edge layer — but no record of the **user-centered "why"** those capabilities
serve. Nothing explained *why* a capability exists, or why one is only partially
implemented. Industry practice (Cockburn use cases; the Requirements Traceability Matrix
literature; the requirements-coverage frameworks) treats requirement / user story /
feature / use case as **one artifact at different fidelity**, each carrying acceptance
criteria, and traces intent → implementation → test to expose coverage gaps. We had the
implementation half of that matrix and none of the intent half.

## Decision

Adopt a **use-cases domain** (`docs/use-cases/`, `tools/usecase/`) as the graph's source
layer:

- **One `UseCase` node type**, fidelity carried by an **open `form` axis**
  (`user-story | feature | requirement | use-case`), not four node types. Required core
  (form, category, actor, statement, acceptance_criteria, fulfilled_by) plus an optional
  Cockburn block (level, preconditions, main_scenario, extensions, end_conditions).
- **`category` reuses the capability axis** (`product | operations | supporting | …`,
  open) — use-cases are product, operations, or supporting, not product-only.
- **Acceptance criteria are a list of strings** (Given/When/Then or rule sentences);
  empty is legal and surfaced as an advisory, not hidden.
- **Coverage is derived, never stored** — `NOT_COVERED | PARTIALLY_COVERED |
  FULLY_COVERED`, computed transitively from `fulfilled_by` edges and each fulfilling
  capability's implementation degree. In this round `FULLY_COVERED` means *implemented*;
  verification-grade coverage arrives with the tests domain (`verifies`) without renaming
  the states.
- **The `fulfilled_by` edge is authored on the use-case side** (`fulfilled_by:` on the
  `UseCase`; `fulfills` is the computed inverse). This keeps **capability files
  read-only** — recording that a capability serves an intent never edits the capability.
- **The corpus is reconstructed, not restated**: use-cases recover the originating human
  problem behind a cluster of capabilities. A correct corpus **overshoots** the current
  build (uncovered/partial intents are the expected, desired signal, not defects).

Guarded by a non-blocking `make usecase-check`; cross-domain endpoint integrity stays with
`graph-check`. Extends ADR-0019 (capabilities-as-intent) and ADR-0020 (typed-edge graph);
supersedes nothing.

## Consequences

- The graph is now a Requirements Traceability Matrix: use-case → capability → code
  (→ test, next round). Coverage gaps and aspirational intent are visible, not implicit.
- The `verifies`/tests edge and a use-case↔use-case `refines` edge remain reserved
  registry additions, proving the model extends without rework.
- "support" vs "supporting": the canonical axis value is **`supporting`** (shared with the
  capability domain); documentation must use it verbatim or the guard flags it.
- A single additive self-registration capability (`map-use-cases`) claims the new tooling;
  no existing capability's intent changed.

## Alternatives considered

- **Separate node types per form** (UserStory/Feature/Requirement/UseCase) — rejected:
  forks the schema and multiplies edge targets, fighting the industry consensus that these
  are one artifact at different fidelity.
- **A stored coverage/status field** — rejected: rots; derivation from links is the same
  discipline capabilities already use.
- **Authoring `fulfills` on the capability side** — rejected: would edit capability files,
  violating the read-only constraint; authoring the inverse on the use-case side yields an
  identical graph.
