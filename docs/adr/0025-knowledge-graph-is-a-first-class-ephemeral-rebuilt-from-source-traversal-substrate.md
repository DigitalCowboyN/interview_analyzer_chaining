---
type: ADR
id: 25
title: Knowledge graph is a first-class, ephemeral, rebuilt-from-source traversal substrate
status: accepted
date: 2026-08-15
supersedes: []
superseded_by: []
governs:
  - tools/graph/traverse
  - tools/graph/neighbors
tags: [adr, knowledge-management, okf, graph, tooling, context-engineering]
source: docs/superpowers/specs/2026-08-15-first-class-knowledge-graph-program-design.md
---
## Context
ADR-0020 adopted a typed-edge property-graph model and a `neighbors` view (inbound + outbound
edges of a single node — one hop). It named, as a *risk*, that the graph is "rendered from the
files that are already the source of truth ... not a live store — it can go stale between
renders," and it rejected an RDF/triple-store or a Neo4j load as a new runtime. That was the
right call, but the graph was treated as a rendered *index* rather than a thing you *use*: one
hop deep, no depth, no direction/type/folder entry lens, no notion of the graph as working
context. The purpose of keeping the corpus current (the R1 forward loop, ADR-0023) only pays
off if there is a reliable graph to *query on demand*.

## Decision
Treat the knowledge graph as a **first-class, ephemeral, rebuilt-from-source traversal
substrate** — the repo's spontaneous short-term memory.

- **Materialize from source, every time.** The repo is the single source of truth; the graph
  is a faithful materialization of it, built fresh for the moment it is needed and not
  persisted between uses. This makes ADR-0020's "rendered from source, not a live store" a
  deliberate *feature* (zero drift by construction), not a risk to mitigate.
- **First-class traversal.** A single primitive `walk(entry, direction, depth) -> Subgraph`:
  *entry* = a node or a selector (by **type**, by **folder/path**, or by predicate);
  *direction* = `out | in | both`; *depth* = an integer or *to exhaustion* (walk until it hits
  an end). Choosing depth up front is what makes discovery **progressive**. `neighbors` becomes
  `walk(node, both, 1)`.
- **The subgraph is the artifact.** A walk returns nodes carrying their **claim + context**
  (the record's own content) plus the edges among them. That subgraph *is* the working context
  handed to a model, the input a policy matches on, and the answer to "what governs / depends
  on / verifies this?". Carrying an ADR's scope along a `governs` edge is how a later code edit
  naturally inherits that decision's context.
- **Not the transcript Neo4j.** This graph maps and governs *our own repo*; it is entirely
  separate from the Neo4j read-model that projects interview/transcript data (ADR-0003). They
  are different graphs for different purposes and must not be conflated.

This **extends ADR-0020** (same registry, addressing, and rendered-from-source stance; the
traversal surface and the "ephemeral working memory" purpose are new) and pairs with ADR-0024
(the corpus substrate the walk traverses). It supersedes nothing.

## Consequences
- The graph becomes usable for arbitrary, unpredictable queries — any node, any direction, any
  depth, any time — which is the precondition for hanging governance (rules/policies/hooks) off
  graph shapes later (the program's L3).
- "Generate fresh, then walk" is the discipline: currency (ADR-0023) and completeness
  (ADR-0024) exist precisely so a just-materialized graph can be trusted.
- Rebuild-from-source has a cost at depth/scale (every walk re-reads the corpus). Accepted for
  now; a **materialized cache** (rebuilt on change, e.g. to make deep CI walks cheaper) is an
  acknowledged possible future, deliberately **out of scope** until a real need appears (YAGNI).
  It would be a cache over the same source of truth, never a second authored store.
- We may sometimes need to *prune* a materialized subgraph from an agent's context when the repo
  changes mid-task; noted as a real future concern, not addressed here.

## Alternatives considered
- **A persisted graph store (Neo4j / triple-store) for the knowledge graph** (rejected, as in
  ADR-0020: it introduces a second store that can drift from source, and a new runtime, for a
  model that is faithfully cheap to rebuild; revisit only if walks become slow — and even then
  as a cache, not a source).
- **Reuse the transcript Neo4j** (rejected: a different graph for a different purpose; coupling
  them would entangle repo-governance with product data).
- **Keep `neighbors` (one hop) as the only traversal** (rejected: one hop cannot answer
  reachability or carry multi-hop governance context — the whole point is walking to a chosen
  or exhaustive depth).
- **Bounded depth only, never to-exhaustion** (rejected: exhaustive walks are needed for
  whole-subgraph questions; bounded depth is the progressive-discovery option, not the only
  one).
