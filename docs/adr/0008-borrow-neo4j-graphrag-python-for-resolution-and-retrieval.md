---
type: ADR
id: 8
title: Borrow neo4j-graphrag-python for resolution/retrieval, not its pipeline
status: superseded
date: 2026-07-04
supersedes: []
superseded_by: [14]
tags: [graphrag, entity-resolution, dependency]
source: docs/superpowers/specs/2026-07-04-mine-layers-design.md
---
## Context
Entity resolution (canonicalizing "the dashboard" / "our analytics
dashboard") and later ask-the-corpus retrieval both have needs that the
`neo4j-graphrag-python` library addresses out of the box, but its
construction pipeline writes directly to Neo4j.

## Decision
Borrow `neo4j-graphrag-python`'s entity-resolution utilities now, and (for a
later phase) its hybrid retrievers (vector + fulltext + Cypher) for
ask-the-corpus retrieval — a borrowed component, not an adopted pipeline.

## Consequences
Entity resolution gets a head start from a maintained library without
bypassing event sourcing. The retrieval half of this decision was later
revisited once the ask-the-corpus design (M4.6) landed — see ADR-0014, which
supersedes the retriever-borrowing line of this decision.

## Alternatives considered
Adopting `neo4j-graphrag-python`'s construction pipeline wholesale (rejected:
writes directly to Neo4j, bypassing event sourcing; loses edit protection,
replay, and lens re-runs).
