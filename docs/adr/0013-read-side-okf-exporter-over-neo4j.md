---
type: ADR
id: 13
title: Read-side OKF exporter over Neo4j
status: accepted
date: 2026-07-10
supersedes: []
superseded_by: []
tags: [okf, export, neo4j]
governs:
  - src/export/
source: docs/superpowers/specs/2026-07-10-okf-export-design.md
---
## Context
Interviews need to be exported as OKF bundles (and queried richly) for both
human and agent consumption. Rendering could pull from aggregate/ESDB state,
from the projected Neo4j graph, or a hybrid of both.

## Decision
The exporter is a projection renderer: parameterized Cypher pulls the
interview's subgraph from Neo4j and pure functions write markdown; the same
reader module backs the REST query endpoints, giving bundles and queries one
data-access layer. Projection lag is handled by an explicit consistency
guard, not ignored.

## Consequences
Exports and REST queries stay consistent with each other (same reader);
Interview aggregate state can stay skeletal (no need to fatten it for
rendering); a projection-lag guard produces a retryable error (409) instead
of a silently incomplete bundle.

## Alternatives considered
Aggregate-side rendering (rejected: aggregate state deliberately keeps only
skeletal lens-item state, forcing raw event re-reads or fattened state;
by-speaker rollups are cross-interview and need Neo4j regardless); a hybrid —
bundles from events, queries from Neo4j (rejected: two data models for the
same content, no consumer-visible gain).
