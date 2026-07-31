---
type: ADR
id: 3
title: The projection service is the sole writer to Neo4j
status: accepted
date: 2026-07-04
supersedes: []
superseded_by: []
tags: [event-sourcing, projection, neo4j]
governs:
  - src/projections/
source: docs/architecture/README.md
---
## Context
Neo4j is meant to be a derived, rebuildable read view of the event log. If
multiple services could write to it directly, replay/rebuild guarantees and
causal ordering would break.

## Decision
The projection service is the only writer to Neo4j. Events are replayed in
each stream's commit-position (causal) order per lane.

## Consequences
Neo4j can be dropped and rebuilt from ESDB at any time; all Neo4j state is
provably derived; new consumers of graph data must go through projection
handlers, never direct writes; projection lag becomes a first-class
operational concern.

## Alternatives considered
Multiple services writing to Neo4j directly (rejected: breaks the
single-writer replay guarantee and causal ordering).
