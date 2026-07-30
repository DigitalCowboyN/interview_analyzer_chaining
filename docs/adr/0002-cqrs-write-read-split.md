---
type: ADR
id: 2
title: CQRS write/read split
status: accepted
date: 2026-07-04
supersedes: []
superseded_by: []
tags: [cqrs, architecture]
source: docs/architecture/README.md
---
## Context
Write-side correctness (aggregates, commands, event emission) and read-side
query concerns (Neo4j graph queries, UI rendering) have different shapes and
scaling needs; coupling them makes both harder to evolve.

## Decision
The write side (aggregates + commands, emitting events to ESDB) and the read
side (Neo4j + queries) are kept strictly separate. The UI mirrors this split:
a workbench surface for writing, a gallery surface for reading.

## Consequences
Commands never read from Neo4j to decide correctness; queries never write;
the two-surface UI design is a direct reflection of this split; read-side
changes (new query shapes, projections) don't touch aggregate logic.

## Alternatives considered
A single unified read/write model (rejected: couples query performance
concerns to write-side correctness and complicates rebuilding/replaying read
state independently).
