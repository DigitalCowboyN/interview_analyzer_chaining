---
type: ADR
id: 1
title: EventStoreDB is the single source of truth
status: accepted
date: 2026-07-04
supersedes: []
superseded_by: []
tags: [event-sourcing, write-side]
governs:
  - src/persistence/
source: docs/architecture/README.md
---
## Context
The system needs an authoritative record of everything that happened so read
models (Neo4j) can be rebuilt and corrections replayed.

## Decision
EventStoreDB holds the canonical event log; Neo4j is a disposable projection.

## Consequences
Read models are rebuildable; all writes go through events; projection lag is a
first-class concern.

## Alternatives considered
Neo4j-as-source-of-truth (rejected: no replay, corrections destructive).
