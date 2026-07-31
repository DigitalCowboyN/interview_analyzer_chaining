---
type: ADR
id: 12
title: Fragment dual-label rename, wire format stays frozen
status: accepted
date: 2026-07-10
supersedes: []
superseded_by: []
tags: [schema-v2, migration, wire-format]
source: docs/superpowers/specs/2026-07-10-layer4-schema-v2-design.md
---
## Context
The graph label `:Sentence` needed a more accurate name (`:Fragment`) to
reflect what Layer 1 actually delivered, but the underlying event type
names, `aggregate_type` strings, and stream names (`Sentence-{id}`) are
frozen wire format (ADR-0004) and can never change.

## Decision
New projections MERGE both labels (`:Fragment:Sentence`); an idempotent
migration Cypher (`MATCH (s:Sentence) WHERE NOT s:Fragment SET s:Fragment`)
backfills `:Fragment` onto existing nodes; every query the system owns moves
to `:Fragment`; the `:Sentence` label is retained as a deprecation shim
(dropping it is a later backlog item) while wire-format identifiers stay
untouched.

## Consequences
Zero rename churn on new code (born writing `:Fragment`); the Layer 4 smoke
asserts the dual-label invariant (`every :Sentence is :Fragment and vice
versa`); per-model vector index DDL stays on `:Sentence` for the migration
window to avoid creating duplicate indexes for nothing.

## Alternatives considered
Renaming the wire-format identifiers alongside the graph label (rejected:
breaks replay of historical events, violates ADR-0004).
