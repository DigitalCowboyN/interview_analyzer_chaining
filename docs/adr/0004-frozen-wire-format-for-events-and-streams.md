---
type: ADR
id: 4
title: Frozen wire format for event types and stream names
status: accepted
date: 2026-07-04
supersedes: []
superseded_by: []
tags: [event-sourcing, wire-format, compatibility]
governs:
  - src/events/
source: docs/architecture/README.md
---
## Context
Stored events are immutable history. If event type names, aggregate type
strings, or stream naming ever changed, historical events would become
unreadable or ambiguous under replay.

## Decision
Event type names, the `Sentence` aggregate type, and `Sentence-{uuid}` stream
names never change, even as the projected graph label evolves (e.g.
`:Sentence` dropped in favor of `:Fragment`). New optional envelope/metadata
fields (like `project_id`) are additive only.

## Consequences
Graph labels, Python naming, and docs are free to rename for clarity while
the append-only event log stays valid forever; renames on the read side
require dual-label or alias strategies instead of touching the write side
(see ADR-0012).

## Alternatives considered
Renaming wire-format identifiers alongside conceptual renames (rejected:
breaks replay of historical events).
