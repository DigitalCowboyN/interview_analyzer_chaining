# Documentation

> **Start here:** [ROADMAP.md](ROADMAP.md) — milestone status, test/coverage
> numbers, and the plan of record.

## Primary documents

| Document | Purpose |
|----------|---------|
| **[ROADMAP.md](ROADMAP.md)** | Canonical roadmap — milestones, status, current stats |
| **[architecture/](architecture/)** | Current architecture: system overview, data flow, event sourcing, database schema |
| **[onboarding/](onboarding/)** | Setup, configuration, dev workflow, troubleshooting |

## Architecture reference

The [architecture/](architecture/) folder is the current, maintained
description of the system:

| Document | Description |
|----------|-------------|
| [architecture/README.md](architecture/README.md) | Index + the system in one diagram + the layered "Mine" model |
| [architecture/system-overview.md](architecture/system-overview.md) | System context, containers, request/projection paths |
| [architecture/data-flow.md](architecture/data-flow.md) | The layered analysis pipeline, ingest → export/ask |
| [architecture/event-sourcing.md](architecture/event-sourcing.md) | Aggregates, events, projection ordering, the SSE bridge |
| [architecture/database-schema.md](architecture/database-schema.md) | Neo4j read-model nodes and relationships |

## Archive

[archive/](archive/) holds historical milestone notes and session summaries
(M2.x dual-write, Phase 2, test-migration analyses, etc.). They are **superseded
and kept for reference only** — do not treat them as current. Where an archived
doc described architecture (e.g. the M2.8 summaries), the current version lives
in [architecture/](architecture/) and [ROADMAP.md](ROADMAP.md).

## Current status

See [ROADMAP.md](ROADMAP.md) for the authoritative, always-updated status
(current milestone, test counts, coverage). It is the single source of truth for
project state — this index intentionally does not duplicate those numbers.
