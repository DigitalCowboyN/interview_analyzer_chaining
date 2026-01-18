# Project Roadmap

> **This is the canonical project roadmap. Update this document when milestone status changes.**

---

## Quick Status

**Last Updated:** 2026-01-18

| Milestone | Status | Description |
|-----------|--------|-------------|
| M1 | ✅ Complete | Core Plumbing (events, ESDB, aggregates) |
| M2.1 | ✅ Complete | Command Layer |
| M2.2 | ✅ Complete | Dual-Write Integration |
| M2.3-M2.5 | ✅ Complete | Projection Infrastructure |
| M2.7 | ✅ Complete | Testing & Validation |
| M2.8 | ✅ Complete | Event-Sourced Architecture (Production Ready) |
| **M2.9** | ⏳ **Next** | User Edit API |
| M3.0 | 📋 Planned | Remove Dual-Write + neo4j 6.x |
| M3.1 | 📋 Planned | Vector Search |
| M3.2 | 📋 Planned | AI Agent Upgrade (openai 2.x) |
| M3.3 | 📋 Planned | Infrastructure Upgrades |

**Current Phase:** M2.9 (User Edit API)
**Tests:** 691 passing, 84 skipped | **Coverage:** 72.2%

---

## Milestone Checklist

### M2.9: User Edit API ⏳ NEXT

- [ ] Review existing `src/api/routers/edits.py` implementation
- [ ] Complete `PUT /interviews/{id}/sentences/{id}` endpoint
- [ ] Complete `PUT /interviews/{id}/sentences/{id}/analysis` endpoint
- [ ] Integration with command handlers
- [ ] Return accepted status with version
- [ ] E2E tests passing
- [ ] Documentation updated

**Dependencies:** None (can start now)

---

### M3.0: Remove Dual-Write 📋 PLANNED

- [ ] Remove direct Neo4j writes from pipeline
- [ ] Projection service becomes SOLE writer
- [ ] Remove deprecated code paths
- [ ] Remove 27 legacy tests
- [ ] Upgrade neo4j 5.28.1 → 6.x
- [ ] Update documentation
- [ ] 1-2 weeks production validation

**Dependencies:** M2.9 complete

---

### M3.1: Vector Search 📋 PLANNED

- [ ] Store sentence embeddings in Neo4j
- [ ] Semantic similarity search endpoints
- [ ] Vector-based clustering for topics
- [ ] Enhanced keyword/topic extraction

**Dependencies:** M3.0 complete (neo4j 6.x required)

---

### M3.2: AI Agent Upgrade 📋 PLANNED

- [ ] Upgrade openai 1.93.3 → 2.x
- [ ] Refactor `src/agents/` implementations
- [ ] Evaluate OpenAI Agents SDK
- [ ] Update anthropic SDK

**Dependencies:** M3.0 complete

---

### M3.3: Infrastructure Upgrades 📋 PLANNED

- [ ] Upgrade pytest 8.3.3 → 9.x
- [ ] Upgrade pytest-cov 6.0.0 → 7.x
- [ ] Upgrade redis 6.2.0 → 7.x
- [ ] Upgrade isort 5.13.2 → 7.x
- [ ] Update performance baselines

**Dependencies:** M3.0 complete

---

## Completed Milestones

<details>
<summary>M1: Core Plumbing ✅</summary>

- Event envelope and domain events
- EventStoreDB client and connection management
- Repository pattern for aggregates
- Interview and Sentence aggregates

</details>

<details>
<summary>M2.1: Command Layer ✅</summary>

- Command base classes and handlers
- Interview and Sentence commands
- Actor tracking and correlation IDs

</details>

<details>
<summary>M2.2: Dual-Write Integration ✅</summary>

- Event-first dual-write pattern
- Pipeline emits events before Neo4j writes
- Event failures abort operations (correct behavior)

</details>

<details>
<summary>M2.3-M2.5: Projection Infrastructure ✅</summary>

- Lane Manager with 12 configurable lanes
- Subscription Manager for ESDB persistent subscriptions
- Projection handlers for Interview and Sentence events
- Monitoring and health checks

</details>

<details>
<summary>M2.7: Testing & Validation ✅</summary>

- Integration tests for event-sourced processing
- E2E pipeline tests
- 72% code coverage

</details>

<details>
<summary>M2.8: Event-Sourced Architecture ✅</summary>

- Dynamic event versioning
- Edit protection across regeneration
- Cardinality enforcement at source
- Deprecation warnings for legacy paths
- **Completed:** 2026-01-17 (Production Ready)

</details>

---

## Dependency Upgrade Schedule

| Package | Current | Target | Milestone | Rationale |
|---------|---------|--------|-----------|-----------|
| neo4j | 5.28.1 | 6.x | **M3.0** | Vector types; single write path |
| openai | 1.93.3 | 2.x | **M3.2** | Agents SDK; function outputs |
| anthropic | >=0.39.0 | Latest | **M3.2** | Keep in sync |
| pytest | 8.3.3 | 9.x | M3.3 | Dev tooling |
| pytest-cov | 6.0.0 | 7.x | M3.3 | Dev tooling |
| redis | 6.2.0 | 7.x | M3.3 | Performance |
| isort | 5.13.2 | 7.x | M3.3 | Dev tooling |

---

## Technical Debt

### Post-M3.0 Cleanup
- [ ] Remove 27 legacy tests (test_neo4j_analysis_writer_legacy.py)
- [ ] Remove deprecated Neo4jMapStorage direct write code
- [ ] Remove deprecated Neo4jAnalysisWriter direct write code
- [ ] Update 11 data integrity tests for eventual consistency
- [ ] Rewrite 5 fault tolerance tests for EventStoreDB

### Future Improvements (Unprioritized)
- [ ] Prometheus metrics exporter (currently in-memory)
- [ ] WebSocket for real-time Neo4j updates
- [ ] CLI tool for replaying parked events
- [ ] Event schema versioning and migration
- [ ] OpenTelemetry distributed tracing
- [ ] Neo4j query optimization for bulk operations
- [ ] Circuit breaker for Neo4j connection failures
- [ ] Event archival/compaction strategy

---

## Architecture Overview

```
Current State (M2.8 - Dual-Write)
─────────────────────────────────
User Upload / Edit API
    ↓
Pipeline / Command Handlers
    ├──→ EventStoreDB (events) ← Source of Truth
    └──→ Neo4j (direct write)  ← Temporary (removed in M3.0)

EventStoreDB
    ↓
Projection Service
    ↓
Neo4j (materialized view)


Target State (M3.0 - Single-Writer)
────────────────────────────────────
User Upload / Edit API
    ↓
Pipeline / Command Handlers
    └──→ EventStoreDB (events only)

EventStoreDB
    ↓
Projection Service
    ↓
Neo4j (sole writer)
```

---

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-01-18 | Bundle neo4j 6.x with M3.0 | Single write path simplifies migration |
| 2026-01-18 | Separate openai 2.x to M3.2 | Orthogonal to event-sourcing; needs dedicated focus |
| 2026-01-18 | Defer pytest/redis to M3.3 | No immediate benefit; low priority |
| 2026-01-18 | Plan vector search for M3.1 | Requires neo4j 6.x vector types |

---

## How to Update This Document

1. **When starting a milestone:** Change status from 📋 to ⏳, update "Current Phase"
2. **When completing tasks:** Check off items in the milestone checklist
3. **When completing a milestone:** Change status to ✅, move to "Completed" section
4. **When making decisions:** Add entry to Decision Log

**Document Owner:** Engineering Team
**Review Cadence:** Update after each milestone completion
