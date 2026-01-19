# Project Roadmap

> **This is the canonical project roadmap. Update this document when milestone status changes.**

---

## Quick Status

**Last Updated:** 2026-01-19

| Milestone | Status | Description |
|-----------|--------|-------------|
| M1 | ✅ Complete | Core Plumbing (events, ESDB, aggregates) |
| M2.1 | ✅ Complete | Command Layer |
| M2.2 | ✅ Complete | Dual-Write Integration |
| M2.3-M2.5 | ✅ Complete | Projection Infrastructure |
| M2.7 | ✅ Complete | Testing & Validation |
| M2.8 | ✅ Complete | Event-Sourced Architecture (Production Ready) |
| M2.9 | ✅ Complete | User Edit API |
| M3.0 | ✅ Complete | Remove Dual-Write + neo4j 6.x |
| **TC** | ✅ **Complete** | Test Coverage Improvement (84.2%) |
| M3.1 | 📋 Planned | Vector Search |
| M3.2 | 📋 Planned | AI Agent Upgrade (openai 2.x) |
| M3.3 | 📋 Planned | Infrastructure Upgrades |

**Current Phase:** Ready for M3.1 (Vector Search)
**Tests:** 902 passing, 54 skipped | **Coverage:** 84.2%

---

## Milestone Checklist

### M2.9: User Edit API ✅ COMPLETE

- [x] Review existing `src/api/routers/edits.py` implementation
- [x] Complete `POST /edits/sentences/{interview_id}/{sentence_index}/edit` endpoint
- [x] Complete `POST /edits/sentences/{interview_id}/{sentence_index}/analysis/override` endpoint
- [x] Complete `GET /edits/sentences/{interview_id}/{sentence_index}/history` endpoint
- [x] Integration with command handlers (SentenceCommandHandler.handle)
- [x] Return 202 Accepted status with version
- [x] E2E tests passing (3 tests)
- [x] Unit tests passing (16 tests)

**Completed:** 2026-01-18

---

### M3.0: Remove Dual-Write ✅ COMPLETE

- [x] Remove direct Neo4j writes from pipeline
- [x] Projection service becomes SOLE writer
- [x] Remove deprecated code paths
- [x] Remove 27+ legacy tests (41 tests deleted)
- [x] Upgrade neo4j 5.28.1 → 6.x (driver) / 5.26.0 (server)
- [x] Update documentation

**Completed:** 2026-01-18

---

### TC: Test Coverage Improvement ✅ COMPLETE

- [x] Phase 0: Projection Infrastructure (150 tests)
  - Handler Registry, Bootstrap, Metrics, Health
  - Subscription Manager, Projection Service
- [x] Phase 1: Event Sourcing Foundation (81 tests)
  - Event Store (92.5%), Repository (94.6%), Command Handlers (89.4%)
- [x] Fix M3.0 integration test compatibility
- [x] Update TEST_COVERAGE_IMPROVEMENT_PLAN.md

**Results:**
- Tests: 554 → 902 (+348 tests)
- Coverage: 66.8% → 84.2% (+17.4%)

**Completed:** 2026-01-19

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

<details>
<summary>M2.9: User Edit API ✅</summary>

- Edit sentence endpoint: `POST /edits/sentences/{id}/{index}/edit`
- Override analysis endpoint: `POST /edits/sentences/{id}/{index}/analysis/override`
- History endpoint: `GET /edits/sentences/{id}/{index}/history`
- Returns 202 Accepted with version
- 16 unit tests + 3 E2E tests
- **Completed:** 2026-01-18

</details>

<details>
<summary>TC: Test Coverage Improvement ✅</summary>

- Phase 0: Projection Infrastructure (150 tests, 78-100% coverage)
- Phase 1: Event Sourcing Foundation (81 tests, 89-95% coverage)
- Fixed M3.0 integration test compatibility
- Coverage: 66.8% → 84.2% (+17.4%)
- Tests: 554 → 902 (+348 tests)
- **Completed:** 2026-01-19

</details>

---

## Dependency Upgrade Schedule

| Package | Current | Target | Milestone | Rationale |
|---------|---------|--------|-----------|-----------|
| neo4j | ~~5.28.1~~ 6.x | ✅ Done | **M3.0** | Vector types; single write path |
| openai | 1.93.3 | 2.x | **M3.2** | Agents SDK; function outputs |
| anthropic | >=0.39.0 | Latest | **M3.2** | Keep in sync |
| pytest | 8.3.3 | 9.x | M3.3 | Dev tooling |
| pytest-cov | 6.0.0 | 7.x | M3.3 | Dev tooling |
| redis | 6.2.0 | 7.x | M3.3 | Performance |
| isort | 5.13.2 | 7.x | M3.3 | Dev tooling |

---

## Technical Debt

### Post-M3.0 Cleanup ✅ DONE
- [x] Remove 27 legacy tests (test_neo4j_analysis_writer_legacy.py)
- [x] Remove deprecated Neo4jMapStorage direct write code
- [x] Remove deprecated Neo4jAnalysisWriter direct write code
- [x] Remove graph_persistence tests (14 tests)
- [ ] Update 11 data integrity tests for eventual consistency (M3.1)
- [ ] Rewrite 5 fault tolerance tests for EventStoreDB (M3.1)

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
Current State (M3.0 - Single-Writer) ✅
───────────────────────────────────────
User Upload / Edit API
    ↓
Pipeline / Command Handlers
    └──→ EventStoreDB (events only) ← Source of Truth

EventStoreDB
    ↓
Projection Service (12 lanes)
    ↓
Neo4j (sole writer, materialized view)
```

---

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-01-19 | Test coverage milestone complete | 84.2% coverage achieved (target 80%), 902 tests |
| 2026-01-18 | M3.0 complete: single-writer architecture | Removed 41 legacy tests, all direct Neo4j writes eliminated |
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
