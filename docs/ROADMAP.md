# Project Roadmap

**Last Updated:** 2026-01-18
**Current Status:** M2.8 Complete (Production Ready)

---

## Completed Milestones

### M1: Core Plumbing ✅
- Event envelope and domain events
- EventStoreDB client and connection management
- Repository pattern for aggregates
- Interview and Sentence aggregates

### M2.1: Command Layer ✅
- Command base classes and handlers
- Interview and Sentence commands
- Actor tracking and correlation IDs

### M2.2: Dual-Write Integration ✅
- Event-first dual-write pattern
- Pipeline emits events before Neo4j writes
- Event failures abort operations (correct behavior)

### M2.3-M2.5: Projection Infrastructure ✅
- Lane Manager with 12 configurable lanes
- Subscription Manager for ESDB persistent subscriptions
- Projection handlers for Interview and Sentence events
- Monitoring and health checks

### M2.7: Testing & Validation ✅
- Integration tests for event-sourced processing
- E2E pipeline tests
- 72% code coverage

### M2.8: Event-Sourced Architecture ✅
- Dynamic event versioning
- Edit protection across regeneration
- Cardinality enforcement at source
- Deprecation warnings for legacy paths
- **Status:** Production Ready

---

## Current Phase: M2.9

### M2.9: User Edit API
**Status:** Not Started
**Priority:** High

**Scope:**
1. Create `src/api/routers/edits.py` with edit endpoints
2. `PUT /interviews/{id}/sentences/{id}` for sentence edits
3. `PUT /interviews/{id}/sentences/{id}/analysis` for analysis overrides
4. Integration with command handlers
5. Return accepted status with version

**Dependencies:** None (can start now)

---

## Future Milestones

### M3.0: Remove Dual-Write (Single-Writer Architecture)
**Status:** Planned
**Priority:** High
**Prerequisite:** M2.9 complete, 1-2 weeks production validation

**Scope:**
1. Remove direct Neo4j writes from pipeline
2. Projection service becomes SOLE writer to Neo4j
3. Remove deprecated code paths
4. Remove legacy tests (27 in test_neo4j_analysis_writer_legacy.py)
5. Update documentation

**Dependency Upgrades (Bundle with M3.0):**

| Package | Current | Target | Rationale |
|---------|---------|--------|-----------|
| **neo4j** | 5.28.1 | 6.x | Single write path simplifies migration; prepares for vector search |

**Why bundle neo4j upgrade with M3.0:**
- M3.0 removes dual-write, leaving only projection handlers writing to Neo4j
- Single code path to update (vs two paths now)
- neo4j 6.x has breaking changes in Bookmark API and error handling
- Cleaner to upgrade when architecture is simplified

---

### M3.1: Vector Search & Semantic Features
**Status:** Planned
**Priority:** Medium
**Prerequisite:** M3.0 complete, neo4j 6.x upgraded

**Scope:**
1. Store sentence embeddings in Neo4j (vector types from neo4j 6.x)
2. Semantic similarity search endpoints
3. Vector-based clustering for topics
4. Enhanced keyword/topic extraction using embeddings

**Why this depends on neo4j 6.x:**
- Bolt 6.0 protocol supports native vector types
- Efficient vector storage and indexing
- Vector similarity functions in Cypher

---

### M3.2: AI Agent Framework Upgrade
**Status:** Planned
**Priority:** Medium
**Prerequisite:** M3.0 complete

**Scope:**
1. Upgrade openai SDK from 1.x to 2.x
2. Refactor agent implementations in `src/agents/`
3. Evaluate OpenAI Agents SDK for structured workflows
4. Update Anthropic SDK if needed

**Dependency Upgrades (Bundle with M3.2):**

| Package | Current | Target | Rationale |
|---------|---------|--------|-----------|
| **openai** | 1.93.3 | 2.x | Required for Agents SDK; improved function call outputs |
| **anthropic** | >=0.39.0 | Latest | Keep in sync with openai |

**Why separate milestone:**
- Agent refactoring is orthogonal to event-sourcing
- openai 2.x is a major rewrite requiring careful migration
- Can be done in parallel with M3.1 if resources allow

---

### M3.3: Infrastructure & Tooling Upgrades
**Status:** Planned
**Priority:** Low
**Prerequisite:** M3.0 complete

**Scope:**
1. Upgrade test infrastructure
2. Upgrade dev tooling
3. Performance baseline updates

**Dependency Upgrades:**

| Package | Current | Target | Rationale |
|---------|---------|--------|-----------|
| **pytest** | 8.3.3 | 9.x | Native TOML config, strict mode (drops Python 3.9) |
| **pytest-cov** | 6.0.0 | 7.x | Keep in sync with pytest |
| **redis** | 6.2.0 | 7.x | Performance improvements for Celery |
| **isort** | 5.13.2 | 7.x | Updated defaults (dev tooling) |

**Why low priority:**
- Dev tooling, not production functionality
- No features needed for roadmap
- Can be done as housekeeping anytime after M3.0

---

## Dependency Upgrade Summary

| Package | Current | Target | Milestone | Rationale |
|---------|---------|--------|-----------|-----------|
| neo4j | 5.28.1 | 6.x | **M3.0** | Vector types for M3.1; single write path |
| openai | 1.93.3 | 2.x | **M3.2** | Agents SDK; function outputs |
| anthropic | >=0.39.0 | Latest | **M3.2** | Keep in sync |
| pytest | 8.3.3 | 9.x | M3.3 | Dev tooling |
| pytest-cov | 6.0.0 | 7.x | M3.3 | Dev tooling |
| redis | 6.2.0 | 7.x | M3.3 | Performance |
| isort | 5.13.2 | 7.x | M3.3 | Dev tooling |

---

## Technical Debt & Improvements

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

## Risk Register

| Risk | Milestone | Mitigation |
|------|-----------|------------|
| neo4j 6.x breaking changes | M3.0 | Upgrade after dual-write removed (single path) |
| openai 2.x agent refactor | M3.2 | Dedicated milestone with thorough testing |
| Vector search performance | M3.1 | Benchmark before production rollout |
| Eventual consistency edge cases | M3.0+ | Add specific tests after architecture stabilizes |

---

## Timeline Guidance

**Note:** No time estimates per project policy. Sequence only.

```
Current State (M2.8 Complete)
    │
    ├── M2.9: User Edit API
    │
    ├── M3.0: Remove Dual-Write + neo4j 6.x
    │       │
    │       ├── M3.1: Vector Search (depends on neo4j 6.x)
    │       │
    │       └── M3.2: AI Agent Upgrade (parallel possible)
    │
    └── M3.3: Infrastructure Upgrades (anytime after M3.0)
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

**Document Owner:** Engineering Team
**Review Cadence:** Update after each milestone completion
