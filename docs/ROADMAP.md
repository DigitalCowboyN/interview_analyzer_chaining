# Project Roadmap

> **This is the canonical project roadmap. Update this document when milestone status changes.**

---

## Quick Status

**Last Updated:** 2026-07-06

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
| **TC** | ✅ Complete | Test Coverage Improvement (90.1%) + Phase 9 Cleanup |
| **TC.10** | ✅ Complete | Test Fixes + Infrastructure Integration |
| **M4.1** | ✅ Complete | Layer 1: Ingestion, Map, Speaker Genesis & Stitching |
| **M4.2** | ✅ Complete | Layer 2: Extractor Registry, Provider Chain, Entities/Claims/Embeddings |
| **M3.1** | ✅ Complete | Vector Search (delivered as Layer 2 embeddings + per-model indexes) |
| M4.3 | 📋 Planned | Layer 3: Lens Engine (meeting_minutes first) |
| M4.4 | 📋 Planned | Layer 5: OKF Export + richer queries |
| M3.2 | 📋 Partial | AI Agent Upgrade (structured outputs landed; openai 2.x SDK bump still pending) |
| M3.3 | 📋 Planned | Infrastructure Upgrades |

**Current Phase:** M4.3 Planning (Layer 3 Lens Engine — see docs/superpowers/specs/2026-07-04-mine-layers-design.md)
**Tests:** 984 unit passing | **Coverage:** 88.8% (unit). ~200 legacy pipeline tests retired in M4.2.

---

## Milestone Checklist

### M4.2: Layer 2 — Extractor Registry & Core Enrichment ✅ COMPLETE

**Spec:** `docs/superpowers/specs/2026-07-04-mine-layers-design.md`
**Plan:** `docs/superpowers/plans/2026-07-05-layer2-extractor-registry.md`

- [x] Agent layer: API-level structured outputs (`call_model(prompt, schema=)` —
      OpenAI json_schema, Anthropic forced tool-use)
- [x] Provider strategy: `ClaudeCodeAgent` (headless `claude -p`) + `FailoverAgent`
      chain (Anthropic Haiku → Claude Code → OpenAI); Anthropic primary
- [x] Extractor registry (`config/extractors.yaml`) + ported prompts with numeric
      confidence; the 7 focused calls kept, never merged
- [x] GraphContextBuilder: speaker-labeled + utterance-aware contexts
- [x] spaCy cross-check flags (function/structure disagreement as review signal)
- [x] New extractors: entity mentions (span-grounded), claims (utterance-scoped)
- [x] Embeddings: `Embedder` protocol (OpenAI + local sentence-transformers,
      config-pinned), events with inline base64 vectors, per-model Neo4j vector indexes
- [x] AnalysisGenerated v2 (dimension confidences, flags, provider provenance)
- [x] Projections: Entity/Claim/embedding handlers + allowlists + drift guard
- [x] Enrichment orchestrator (resume-aware) + `python -m src.enrichment` CLI +
      ingest `--enrich` chaining
- [x] API `/analysis/` + Celery task rewired to ingest+enrich
- [x] Legacy pipeline retired (pipeline.py, sentence_analyzer, context_builder,
      analysis_service, pipeline_event_emitter, llm_responses; ~200 tests)
- [x] Golden/parity assertions; Layer 2 projection smoke test

**Completed:** 2026-07-06 (final-review fix wave 2026-07-09)

**Deferred to M4.3 entry (from the M4.2 final review — tracked debt):**
- Per-model embedding property (`embedding_<model>`) or single-index +
  mandatory `embedding_model` filter — today all per-model vector indexes
  target the shared `embedding` property (cross-contamination risk only when
  two models share dimensions).
- FailoverAgent chain construction is all-or-nothing (one missing API key
  kills the whole chain); skip unconstructible providers with a warning.
- Delete now-dead `src/io/local_storage.py` writers + `config.yaml`
  `classification` block ("registry is the only path" made literal).
- Per-dimension provider provenance (currently last-successful-call per unit);
  MENTIONS edge collapses duplicate mentions of one entity per fragment;
  embedder dim not validated against emitted vectors; EntitiesExtracted
  provider not materialized in graph; batch CLI aborts on first failing file.
- From the fix-wave re-review: strict-schema test recursion doesn't descend
  into inline `properties`/`items` (covered today via `$defs` hoisting);
  zero-claim utterances re-extract on every unforced resume (retry semantics,
  but unbounded); fully-failed fragments still consume embedding compute
  before being discarded.

---

### M4.1: Layer 1 — Ingestion, Map, Speaker Genesis & Stitching ✅ COMPLETE

**Spec:** `docs/superpowers/specs/2026-07-04-mine-layers-design.md`
**Plan:** `docs/superpowers/plans/2026-07-04-layer1-ingestion-map-speakers-stitching.md`

- [x] Offset-preserving segmentation (`segment_text_with_offsets`)
- [x] Ingestion package: format detector (labeled/flat) + normalizer producing
      offset-grounded fragments (`source[start:end] == fragment` invariant)
- [x] Sentence aggregate: `start_char`/`end_char` + correctable speaker
      attribution (`SpeakerAttributed`/`SpeakerReattributed`, human lock)
- [x] Interview aggregate: speaker lifecycle (`SpeakerCreated`/`Renamed`/`Merged`)
      and stitching overlay (`UtteranceIdentified`/`InterruptionRecorded`/`StitchRemoved`)
- [x] Windowed LLM speaker inference with deterministic overlap reconciliation
- [x] Stitcher: baseline grouping + LLM refinement (invalid proposals degrade
      to baseline; transcript text never rewritten)
- [x] Ingestion orchestrator + upgraded map (.jsonl with offsets, speaker,
      confidence, utterance per fragment); `python -m src.ingestion <file>`
- [x] Projection handlers: Speaker + Utterance nodes (`HAS_PARTICIPANT`,
      `SPOKEN_BY`, `SPOKE`, `PART_OF_UTTERANCE`, `INTERRUPTS`)
- [x] Correction API: rename/merge/split speakers, reattribute fragments,
      remove stitches (202 + version; human events lock fields)
- [x] Golden crosstalk fixture (deterministic, recorded LLM responses)

**Completed:** 2026-07-04

---

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
- [x] Phase 2: Pipeline Tests (21 tests)
  - Pipeline coverage: 52% → 81.4%
- [x] Phase 4: Parked Events Tests (20 tests)
  - Parked Events coverage: 29% → 100%
  - Fixed production bugs: invalid EventEnvelope.metadata reference
- [x] Phase 5: Critical Gaps (27 tests)
  - run_projection_service.py: 0% → 100%
  - sentence_analyzer.py: 65% → 92%
- [x] Phase 6: Domain Model (96 tests)
  - aggregates.py: 75% → 99%
  - environment.py: 57% → 95%
  - Fixed bug: detect_environment() f.read() called twice
- [x] Phase 7: Integration Test Infrastructure (10 tests)
  - Makefile Python auto-detection (pyenv/Homebrew/system)
  - ESDB environment-aware connection (host/docker/CI)
  - `make test-infra-up/down` and `make test-integration-full` targets
  - 75 integration tests passing, 44 skipped (architectural)
- [x] Fix M3.0 integration test compatibility
- [x] Update TEST_COVERAGE_IMPROVEMENT_PLAN.md
- [x] Phase 9: Architectural test cleanup
  - Deleted 11 deprecated tests (test_neo4j_analysis_writer_lifecycle.py)
  - Refactored 8 data integrity tests for projection pattern
  - Documented skipped tests with architectural rationale

**Results:**
- Tests: 554 → 1027 (+473 tests)
- Coverage: 66.8% → 90.1% (+23.3%)
- Skipped tests: Reduced from 44 architectural skips to 33 (11 deleted, 8 refactored)

**Completed:** 2026-01-25

---

### TC.10: Test Fixes + Infrastructure Integration ✅ COMPLETE

**Goal:** Fix remaining test issues and enable infrastructure-dependent integration tests

- [x] Fix projection handler unit tests (mock bug: Neo4jConnectionManager.get_session)
- [x] Fix pipeline conftest fixture (mock bug: classify_sentence vs analyze_sentence)
- [x] Add integration markers to live API tests (26 tests now properly marked)
- [x] Fix test_projection_rebuild.py infrastructure test
  - Made Neo4jMapStorage.initialize() a no-op for M3.0 single-writer architecture
  - Added `make test-rebuild` target
- [x] Fix InterviewCreatedData to include project_id in event data
  - Handlers now correctly read project_id from event payload
- [x] Fix test infrastructure EventStore connection (ESDB_CONNECTION_STRING override)
  - Tests now use EVENTSTORE_TEST_* variables following Neo4j pattern
- [x] Fix test_wait_for_ready_timeout flaky test
  - Now uses definitely-unavailable port instead of assuming production Neo4j absent

**Test Status:**
- Unit tests (`-m "not integration"`): 977 passed, 3 skipped
- Integration tests: 119 passed, 12 skipped (architectural)
- Full suite: All tests passing

**Completed:** 2026-01-28

---

### M3.1: Vector Search 📋 PLANNED

> Note: vector search builds on the Layer 1 fragment/utterance nodes (M4.1)
> and is expected to fold into Layer 2's embedding extractors (M4.2).

- [ ] Store sentence embeddings in Neo4j
- [ ] Semantic similarity search endpoints
- [ ] Vector-based clustering for topics
- [ ] Enhanced keyword/topic extraction
- [ ] Rewrite 11 fault tolerance tests for EventStoreDB (`test_neo4j_fault_tolerance.py`)
- [ ] Update 11 data integrity tests for eventual consistency (`test_neo4j_data_integrity.py`)

**Dependencies:** M3.0 complete (neo4j 6.x required)
**Skipped tests addressed:** 11 fault tolerance (rewrite for ESDB)

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
- [ ] Re-establish performance baselines for M3.0 single-writer architecture
- [ ] Unskip 7 performance benchmark tests (`test_neo4j_performance_benchmarks.py`)

**Dependencies:** M3.0 complete
**Skipped tests addressed:** 7 performance benchmarks (re-baseline for single-writer)

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
- Phase 2: Pipeline Tests (21 tests, 81.4% coverage)
- Phase 4: Parked Events Tests (20 tests, 100% coverage)
- Phase 5: Critical Gaps (27 tests) - run_projection_service 100%, sentence_analyzer 92%
- Phase 6: Domain Model (96 tests) - aggregates 99%, environment 95%
- Phase 7: Integration Test Infrastructure (10 tests) - ESDB environment detection, Makefile improvements
- Fixed M3.0 integration test compatibility
- Fixed production bugs in parked_events.py (invalid metadata reference)
- Fixed bug in environment.py (detect_environment f.read() called twice)
- Coverage: 66.8% → 90.1% (+23.3%)
- Tests: 554 → 1027 (+473 tests)
- **Completed:** 2026-01-25

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
- [ ] Rewrite 11 fault tolerance tests for EventStoreDB (M3.1)
- [ ] Re-baseline 7 performance benchmark tests for single-writer (M3.3)

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

## Skipped Tests Inventory

**Total skipped: 15** (3 unit + 12 integration)

### Unit (3 skipped)

| Test | Reason | Milestone |
|------|--------|-----------|
| `test_helpers.py` (2 tests) | `openpyxl` not installed | N/A — optional dependency |
| `test_text_processing.py` (1 test) | Import-time exception logging untestable without reload | N/A — test limitation |

### Integration (12 skipped)

| Test File | Tests | Reason | Milestone |
|-----------|-------|--------|-----------|
| `test_neo4j_fault_tolerance.py` | 11 | M2.8: Neo4j fault tolerance irrelevant; ESDB is source of truth | **M3.1** |
| `test_neo4j_performance_benchmarks.py` | 1 | `psutil` not installed for memory benchmark | N/A — optional dependency |

**Note:** `test_neo4j_performance_benchmarks.py` has a module-level skip covering all 7 tests, but pytest only counts it as 1 skip in the summary. The 7 tests are tracked under M3.3 for re-baselining.

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
| 2026-07-06 | M4.2 (Layer 2) complete; legacy pipeline retired | Registry is the sole enrichment path; ~200 legacy tests removed |
| 2026-07-06 | Generalized provider strategy (interface + config chain) | Anthropic Haiku primary → Claude Code harness → OpenAI; embeddings config-pinned, model-tagged, never silently switched |
| 2026-07-06 | Embeddings ride as events with inline base64 vectors | Preserves single-writer + replay purity; direct Neo4j writes rejected |
| 2026-07-06 | Bumped anthropic model claude-3-haiku-20240307 → claude-haiku-4-5-20251001 | Old model retired (not_found_error); verified live |
| 2026-07-06 | Vector-index DDL runs in its own auto-commit session | Neo4j forbids schema DDL inside a data-write transaction |
| 2026-07-04 | M4.1 (Layer 1) complete: speakers, utterances, offset-grounded map | Spec: docs/superpowers/specs/2026-07-04-mine-layers-design.md |
| 2026-07-04 | Stitching is an overlay, never a rewrite | Interview must be viewable as-spoken; interpretation is additive + correctable |
| 2026-07-04 | Speaker inference reconciles windows by deterministic overlap voting | LLM-based reconciliation deferred until golden evaluation demands it |
| 2026-01-31 | Unskipped 4 integration tests | Fixed Neo4j connection (NEO4J_URI override), .env loading, M3.0 test update |
| 2026-01-31 | Fixed performance test Neo4j connection | Override NEO4J_URI in setup_test_environment so handlers use test DB |
| 2026-01-31 | Added .env loading to root conftest | API keys from .env now available for all tests without manual export |
| 2026-01-31 | Updated test_concurrent_file_processing for M3.0 | Added projection handler replay before Neo4j verification |
| 2026-01-28 | TC.10 complete | All infrastructure tests passing, event env vars fixed |
| 2026-01-28 | Added project_id to InterviewCreatedData | Handler needs project_id in event data, not just envelope |
| 2026-01-28 | Made Neo4jMapStorage.initialize() no-op | M3.0 single-writer: projection service is sole Neo4j writer |
| 2026-01-28 | Use EVENTSTORE_TEST_* vars in test fixtures | Follow Neo4j pattern; ignore production .env values |
| 2026-01-26 | TC.10 phase for infrastructure tests | Projection rebuild test needs dedicated Make target and infra |
| 2026-01-26 | Fixed mock bugs in unit tests | Neo4j session mock used wrong pattern; pipeline fixture mocked wrong method |
| 2026-01-26 | Added integration markers to 26 live API tests | Proper skipping when running unit tests without API keys |
| 2026-01-25 | Integration test infrastructure complete | ESDB environment detection, Make targets for test orchestration |
| 2026-01-24 | Test coverage 90.1% achieved | 1000 tests, added Phase 5-6, fixed environment.py bug |
| 2026-01-24 | Test coverage milestone extended | 86.0% coverage achieved, 943 tests, fixed parked_events.py bugs |
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
