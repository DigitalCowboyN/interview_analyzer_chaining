# Test Coverage Improvement Plan

> **Created:** 2026-01-18
> **Updated:** 2026-01-24
> **Current Coverage:** 90.1% (1027 tests passing, 45 skipped) ✅ STRETCH GOAL MET
> **Target Coverage:** 90%+ (stretch goal)

---

## Testing Principles

> **IMPORTANT: Tests must validate production code correctness, not just achieve coverage.**

### Core Guidelines

1. **Test Real Behavior** - Tests verify that production code works correctly. Never modify production code just to make tests pass. If code must change, understand and verify the impact across all affected areas.

2. **Test-Driven Mindset** - Each test should answer: "What behavior am I verifying?" If you can't articulate the behavior, don't write the test.

3. **Arrange-Act-Assert** - Structure tests clearly:
   - **Arrange**: Set up preconditions and inputs
   - **Act**: Execute the code under test
   - **Assert**: Verify expected outcomes

4. **One Behavior Per Test** - Each test should verify one specific behavior. Multiple assertions are fine if they verify aspects of the same behavior.

5. **Descriptive Names** - Test names should describe the scenario and expected outcome:
   - ✅ `test_save_aggregate_raises_concurrency_error_when_version_mismatch`
   - ❌ `test_save_aggregate_error`

6. **Independence** - Tests must not depend on execution order or shared mutable state. Each test sets up its own preconditions.

7. **No Test Logic** - Tests should not contain conditionals, loops, or complex logic. If setup is complex, use fixtures.

8. **Mock at Boundaries** - Mock external dependencies (databases, APIs, file systems), not internal classes. Test real integration where practical.

9. **Impact Analysis** - Before changing production code to fix a test:
   - Identify all callers of the changed code
   - Verify existing tests still pass
   - Consider if the "fix" masks a real bug

10. **Prefer Integration Over Mocking** - When feasible, test against real dependencies (using test containers or in-memory implementations) rather than extensive mocking.

---

## Executive Summary

This plan prioritizes test coverage improvements based on:
1. **Criticality** - Impact on system functionality if broken
2. **Risk** - Likelihood of bugs/regressions
3. **Coverage Gap** - Current % vs target

### Coverage by Component

| Component | Current | Target | Gap | Priority |
|-----------|---------|--------|-----|----------|
| Projection Service | 78-100% | 80% | ✅ DONE | P0 |
| Event Store/Repository | 94-95% | 85% | ✅ DONE | P1 |
| Command Handlers | 95.5% | 85% | ✅ DONE | P1 |
| Pipeline | 83.5% | 75% | ✅ DONE | P2 |
| API Routers | 73-96% | 90% | MEDIUM | P3 |
| Parked Events | 100% | 80% | ✅ DONE | P4 |
| run_projection_service.py | 100% | 80% | ✅ DONE | P5 |
| sentence_analyzer.py | 92.1% | 85% | ✅ DONE | P5 |
| aggregates.py | 98.9% | 85% | ✅ DONE | P6 |
| environment.py | 94.8% | 75% | ✅ DONE | P6 |
| Skipped Integration Tests | 60 skipped | 0 skipped | MEDIUM | P7 |
| E2E Tests | 6 tests | 15+ tests | MEDIUM | P8 |

---

## Phase 0: Critical Infrastructure ✅ COMPLETE

**Goal:** Cover the projection service that has ZERO test coverage.
**Status:** All 6 modules complete with 150 tests total.

### P0.1: Handler Registry (`src/projections/handlers/registry.py`) ✅
**Effort:** Low | **Impact:** High | **Tests:** 27 | **Coverage:** 100%

```
Unit Tests Needed:
- test_register_handler_success
- test_register_handler_duplicate_warning
- test_get_handler_found
- test_get_handler_not_found_returns_none
- test_get_registered_types
- test_global_registry_singleton
```

### P0.2: Bootstrap (`src/projections/bootstrap.py`) ✅
**Effort:** Medium | **Impact:** High | **Tests:** 17 | **Coverage:** 100%

### P0.3: Metrics (`src/projections/metrics.py`) ✅
**Effort:** Medium | **Impact:** Medium | **Tests:** 17 | **Coverage:** 98.5%

### P0.4: Health (`src/projections/health.py`) ✅
**Effort:** Low | **Impact:** Medium | **Tests:** 33 | **Coverage:** 100%

### P0.5: Subscription Manager (`src/projections/subscription_manager.py`) ✅
**Effort:** High | **Impact:** Critical | **Tests:** 30 | **Coverage:** 78.8%

### P0.6: Projection Service (`src/projections/projection_service.py`) ✅
**Effort:** High | **Impact:** Critical | **Tests:** 26 | **Coverage:** 85.7%

### P0.7: Run Projection Service (`src/run_projection_service.py`)
**Effort:** Medium | **Impact:** High | **Status:** Deferred (CLI entry point, lower priority)

---

## Phase 1: Event Sourcing Foundation ✅ COMPLETE

**Goal:** Strengthen the event sourcing core to 85%+
**Status:** All 3 modules complete with 81 additional tests.

### P1.1: Event Store (`src/events/store.py`) ✅
**Effort:** Medium | **Impact:** High | **Tests:** 34 | **Coverage:** 92.5%

### P1.2: Repository (`src/events/repository.py`) ✅
**Effort:** Medium | **Impact:** High | **Tests:** 27 | **Coverage:** 94.6%

### P1.3: Command Handlers (`src/commands/handlers.py`) ✅
**Effort:** Medium | **Impact:** High | **Tests:** 20 | **Coverage:** 89.4%

---

## Phase 2: Pipeline Coverage ✅ COMPLETE

**Goal:** Increase pipeline coverage to 75%+
**Status:** Pipeline coverage improved from 52% to 81.4% with 21 tests.

### P2.1: Pipeline Orchestrator (`src/pipeline.py`) ✅
**Effort:** Medium | **Impact:** High | **Tests:** 21 | **Coverage:** 81.4%

```
Unit Tests Needed:
# Initialization
- test_pipeline_init_with_config
- test_pipeline_init_with_event_emitter
- test_pipeline_init_without_event_emitter

# File Processing
- test_discover_input_files
- test_discover_input_files_empty_directory
- test_filter_already_processed_files

# Sentence Processing
- test_segment_sentences_success
- test_segment_sentences_empty_file
- test_create_sentence_map

# Analysis
- test_build_context_windows
- test_analyze_sentence_success
- test_analyze_sentence_llm_error
- test_consolidate_analysis_results

# Persistence
- test_save_results_to_jsonl
- test_emit_events_for_results
- test_handle_persistence_error

# Verification
- test_verify_output_files_exist
- test_verify_output_counts_match

Integration Tests Needed:
- test_full_file_processing_with_mock_llm
- test_pipeline_event_emission_flow
- test_pipeline_error_recovery
```

---

## Phase 3: API & Services (74-94%)

**Goal:** Achieve 90%+ on API layer

### P3.1: Files API (`src/api/routers/files.py`)
**Current:** 74% | **Target:** 90%

```
Unit Tests Needed:
- test_upload_file_success
- test_upload_file_invalid_format
- test_upload_file_too_large
- test_get_files_list
- test_get_file_by_id
- test_get_file_not_found
- test_delete_file_success
- test_delete_file_not_found
```

### P3.2: Analysis API (`src/api/routers/analysis.py`)
**Current:** 88% | **Target:** 95%

```
Unit Tests Needed:
- test_get_analysis_by_interview
- test_get_analysis_by_sentence
- test_analysis_not_found
```

---

## Phase 4: Resilience & Operations ✅ COMPLETE

### P4.1: Parked Events (`src/projections/parked_events.py`) ✅
**Effort:** Medium | **Impact:** High | **Tests:** 20 | **Coverage:** 100%

**Note:** Tests uncovered and fixed production bugs:
- `EventEnvelope.metadata` attribute did not exist - fixed to use `actor`, `correlation_id`, `source`
- Invalid `event_id` format (`parked-{uuid}`) - fixed to generate proper UUID

---

## Phase 5: Critical Gaps ✅ COMPLETE

**Goal:** Address zero/low coverage in critical production code
**Priority:** P1 - These gaps pose real risk
**Status:** Complete (27 tests added)

### P5.1: Run Projection Service (`src/run_projection_service.py`) ✅
**Coverage:** 0% → 100% | **Tests:** 15 | **Effort:** Medium | **Impact:** High

CLI entry point for the projection service. Currently completely untested.

```
Unit Tests Needed:
# Argument Parsing
- test_parse_args_defaults
- test_parse_args_custom_lanes
- test_parse_args_verbose_flag

# Service Initialization
- test_create_projection_service
- test_signal_handler_registration
- test_graceful_shutdown_on_sigterm
- test_graceful_shutdown_on_sigint

# Main Loop
- test_main_runs_service
- test_main_handles_keyboard_interrupt
- test_main_logs_startup_info
```

### P5.2: Sentence Analyzer (`src/agents/sentence_analyzer.py`) ✅
**Coverage:** 65.3% → 92.1% | **Tests:** 12 added | **Effort:** Medium | **Impact:** Critical

Core analysis logic with significant gaps in error handling paths. Added tests for config errors and validation error handling for all 7 response types.

```
Unit Tests Needed:
# LLM Response Handling
- test_parse_llm_response_valid_json
- test_parse_llm_response_invalid_json
- test_parse_llm_response_missing_fields
- test_parse_llm_response_extra_fields

# Retry Logic
- test_analyze_retries_on_rate_limit
- test_analyze_retries_on_timeout
- test_analyze_fails_after_max_retries

# Edge Cases
- test_analyze_empty_sentence
- test_analyze_very_long_sentence
- test_analyze_special_characters
- test_analyze_non_english_text
```

---

## Phase 6: Domain Model Coverage ✅ COMPLETE

**Goal:** Strengthen domain logic and utility coverage
**Priority:** P2 - Important for correctness
**Status:** Complete (96 tests added)

### P6.1: Aggregates (`src/events/aggregates.py`) ✅
**Coverage:** 74.7% → 98.9% | **Tests:** 31 | **Effort:** Medium | **Impact:** High

Domain aggregates with event application logic. Full coverage for Interview and Sentence aggregates, event application, state transitions, and edit protection.

```
Unit Tests Needed:
# Interview Aggregate
- test_interview_apply_status_changed
- test_interview_apply_metadata_updated
- test_interview_invalid_state_transition

# Sentence Aggregate
- test_sentence_apply_text_edited
- test_sentence_apply_analysis_overridden
- test_sentence_version_increment
- test_sentence_edit_protection

# Edge Cases
- test_aggregate_apply_unknown_event_type
- test_aggregate_replay_from_events
- test_aggregate_concurrent_modification
```

### P6.2: Environment Utils (`src/utils/environment.py`) ✅
**Coverage:** 56.5% → 94.8% | **Tests:** 65 | **Effort:** Low | **Impact:** Low

Fixed bug: detect_environment() called f.read() twice (second call returned empty, breaking containerd detection).

Environment detection and path resolution.

```
Unit Tests Needed:
# Platform Detection
- test_detect_docker_environment
- test_detect_ci_environment
- test_detect_development_environment

# Path Resolution
- test_resolve_input_path_absolute
- test_resolve_input_path_relative
- test_resolve_output_path_with_env_var

# Configuration
- test_get_neo4j_config_from_env
- test_get_esdb_config_from_env
- test_missing_required_env_var
```

### P6.3: Pipeline Error Paths (`src/pipeline.py`)
**Current:** 83.5% | **Target:** 90% | **Effort:** Medium | **Impact:** Medium

84 lines still uncovered, mostly error handling.

```
Unit Tests Needed:
# Error Recovery
- test_file_processing_io_error_recovery
- test_analysis_timeout_handling
- test_partial_batch_failure_continues

# Concurrent Processing
- test_semaphore_limits_concurrency
- test_task_cancellation_cleanup

# Event Emission Failures
- test_event_emission_retry
- test_event_emission_failure_aborts
```

---

## Phase 7: Integration Test Infrastructure ✅ COMPLETE

**Goal:** Enable integration tests to run via Make targets with proper environment detection
**Status:** Complete (2026-01-25)

### P7.1: Makefile Python Auto-Detection ✅
**File:** `Makefile`

Fixed Python detection to work with pyenv, Homebrew, and system Python:
```makefile
PYTHON ?= $(shell command -v python 2>/dev/null || command -v python3 2>/dev/null)
```

### P7.2: ESDB Environment-Aware Connection ✅
**Files:** `src/events/store.py`, `src/api/routers/edits.py`

Added automatic environment detection for EventStoreDB connection:
- Host environment → `localhost:2113`
- Docker/CI environment → `eventstore:2113`

Priority order: 1) config file, 2) `ESDB_CONNECTION_STRING` env var, 3) auto-detect

### P7.3: Docker Compose Explicit Overrides ✅
**File:** `docker-compose.yml`

Added explicit `ESDB_CONNECTION_STRING` to app, worker, and projection services.

### P7.4: Environment Detection Tests ✅
**Files:** `tests/events/test_store_unit.py`, `tests/api/test_edit_api_unit.py`

Added 10 new tests for environment detection:
- Host environment uses localhost
- Docker environment uses eventstore service
- CI environment uses eventstore service
- Environment variable overrides detection
- Config file takes highest precedence

### P7.5: Command Handler Test Fixes ✅
**File:** `tests/commands/test_command_handlers.py`

Fixed tests to use mocked EventStoreClient instead of connecting to real ESDB.

### Integration Test Results
```
make test-infra-up    → Starts neo4j-test + eventstore with health checks
Integration tests     → 75 passed, 44 skipped (architectural)
make test-infra-down  → Stops infrastructure
```

### Skipped Tests (Architectural)

See Phase 9 for the plan to address these 44 skipped tests.

---

## Phase 8: E2E Test Expansion 📋 PLANNED

**Goal:** Comprehensive end-to-end coverage of user workflows
**Priority:** P2 - Validates full system integration
**Current:** 6 E2E tests | **Target:** 15+ E2E tests

### P8.1: Complete Interview Workflow
```
test_e2e_complete_interview_workflow:
  1. Upload file via API
  2. Wait for pipeline processing
  3. Verify events in EventStoreDB
  4. Wait for projection to complete
  5. Query analysis via API
  6. Verify Neo4j state matches
```

### P8.2: User Edit Workflow
```
test_e2e_user_edit_full_flow:
  1. Process interview
  2. Edit sentence text via API
  3. Verify SentenceTextEdited event
  4. Wait for projection
  5. Query sentence via API - verify updated text
  6. Query Neo4j - verify updated text
```

### P8.3: Analysis Override Workflow
```
test_e2e_analysis_override_flow:
  1. Process interview
  2. Override analysis via API
  3. Verify AnalysisOverridden event
  4. Regenerate analysis
  5. Verify edit protection preserved
```

### P8.4: Projection Service Recovery
```
test_e2e_projection_recovery:
  1. Process interview
  2. Stop projection service
  3. Process another interview (events queue)
  4. Restart projection service
  5. Verify catch-up processes queued events
  6. Verify Neo4j state correct
```

### P8.5: Concurrent Processing
```
test_e2e_concurrent_file_processing:
  1. Upload 5 files simultaneously
  2. Verify all processed correctly
  3. Verify no race conditions in Neo4j
  4. Verify event ordering preserved per aggregate
```

### P8.6: Error Scenarios
```
test_e2e_llm_failure_handling:
  1. Upload file
  2. Mock LLM to fail
  3. Verify retry behavior
  4. Verify parked events after max retries

test_e2e_neo4j_unavailable:
  1. Process interview
  2. Make Neo4j unavailable
  3. Verify events still stored
  4. Restore Neo4j
  5. Verify projection catches up
```

---

## Implementation Priority Matrix

| Phase | Description | Est. Tests | Coverage Impact | Effort | Status |
|-------|-------------|------------|-----------------|--------|--------|
| P0 | Projection Infrastructure | 150 | +10% | High | ✅ DONE |
| P1 | Event Sourcing Foundation | 81 | +8% | High | ✅ DONE |
| P2 | Pipeline | 21 | +6% | Medium | ✅ DONE |
| P3 | API Layer | ~15 | +2% | Low | 📋 Planned |
| P4 | Resilience (Parked Events) | 20 | +2% | Medium | ✅ DONE |
| **P5** | **Critical Gaps** | **~20** | **+2%** | **Medium** | **📋 Planned** |
| **P6** | **Domain Model** | **~25** | **+2%** | **Medium** | **📋 Planned** |
| **P7** | **Integration Modernization** | **~40** | **+1%** | **High** | **📋 Planned** |
| **P8** | **E2E Expansion** | **~10** | **+1%** | **High** | **📋 Planned** |
| | | | | | |
| **Completed** | | **272** | **+19.2%** | | ✅ |
| **Remaining** | | **~110** | **+4%** | | 📋 |
| **Total** | | **~382** | **86% → 90%** | | |

---

## Test File Structure

```
tests/
├── projections/
│   ├── test_registry_unit.py              # ✅ DONE
│   ├── test_bootstrap_unit.py             # ✅ DONE
│   ├── test_metrics_unit.py               # ✅ DONE
│   ├── test_health_unit.py                # ✅ DONE
│   ├── test_subscription_manager_unit.py  # ✅ DONE
│   ├── test_projection_service_unit.py    # ✅ DONE
│   ├── test_parked_events_unit.py         # ✅ DONE
│   ├── test_lane_manager_unit.py          # EXISTS
│   ├── test_projection_handlers_unit.py   # EXISTS
│   └── test_run_projection_service_unit.py # 📋 P5.1
├── events/
│   ├── test_store_unit.py                 # ✅ DONE
│   ├── test_repository_unit.py            # ✅ DONE
│   ├── test_aggregates_unit.py            # 📋 P6.1
│   └── test_core_plumbing_validation.py   # EXISTS
├── commands/
│   ├── test_command_handlers_unit.py      # ✅ EXPANDED
│   └── test_commands_unit.py              # EXISTS
├── agents/
│   ├── test_sentence_analyzer_unit.py     # 📋 P5.2 (expand)
│   └── ... (existing)
├── pipeline/
│   ├── test_pipeline_execution_unit.py    # ✅ DONE
│   ├── test_pipeline_error_paths_unit.py  # 📋 P6.3
│   └── test_pipeline_orchestrator_unit.py # EXISTS
├── utils/
│   ├── test_environment_unit.py           # 📋 P6.2
│   └── ... (existing)
├── integration/
│   ├── test_neo4j_data_integrity.py       # 📋 P7.1 (update)
│   ├── test_neo4j_fault_tolerance.py      # 📋 P7.2 (update)
│   ├── test_neo4j_performance_benchmarks.py # 📋 P7.3 (update)
│   ├── test_projection_rebuild.py         # 📋 P7.4 (enable)
│   ├── test_projection_dual_write_validation.py # 📋 P7.4 (enable)
│   └── ... (existing)
└── e2e/
    ├── test_e2e_file_processing.py        # EXISTS (expand)
    ├── test_e2e_user_edits.py             # EXISTS (expand)
    ├── test_e2e_projection_recovery.py    # 📋 P8.4
    └── test_e2e_error_scenarios.py        # 📋 P8.6
```

---

## Execution Plan

### ✅ Completed (Phases 0-7)
- Phase 0: Projection Infrastructure (150 tests)
- Phase 1: Event Sourcing Foundation (81 tests)
- Phase 2: Pipeline Coverage (21 tests)
- Phase 4: Parked Events (20 tests)
- Phase 5: Critical Gaps (27 tests) - run_projection_service 100%, sentence_analyzer 92%
- Phase 6: Domain Model (96 tests) - aggregates 99%, environment 95%
- Phase 7: Integration Modernization - removed 5 obsolete dual-write tests, enabled 2 projection rebuild tests, fixed 8 SpaCy tests

### Next: Phase 8 - E2E Expansion
- P8.1-P8.6: Add ~10 new E2E scenarios

---

## Success Metrics

| Metric | Start | Current | Target |
|--------|-------|---------|--------|
| Overall Coverage | 66.8% | **90.1%** ✅ | 90%+ |
| Tests Passing | 554 | **1027** ✅ | 1050+ |
| Tests Skipped | 54 | **45** | <10 |
| E2E Tests | 6 | 6 | 15+ |

### Coverage by Module (Current State)

| Module | Coverage | Status |
|--------|----------|--------|
| run_projection_service.py | 100% | ✅ Complete |
| environment.py | 94.8% | ✅ Excellent |
| sentence_analyzer.py | 92.1% | ✅ Excellent |
| aggregates.py | 98.9% | ✅ Complete |
| pipeline.py | 83.5% | ✅ Good |
| projection_service.py | 85.7% | ✅ Good |
| event store/repository | 94-95% | ✅ Excellent |
| parked_events.py | 100% | ✅ Complete |

---

## Phase 9: Architectural Test Cleanup ⏳ IN PROGRESS

**Goal:** Address 44 skipped tests that were invalidated by M3.0 architecture changes
**Priority:** P1 - Technical debt from architecture migration
**Status:** In Progress (2026-01-25)

### Architecture Context (IMPORTANT - Read Before Compaction)

**M3.0 changed the write architecture:**
- **OLD (M2.x):** Pipeline → Neo4j (direct write, immediate consistency)
- **NEW (M3.0):** Pipeline → EventStoreDB → Projection Service → Neo4j (eventual consistency)

**Key implications for tests:**
1. `Neo4jAnalysisWriter` and `Neo4jMapStorage` direct writes are DEPRECATED
2. The projection service is now the SOLE writer to Neo4j
3. Tests cannot verify Neo4j state immediately after pipeline runs
4. EventStoreDB is the source of truth, not Neo4j
5. Neo4j can be completely rebuilt from events (projection rebuild)

**Dual-write was removed in M3.0** - There is no longer any code path that writes
directly to Neo4j from the pipeline. All Neo4j writes go through projection handlers.

### P9.1: Remove Deprecated Direct-Write Tests ⏳ IN PROGRESS
**File:** `tests/integration/test_neo4j_analysis_writer_lifecycle.py`
**Tests:** 11 | **Action:** DELETE | **Effort:** Low

These tests exercise deprecated direct-write functionality:
- Writer initialization/finalization lifecycle
- Concurrent direct writes to Neo4j
- Thread safety of direct writer
- Error recovery in direct write mode

**Why delete (not refactor):**
- Write lifecycle is now handled by projection service subscription manager
- Thread safety is handled by lane manager (consistent hashing by aggregate_id)
- Error recovery is handled by checkpoint/retry in subscription manager
- All these behaviors are already tested in `tests/projections/` (150 tests)

**Existing coverage that replaces these tests:**
- `test_projection_handlers_unit.py` - handler logic
- `test_subscription_manager.py` - subscription lifecycle
- `test_lane_manager.py` - concurrent processing
- `test_projection_service.py` - end-to-end projection

### P9.2: Refactor Data Integrity Tests for Projection Pattern 📋 PLANNED
**File:** `tests/integration/test_neo4j_data_integrity.py`
**Tests:** 11 | **Action:** REFACTOR | **Effort:** Medium

These tests verify VALID behaviors that still matter:
- Transaction atomicity (data written correctly)
- Relationship integrity (graph structure valid)
- No orphaned nodes
- Constraint validation

**Refactor approach:**
```python
# OLD: Direct write (immediate)
await writer.write_result(analysis_data)
# Verify immediately...

# NEW: Event + projection (eventual)
event = create_analysis_generated_event(analysis_data)
await event_store.append(stream_name, event)
handler = AnalysisGeneratedHandler()
await handler.handle(event)
# Now verify Neo4j state...
```

**Tests to refactor:**
- `test_transaction_atomicity_success` → verify projection handler atomicity
- `test_relationship_consistency` → verify projection creates correct relationships
- `test_relationship_uniqueness` → verify idempotent projection behavior
- `test_cascade_relationship_integrity` → verify shared dimension node reuse
- `test_cross_component_data_sync` → verify event → projection → Neo4j flow
- `test_data_update_consistency` → verify projection update handling
- `test_orphaned_node_detection` → verify no orphans after projection
- `test_data_type_consistency` → verify projection data types
- `test_constraint_validation` → verify projection constraint compliance

### P9.3: Fault Tolerance Tests ✅ DEFERRED
**File:** `tests/integration/test_neo4j_fault_tolerance.py`
**Tests:** 7 | **Action:** KEEP SKIPPED | **Status:** Deferred

**Rationale for deferral:**
1. EventStoreDB client library handles connection retry internally
2. Projection service retry/parked events logic already tested in Phase 0:
   - `test_subscription_manager.py` - checkpoint management, error handling
   - `test_projection_handlers_unit.py` - retry logic, parked events
   - `test_parked_events_unit.py` - event parking and replay
3. The file already has comprehensive skip documentation explaining M3.0 architecture
4. Writing new EventStoreDB fault tolerance tests requires infrastructure setup
   (simulating ESDB failures) that is better suited for production monitoring

**Existing coverage that addresses fault tolerance concerns:**
- `tests/projections/test_projection_handlers_unit.py::TestBaseHandlerRetryLogic`
- `tests/projections/test_parked_events_unit.py` (20 tests, 100% coverage)
- `tests/projections/test_subscription_manager.py` (checkpoint recovery)

**Keep skipped until:** Production deployment reveals specific fault scenarios to test

### P9.4: Defer Performance Benchmarks to Production Baseline 📋 PLANNED
**File:** `tests/integration/test_neo4j_performance_benchmarks.py`
**Tests:** 7 | **Action:** DEFER | **Effort:** N/A

**Why defer:**
- Current baselines were established pre-M2.8
- M3.0 architecture has different performance characteristics
- Event emission + projection overhead is intentional
- Should re-baseline from production metrics, not synthetic tests

**When to revisit:**
- After production deployment provides real-world performance data
- When establishing SLOs for projection throughput
- Consider measuring projection latency, not direct write speed

### P9.5: Performance Test Configuration ✅ DOCUMENTED
**File:** `tests/integration/test_performance.py`
**Tests:** 2 | **Action:** KEEP SKIPPED | **Status:** Documented

These are performance tests (not correctness tests) with known configuration issues:

1. `test_concurrent_projection_processing` - Flaky due to test order sensitivity
   - **Issue:** Shared state pollution from other tests
   - **Skip reason:** Already documented in code
   - **Future fix:** Requires better test isolation infrastructure

2. `test_concurrent_file_processing` - test_mode not propagated
   - **Issue:** Pipeline writes to production Neo4j, test verifies against test Neo4j
   - **Skip reason:** Already documented in code
   - **Future fix:** Add test_mode to PipelineOrchestrator (requires refactoring)

**Rationale for keeping skipped:**
- Performance tests, not correctness tests
- Coverage already at 90%+
- Other performance tests in the file ARE running and passing
- Fixing requires infrastructure changes beyond Phase 9 scope

### P9.6: Keep Projection Rebuild Tests (Already Correct) ✅ DONE
**File:** `tests/integration/test_projection_rebuild.py`
**Tests:** 2 | **Action:** KEEP | **Status:** Correct

These tests already use the correct pattern:
- Pipeline processes file → creates events only
- Events processed through projection handlers → creates Neo4j state
- Delete Neo4j → Replay events → Verify state rebuilt correctly

**Current skip reason:** Requires valid OpenAI API key (conditional skip is correct)

### Summary Table

| Sub-Phase | File | Tests | Action | Priority | Status |
|-----------|------|-------|--------|----------|--------|
| P9.1 | test_neo4j_analysis_writer_lifecycle.py | 11 | DELETE | High | ✅ |
| P9.2 | test_neo4j_data_integrity.py | 8 | REFACTOR | High | ✅ |
| P9.3 | test_neo4j_fault_tolerance.py | 7 | KEEP SKIPPED | Medium | ✅ |
| P9.4 | test_neo4j_performance_benchmarks.py | 7 | DEFER | Low | ✅ |
| P9.5 | test_performance.py | 2 | DOCUMENTED | Medium | ✅ |
| P9.6 | test_projection_rebuild.py | 2 | KEEP | N/A | ✅ |
| | **Total** | **37** | | | |

**Phase 9 outcome:**
- 11 tests deleted (test_neo4j_analysis_writer_lifecycle.py removed)
- 8 tests refactored (data integrity now uses projection handlers)
- 7 tests kept skipped (fault tolerance - covered by projection tests)
- 7 tests deferred (performance baselines - wait for production data)
- 2 tests documented (configuration issues - performance tests, not correctness)
- 2 tests kept as-is (projection rebuild - conditional skip is correct)
- Skipped tests reduced from 44 to ~33 (removed deprecated, refactored data integrity)

---

## Notes

### Test Guidelines
- All new tests should use `pytest-asyncio` for async code
- Integration tests should use `@pytest.mark.integration` marker
- E2E tests should use `@pytest.mark.e2e` marker
- Mock external services (EventStoreDB, Neo4j, LLM APIs) in unit tests
- Use fixtures from `conftest.py` for common setup

### Phase Priority
1. **P5 Critical** - Address 0% coverage and core business logic gaps
2. **P6 Domain** - Strengthen aggregate and domain model testing
3. **P7 Integration** - Modernize tests for event-sourced architecture
4. **P8 E2E** - Validate complete user workflows

### Skipped Test Categories (54 total)
- **M2.8 Architecture Changes** (~35): Tests assume direct Neo4j writes
- **Performance Baselines** (~8): Outdated for event-sourced flow
- **Environment-Specific** (~6): Windows permissions, Docker detection
- **Optional Dependencies** (~5): openpyxl, psutil not installed
