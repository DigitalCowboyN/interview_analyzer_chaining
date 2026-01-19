# Test Coverage Improvement Plan

> **Created:** 2026-01-18
> **Updated:** 2026-01-19
> **Current Coverage:** 84.2% (902 tests passing, 54 skipped) ✅ TARGET MET
> **Target Coverage:** 80%+

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
| Event Store/Repository (62-63%) | 62% | 85% | HIGH | P1 |
| Command Handlers (63%) | 63% | 85% | HIGH | P1 |
| Pipeline (52%) | 52% | 75% | HIGH | P2 |
| API Routers (74-90%) | 82% | 90% | MEDIUM | P3 |

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

## Phase 2: Pipeline Coverage (52%)

**Goal:** Increase pipeline coverage to 75%+

### P2.1: Pipeline Orchestrator (`src/pipeline.py`)
**Current:** 52% | **Target:** 75%

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

## Phase 4: Resilience & Operations

### P4.1: Parked Events (`src/projections/parked_events.py`)
**Current:** 29% | **Target:** 80%

```
Unit Tests Needed:
- test_parked_event_creation
- test_parked_event_to_dict
- test_park_event_success
- test_park_event_with_error_capture
- test_get_parked_events_success
- test_get_parked_events_empty
- test_get_parked_count
- test_replay_parked_event (future)
```

---

## End-to-End Test Gaps

### E2E Tests Needed:

```
1. test_e2e_complete_interview_workflow
   - Upload file → Process → Query results → Verify Neo4j

2. test_e2e_user_edit_to_neo4j_projection
   - Edit sentence → Event emitted → Projection updates Neo4j

3. test_e2e_projection_service_recovery
   - Kill projection service → Restart → Verify catch-up

4. test_e2e_concurrent_file_processing
   - Upload multiple files → Verify parallel processing

5. test_e2e_api_error_responses
   - Test all error conditions return proper HTTP codes
```

---

## Implementation Priority Matrix

| Phase | Tests | Est. Tests | Coverage Impact | Effort |
|-------|-------|------------|-----------------|--------|
| P0 | Projection Infrastructure | ~45 | +10% | High |
| P1 | Event Sourcing Foundation | ~35 | +8% | High |
| P2 | Pipeline | ~25 | +6% | Medium |
| P3 | API Layer | ~15 | +3% | Low |
| P4 | Resilience | ~10 | +2% | Medium |
| E2E | End-to-End | ~5 | +1% | Medium |
| **Total** | | **~135** | **+30%** | |

---

## Test File Structure

```
tests/
├── projections/
│   ├── test_registry_unit.py          # NEW
│   ├── test_bootstrap_unit.py         # NEW
│   ├── test_metrics_unit.py           # NEW
│   ├── test_health_unit.py            # NEW
│   ├── test_subscription_manager_unit.py  # NEW
│   ├── test_projection_service_unit.py    # NEW
│   ├── test_parked_events_unit.py     # NEW
│   ├── test_lane_manager_unit.py      # EXISTS
│   └── test_projection_handlers_unit.py   # EXISTS
├── events/
│   ├── test_store_unit.py             # NEW
│   ├── test_repository_unit.py        # NEW
│   └── test_core_plumbing_validation.py   # EXISTS
├── commands/
│   ├── test_handlers_unit.py          # EXPAND
│   └── test_commands_unit.py          # EXISTS
├── pipeline/
│   ├── test_pipeline_unit.py          # EXPAND
│   └── test_pipeline_orchestrator_unit.py # EXISTS
└── integration/
    ├── test_projection_service_integration.py  # NEW
    ├── test_event_store_integration.py         # NEW
    └── ... (existing)
```

---

## Execution Plan

### Week 1: Phase 0 (Critical)
- Day 1-2: Registry + Bootstrap tests
- Day 3: Metrics + Health tests
- Day 4-5: Subscription Manager tests

### Week 2: Phase 0 + P1
- Day 1-2: Projection Service tests
- Day 3-4: Event Store tests
- Day 5: Repository tests

### Week 3: P1 + P2
- Day 1-2: Command Handlers tests
- Day 3-5: Pipeline tests

### Week 4: P3 + P4 + E2E
- Day 1-2: API tests
- Day 3: Parked Events tests
- Day 4-5: E2E tests

---

## Success Metrics

| Metric | Current | Week 2 | Week 4 |
|--------|---------|--------|--------|
| Overall Coverage | 66.8% | 75% | 80%+ |
| Projection Coverage | 0% | 70% | 80% |
| Event Sourcing Coverage | 62% | 80% | 85% |
| Pipeline Coverage | 52% | 65% | 75% |
| Tests Passing | 554 | 620 | 690+ |

---

## Notes

- All new tests should use `pytest-asyncio` for async code
- Integration tests should use `@pytest.mark.integration` marker
- E2E tests should use `@pytest.mark.e2e` marker
- Mock external services (EventStoreDB, Neo4j, LLM APIs) in unit tests
- Use fixtures from `conftest.py` for common setup
