# Test Coverage Improvement Plan

> **Created:** 2026-01-18
> **Current Coverage:** 66.8% (554 tests passing, 8 skipped)
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
| Projection Service (0%) | 0% | 80% | CRITICAL | P0 |
| Event Store/Repository (62-63%) | 62% | 85% | HIGH | P1 |
| Command Handlers (63%) | 63% | 85% | HIGH | P1 |
| Pipeline (52%) | 52% | 75% | HIGH | P2 |
| API Routers (74-90%) | 82% | 90% | MEDIUM | P3 |

---

## Phase 0: Critical Infrastructure (0% Coverage)

**Goal:** Cover the projection service that has ZERO test coverage.

### P0.1: Handler Registry (`src/projections/handlers/registry.py`)
**Effort:** Low | **Impact:** High

```
Unit Tests Needed:
- test_register_handler_success
- test_register_handler_duplicate_warning
- test_get_handler_found
- test_get_handler_not_found_returns_none
- test_get_registered_types
- test_global_registry_singleton
```

### P0.2: Bootstrap (`src/projections/bootstrap.py`)
**Effort:** Medium | **Impact:** High

```
Unit Tests Needed:
- test_create_handler_registry_all_handlers_registered
- test_create_handler_registry_with_parked_events_manager
- test_create_handler_registry_without_parked_events_manager
- test_interview_handlers_registered
- test_sentence_handlers_registered
```

### P0.3: Metrics (`src/projections/metrics.py`)
**Effort:** Medium | **Impact:** Medium

```
Unit Tests Needed:
- test_counter_increment_and_get
- test_gauge_set_and_get
- test_histogram_record_values
- test_histogram_statistics (min, max, avg, p50, p95, p99)
- test_histogram_memory_limit
- test_metrics_reset
- test_metrics_timer_context_manager
- test_global_metrics_singleton
```

### P0.4: Health (`src/projections/health.py`)
**Effort:** Low | **Impact:** Medium

```
Unit Tests Needed:
- test_get_health_status_healthy
- test_get_health_status_with_parked_events
- test_get_health_status_service_error
- test_uptime_calculation
- test_format_health_response
```

### P0.5: Subscription Manager (`src/projections/subscription_manager.py`)
**Effort:** High | **Impact:** Critical

```
Unit Tests Needed:
- test_subscription_manager_init
- test_start_subscription_creates_persistent_subscription
- test_stop_subscription_cancels_subscription
- test_event_filtering_by_allowlist
- test_event_routing_to_lane_manager
- test_get_status_returns_subscription_info
- test_handle_event_error_with_retry

Integration Tests Needed:
- test_subscription_connection_to_eventstore
- test_subscription_receives_events
- test_subscription_reconnection_on_error
```

### P0.6: Projection Service (`src/projections/projection_service.py`)
**Effort:** High | **Impact:** Critical

```
Unit Tests Needed:
- test_projection_service_init_with_dependencies
- test_start_initializes_all_components
- test_stop_shuts_down_gracefully
- test_get_health_aggregates_component_status
- test_error_during_start_rolls_back

Integration Tests Needed:
- test_full_service_lifecycle
- test_service_processes_events_to_neo4j
- test_service_health_endpoint
```

### P0.7: Run Projection Service (`src/run_projection_service.py`)
**Effort:** Medium | **Impact:** High

```
Unit Tests Needed:
- test_argument_parsing_defaults
- test_argument_parsing_custom_values
- test_signal_handler_graceful_shutdown
- test_logging_configuration
```

---

## Phase 1: Event Sourcing Foundation (62-63% Coverage)

**Goal:** Strengthen the event sourcing core to 85%+

### P1.1: Event Store (`src/events/store.py`)
**Current:** 62% | **Target:** 85%

```
Unit Tests Needed:
- test_connect_success
- test_connect_failure_raises
- test_disconnect_closes_connection
- test_append_events_success
- test_append_events_concurrency_error
- test_append_events_retry_on_transient_error
- test_read_stream_success
- test_read_stream_not_found
- test_read_stream_to_envelope_conversion
- test_get_stream_version
- test_stream_exists

Integration Tests Needed:
- test_full_event_lifecycle_with_eventstore
- test_concurrent_appends_optimistic_locking
- test_large_event_batch_performance
```

### P1.2: Repository (`src/events/repository.py`)
**Current:** 63% | **Target:** 85%

```
Unit Tests Needed:
- test_load_aggregate_found
- test_load_aggregate_not_found
- test_save_aggregate_new
- test_save_aggregate_existing_with_version
- test_save_aggregate_concurrency_conflict
- test_save_with_retry_on_conflict
- test_exists_returns_true_when_found
- test_exists_returns_false_when_not_found
- test_get_version_returns_current_version

Integration Tests Needed:
- test_interview_aggregate_full_lifecycle
- test_sentence_aggregate_full_lifecycle
- test_concurrent_modifications_handled
```

### P1.3: Command Handlers (`src/commands/handlers.py`)
**Current:** 63% | **Target:** 85%

```
Unit Tests Needed:
# InterviewCommandHandler
- test_handle_create_interview_command
- test_handle_update_metadata_command
- test_handle_change_status_command
- test_handle_unknown_command_raises

# SentenceCommandHandler
- test_handle_create_sentence_command
- test_handle_edit_sentence_command
- test_handle_generate_analysis_command
- test_handle_override_analysis_command
- test_handle_preserves_edited_flag
- test_handle_validation_error

Integration Tests Needed:
- test_command_to_event_to_aggregate_flow
- test_command_handler_uses_repository_correctly
```

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
