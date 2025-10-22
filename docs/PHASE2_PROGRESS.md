# Phase 2: Event-Sourced Architecture Implementation Progress

## Completed Milestones

### ✅ M2.1: Command Layer & API Integration

**Status:** Complete  
**Commit:** d77f890

**Implemented:**

- Command base classes (`Command`, `CommandResult`, `CommandHandler`)
- Interview commands (`CreateInterviewCommand`, `UpdateInterviewCommand`, `ChangeInterviewStatusCommand`)
- Sentence commands (`CreateSentenceCommand`, `EditSentenceCommand`, `GenerateAnalysisCommand`, `OverrideAnalysisCommand`, `RegenerateAnalysisCommand`)
- Command handlers with validation and error handling
- Actor tracking and correlation IDs
- Repository integration for aggregate loading/saving
- Fixed EventStoreDB client API (use `limit` instead of `count`)

**Files Created:**

- `src/commands/__init__.py`
- `src/commands/interview_commands.py`
- `src/commands/sentence_commands.py`
- `src/commands/handlers.py`
- `tests/commands/__init__.py`
- `tests/commands/test_command_handlers.py`

---

### ✅ M2.3: Projection Service Infrastructure

**Status:** Complete  
**Commit:** 66097a5

**Implemented:**

- Configuration for lanes, subscriptions, retry policies
- Lane Manager with 12 configurable lanes
- Partitioned processing by `interview_id` using consistent hashing
- Subscription Manager for ESDB persistent subscriptions
- Category-per-subscription pattern (`$ce-Interview`, `$ce-Sentence`)
- Event type allowlists for filtering
- Parked Events Manager (DLQ) for failed events
- Projection Service orchestrator
- Retry-to-park error handling with exponential backoff

**Files Created:**

- `src/projections/__init__.py`
- `src/projections/config.py`
- `src/projections/lane_manager.py`
- `src/projections/subscription_manager.py`
- `src/projections/parked_events.py`
- `src/projections/projection_service.py`

**Key Features:**

- 12 lanes (configurable via `PROJECTION_LANE_COUNT`)
- In-order processing per interview within each lane
- Parallel processing across lanes
- Persistent subscriptions with automatic checkpoint management
- Exponential backoff: 1s → 2s → 4s → 8s → 16s (max 60s)
- Failed events parked after 5 retry attempts

---

### ✅ M2.4: Projection Handlers Implementation

**Status:** Complete  
**Commit:** 3b7e8e8

**Implemented:**

- Base handler with version checking and retry logic
- Handler registry for event routing
- Interview event handlers:
  - `InterviewCreatedHandler` - Creates Project and Interview nodes
  - `InterviewMetadataUpdatedHandler` - Updates Interview metadata
  - `InterviewStatusChangedHandler` - Updates Interview status
- Sentence event handlers:
  - `SentenceCreatedHandler` - Creates Sentence nodes and links to Interview
  - `SentenceEditedHandler` - Updates Sentence text and sets edited flag
  - `AnalysisGeneratedHandler` - Creates Analysis node and links dimensions
  - `AnalysisOverriddenHandler` - Marks analysis as overridden

**Files Created:**

- `src/projections/handlers/__init__.py`
- `src/projections/handlers/base_handler.py`
- `src/projections/handlers/registry.py`
- `src/projections/handlers/interview_handlers.py`
- `src/projections/handlers/sentence_handlers.py`

**Key Features:**

- Idempotent handlers with version guards
- Automatic retry with exponential backoff
- Failed events parked to DLQ
- Links dimension nodes: FunctionType, StructureType, Purpose, Keywords, Topics, DomainKeywords
- Stores `event_version` on each node for idempotency

---

### ✅ M2.5: Monitoring & Observability

**Status:** Complete  
**Commit:** 0fb22f1

**Implemented:**

- Health check endpoint with detailed component status
- Metrics tracking (counters, gauges, histograms)
- Formatted health status output
- MetricsTimer context manager for operation timing

**Files Created:**

- `src/projections/health.py`
- `src/projections/metrics.py`

**Metrics Tracked:**

- Events processed, failed, parked, skipped
- Processing latency (p50, p95, p99)
- Queue depth per lane
- Neo4j write errors
- Uptime and lane status

**Health Status Includes:**

- Overall service status (healthy/unhealthy)
- Lane status (running, queue depth, events processed/failed)
- Subscription status (running/stopped per subscription)
- Parked events count per aggregate type
- Uptime

---

## Remaining Milestones

### ⏳ M2.2: Dual-Write Integration

**Status:** Not Started  
**Next Steps:**

1. Modify `src/pipeline.py` to emit events during file processing
2. Update `src/io/neo4j_map_storage.py` to emit `SentenceCreated` events
3. Update `src/io/neo4j_analysis_writer.py` to emit `AnalysisGenerated` events
4. Add error handling for event emission failures
5. Maintain backward compatibility (Neo4j writes still happen)

---

### ⏳ M2.6: User Edit API Integration

**Status:** Not Started  
**Next Steps:**

1. Create `src/api/routers/edits.py` with edit endpoints
2. Implement `PUT /interviews/{interview_id}/sentences/{sentence_id}` for sentence edits
3. Implement `PUT /interviews/{interview_id}/sentences/{sentence_id}/analysis` for analysis overrides
4. Integrate with command handlers
5. Return accepted status with version

---

### ⏳ M2.7: Testing & Validation

**Status:** Not Started  
**Next Steps:**

1. Create integration tests for event-sourced file processing
2. Test projection replay (clear Neo4j, replay events)
3. Test user edit workflows
4. Test idempotency (replay same events)
5. Test partition ordering
6. Test parked events and replay
7. Performance validation

---

### ⏳ M2.8: Remove Dual-Write

**Status:** Not Started (After 1-2 weeks validation)  
**Next Steps:**

1. Deploy projection service to production
2. Monitor dual-write for 1-2 weeks
3. Compare Neo4j state from both paths
4. Feature flag to disable direct writes
5. Monitor for 48 hours
6. Remove direct write code if stable

---

## Architecture Summary

### Current State

- **M1 (Core Plumbing):** ✅ Complete
  - Event envelope, domain events, EventStoreDB client, repository pattern, aggregates
- **M2.1 (Commands):** ✅ Complete
- **M2.3 (Projection Infrastructure):** ✅ Complete
- **M2.4 (Projection Handlers):** ✅ Complete
- **M2.5 (Monitoring):** ✅ Complete

### Data Flow (Once M2.2 Complete)

1. **File Upload** → `CreateInterviewCommand` → `InterviewCreated` event → ESDB
2. **Pipeline Processing:**
   - Segments text → `CreateSentenceCommand` → `SentenceCreated` events → ESDB
   - Analyzes sentences → `GenerateAnalysisCommand` → `AnalysisGenerated` events → ESDB
   - **Dual-write:** Also writes to Neo4j directly (for backward compatibility)
3. **Projection Service:**
   - Consumes events from ESDB via persistent subscriptions
   - Routes events to lanes by `interview_id`
   - Handlers update Neo4j (idempotent, version-guarded)
   - Failed events parked to DLQ

### Key Design Decisions

- **12 lanes** (configurable) for partitioned processing
- **Category streams** (`$ce-Interview`, `$ce-Sentence`) for subscriptions
- **Event type allowlists** for filtering (not ESDB filters)
- **Retry-to-park** pattern: 5 retries with exponential backoff, then park
- **Version guards** on Neo4j nodes for idempotency
- **Dual-write** during transition period for safety

---

## Next Actions

1. **Implement M2.2 (Dual-Write Integration)**

   - This connects the existing pipeline to the event-sourced architecture
   - Allows testing the full flow end-to-end

2. **Test End-to-End**

   - Upload a file
   - Verify events in ESDB
   - Verify Neo4j updated by projection service
   - Check health endpoint

3. **Implement M2.6 (User Edit API)**

   - Add edit endpoints
   - Test user edits flow through events to Neo4j

4. **Comprehensive Testing (M2.7)**

   - Integration tests
   - Performance validation
   - Idempotency verification

5. **Production Deployment & Monitoring**
   - Deploy with dual-write enabled
   - Monitor for 1-2 weeks
   - Remove dual-write (M2.8) after validation

---

## Technical Debt / Future Improvements

- [ ] Add Prometheus metrics exporter (currently just in-memory)
- [ ] Implement WebSocket for real-time Neo4j updates (read-your-own-writes)
- [ ] Add CLI tool for replaying parked events
- [ ] Implement event schema versioning and migration
- [ ] Add distributed tracing (OpenTelemetry)
- [ ] Optimize Neo4j queries for bulk operations
- [ ] Add circuit breaker for Neo4j connection failures
- [ ] Implement event archival/compaction strategy

---

## Questions / Decisions Needed

1. **Docker Integration:** Should we add a `projection-service` container to `docker-compose.yml` now or wait until M2.2?

   - **Recommendation:** Add now for easier testing

2. **EventStoreDB Setup:** Do we need to update `docker-compose.yml` to ensure EventStoreDB is properly configured?

   - **Current:** EventStoreDB service exists but may need configuration tweaks

3. **Testing Strategy:** Should we test M2.1-M2.5 in isolation before M2.2, or proceed directly to M2.2?
   - **Recommendation:** Proceed to M2.2 to enable end-to-end testing

---

## Commits Summary

```
0fb22f1 feat(projections): Add monitoring and observability infrastructure
3b7e8e8 feat(projections): Implement projection handlers for Interview and Sentence events
66097a5 feat(projections): Implement projection service infrastructure
d77f890 feat(commands): Implement command layer with handlers for Interview and Sentence aggregates
```

**Total Lines Added:** ~2,700 lines of production code
**Total Files Created:** 21 new files
**Test Coverage:** Command layer has basic tests; projection service needs integration tests (M2.7)
