# Documentation

This directory contains documentation for the Interview Analyzer project.

## Project Roadmap

**[ROADMAP.md](ROADMAP.md)** - Complete project roadmap including:
- Completed milestones (M1 through M2.8)
- Upcoming work (M2.9 User Edit API, M3.0 Remove Dual-Write)
- Planned features (M3.1 Vector Search, M3.2 AI Agent Upgrade)
- Dependency upgrade schedule (neo4j 6.x, openai 2.x, etc.)

---

## M2.8 Event-Sourced Architecture

### Executive Summary (Start Here)
📄 **[M2.8_MIGRATION_SUMMARY.md](M2.8_MIGRATION_SUMMARY.md)** - Complete overview of the M2.8 migration
- Mission accomplished: 100% of M2.8 core tests passing
- Key technical achievements (dynamic versioning, edit protection, cardinality)
- Architecture decisions validated
- Remaining work recommendations

**Status:** ✅ Core M2.8 migration complete (696/776 tests passing, 89.7%)

## M2.8 Architecture Reference

### Detailed Analysis
📄 **[M2.8_TEST_MIGRATION_COMPLETE_ANALYSIS.md](M2.8_TEST_MIGRATION_COMPLETE_ANALYSIS.md)**
- Categorization of all 62 test failures
- Regression vs expected failure identification
- Detailed recommendations for each category
- New tests needed for M2.8

### Architectural Decisions
📄 **[M2.8_OVERWRITE_BEHAVIOR_ANALYSIS.md](M2.8_OVERWRITE_BEHAVIOR_ANALYSIS.md)**
- Event sourcing patterns in M2.8
- Delete-and-replace pattern justification
- Comparison of architectural approaches
- Rejection of `is_current` flag approach

## Key M2.8 Concepts

### Event-First Dual-Write
- EventStoreDB is the immutable source of truth
- Neo4j writes are secondary (temporary during transition)
- Events MUST succeed before Neo4j writes
- Projection service rebuilds Neo4j from events

### Dynamic Event Versioning
- Events are versioned by reading the stream
- Enables re-analysis scenarios (overwrite behavior)
- Each `AnalysisGenerated` event gets a unique version

### Edit Protection
- User edits preserved across system-generated updates
- Projection handlers query for `is_edited` flags
- Edited relationships kept, unedited ones replaced

### Cardinality Enforcement
- Limits applied at event emission time (source)
- Keywords limited to 6 by default (configurable per project)
- Ensures data quality at the source

## Architecture Journey

### Phase 1: M2.2 Dual-Write (Current)
- Pipeline writes to BOTH Neo4j (direct) AND EventStoreDB (events)
- Event-first order: Events → Neo4j
- Validation period for consistency

### Phase 2: M2.8 Single-Writer (Future)
- Remove direct Neo4j writes from pipeline
- Events are ONLY source
- Projection service is SOLE writer to Neo4j

## File Organization

### Active Documentation (Current Reference)
```
docs/
├── M2.8_MIGRATION_SUMMARY.md                    ⭐ Start here
├── M2.8_TEST_MIGRATION_COMPLETE_ANALYSIS.md    📊 Test details
├── M2.8_OVERWRITE_BEHAVIOR_ANALYSIS.md         🏗️  Architecture
└── archive/                                     📦 Historical docs
    ├── README.md
    ├── M2.8_TEST_FAILURE_DEEP_ANALYSIS.md
    ├── M2.3_TO_M2.8_EXECUTION_PLAN.md
    ├── M2.3_TO_M2.8_REASSESSMENT.md
    ├── M2.8_SESSION_COMPLETE.md
    ├── M2.8_TEST_MIGRATION_PLAN.md
    └── M2.8_TRANSITION_COMPLETE.md
```

### Archived Documentation
📦 **[archive/](archive/)** - Historical documentation from M2.8 migration
- Planning documents for completed work
- Deep dives into resolved issues
- Session summaries superseded by migration summary
- See [archive/README.md](archive/README.md) for details

## Test Organization

### M2.8 Tests (Event-First Pattern)
📄 `tests/integration/test_neo4j_analysis_writer.py`
- 14 tests using event-first dual-write pattern
- 100% passing (validates M2.8 architecture)
- Uses `process_events_through_projection()` helper
- Example: `test_single_dimension_edit_protection`

### Legacy Tests (Direct-Write Pattern)
📄 `tests/integration/test_neo4j_analysis_writer_legacy.py`
- 27 tests using direct Neo4j write path (deprecated)
- Skipped in CI (marked with deprecation reason)
- Kept for backward compatibility reference
- Will be removed in M3.0

## Quick Reference

### Running M2.8 Tests
```bash
# Run all M2.8 analysis writer tests
pytest tests/integration/test_neo4j_analysis_writer.py -v

# Run dual-write pipeline tests
pytest tests/integration/test_dual_write_pipeline.py -v

# Run projection handler tests
pytest tests/projections/test_projection_handlers_unit.py -v
```

### Key Files Modified in M2.8
- `src/pipeline_event_emitter.py` - Dynamic versioning, cardinality
- `src/projections/handlers/sentence_handlers.py` - Edit protection
- `src/io/neo4j_analysis_writer.py` - Event-first dual-write
- `src/pipeline.py` - Event emission order

## Remaining Work

### High Priority
1. Fix 5 E2E pipeline tests (2-3 hours)
2. Migrate 3 integration tests to M2.8 pattern (1-2 hours)

### Medium Priority
3. Update 10 data integrity tests for eventual consistency (3-4 hours)
4. Decision on 24 legacy tests (keep vs remove)

### Low Priority
5. Rewrite fault tolerance tests for event store (3-4 hours)
6. Update performance baselines (2-3 hours)

See [M2.8_MIGRATION_SUMMARY.md](M2.8_MIGRATION_SUMMARY.md) for detailed recommendations.

## Contributing

When updating M2.8 documentation:
1. Update this README if adding new major documentation
2. Keep M2.8_MIGRATION_SUMMARY.md as the primary reference
3. Archive superseded documents to `archive/` with explanation
4. Update `archive/README.md` when archiving documents

## Questions?

For M2.8 architecture questions:
- Start with [M2.8_MIGRATION_SUMMARY.md](M2.8_MIGRATION_SUMMARY.md)
- Review [M2.8_OVERWRITE_BEHAVIOR_ANALYSIS.md](M2.8_OVERWRITE_BEHAVIOR_ANALYSIS.md) for architectural decisions
- Check [M2.8_TEST_MIGRATION_COMPLETE_ANALYSIS.md](M2.8_TEST_MIGRATION_COMPLETE_ANALYSIS.md) for test details

---

**Last Updated:** 2026-01-11
**M2.8 Status:** Core migration complete, 100% targeted tests passing
