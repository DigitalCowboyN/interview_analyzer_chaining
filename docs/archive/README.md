# M2.8 Migration - Archived Documentation

**Status:** Historical reference
**Date Archived:** 2026-01-11

This directory contains historical documentation from the M2.8 migration to event-sourced architecture. These documents have been archived because they are:
- Superseded by more current documentation
- Historical deep dives into issues that have been resolved
- Planning documents for work that has been completed

## Archived Documents

### Historical Deep Dives
- **M2.8_TEST_FAILURE_DEEP_ANALYSIS.md** - Deep analysis of 4 overwrite test failures
  - **Status:** Issues resolved with dynamic versioning
  - **Superseded by:** M2.8_MIGRATION_SUMMARY.md (implementation details)

### Session Summaries
- **M2.8_SESSION_COMPLETE.md** - Summary from initial migration session
  - **Status:** Superseded by comprehensive migration summary
  - **Superseded by:** M2.8_MIGRATION_SUMMARY.md

- **M2.8_TRANSITION_COMPLETE.md** - Transition completion summary
  - **Status:** Historical milestone, work continued beyond this point
  - **Superseded by:** M2.8_MIGRATION_SUMMARY.md

### Planning Documents
- **M2.3_TO_M2.8_EXECUTION_PLAN.md** - Original execution plan for M2.3→M2.8 transition
  - **Status:** Plan executed, work complete
  - **Superseded by:** M2.8_MIGRATION_SUMMARY.md (what was actually done)

- **M2.3_TO_M2.8_REASSESSMENT.md** - Reassessment of migration approach
  - **Status:** Decisions implemented, work complete
  - **Superseded by:** M2.8_OVERWRITE_BEHAVIOR_ANALYSIS.md (architectural decisions)

- **M2.8_TEST_MIGRATION_PLAN.md** - Test migration planning document
  - **Status:** Tests migrated, plan executed
  - **Superseded by:** M2.8_TEST_MIGRATION_COMPLETE_ANALYSIS.md

## Active Documentation

For current M2.8 architecture and migration information, see:

### Executive Summary (Start Here)
- **../M2.8_MIGRATION_SUMMARY.md** - Complete overview of M2.8 migration
  - What was accomplished
  - Key technical achievements
  - Architecture decisions validated
  - Remaining work recommendations

### Detailed References
- **../M2.8_TEST_MIGRATION_COMPLETE_ANALYSIS.md** - Detailed test failure analysis
  - Categorization of all 62 test failures
  - Regression vs expected failure identification
  - Recommendations for each category

- **../M2.8_OVERWRITE_BEHAVIOR_ANALYSIS.md** - Architectural decision record
  - Analysis of event sourcing patterns
  - Delete-and-replace pattern justification
  - Comparison of architectural approaches

## When to Reference Archived Docs

These archived documents may be useful for:
- Understanding the evolution of architectural decisions
- Learning about specific issues encountered during migration
- Historical context for why certain approaches were taken
- Training materials showing problem-solving process

## Archival Policy

Documents are archived when they:
1. Have been superseded by more current documentation
2. Describe problems that have been fully resolved
3. Contain planning for work that has been completed
4. Are primarily of historical interest

Documents are kept active when they:
1. Serve as current architectural reference
2. Document ongoing or future work
3. Provide operational guidance
4. Are frequently referenced by the team
