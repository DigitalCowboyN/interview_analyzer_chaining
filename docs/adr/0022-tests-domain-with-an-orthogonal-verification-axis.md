---
type: ADR
id: 22
title: Tests domain with an orthogonal verification axis
status: accepted
date: 2026-08-06
supersedes: []
superseded_by: []
tags: [adr, knowledge-management, okf, tests, verification, graph, tooling]
source: docs/superpowers/specs/2026-08-06-tests-domain-design.md
---
## Context

The use-cases domain (ADR-0021) traced intent → capability → code, but nothing recorded
whether any of it was **verified**. The Requirements Traceability Matrix literature
separates *implemented* from *verified* — high implementation coverage with no test
traceability means the wrong things may be tested intensively while what matters is
unproven. The system has ~1,600 test functions and pytest `unit`/`integration` markers,
so tests are code artifacts to be **derived**, not authored.

## Decision

Adopt a **tests domain** (`docs/tests/`, `tools/testmap/`) that closes the RTM:

- **`Test` nodes are derived per test file** by scanning `tests/`. `test_type`
  (`unit | integration | e2e`, an open set) is derived from path + filename. No Test
  markdown files.
- **The `verifies` edge is derived and polymorphic** (Test → CodeUnit | UseCase |
  Capability), from two sources: **derived → code** by the tests-mirror-source path
  convention (the ~1,590 bulk, zero authoring), and **authored → intent** via a
  module-level `# verifies: <domain>:<id>` marker for the few acceptance/e2e tests that
  validate a use-case's criteria or a capability. Endpoints are fully `<domain>:<id>`-
  addressed, so the existing prefix-based `check_endpoints` catches dangling markers — no
  polymorphic resolver needed.
- **Verification is a distinct, orthogonal axis** from implementation coverage:
  `UNVERIFIED | PARTIALLY_VERIFIED | VERIFIED`, derived transitively (a use-case through
  its fulfilling capabilities' tested code) plus any direct authored acceptance marker. A
  use-case can honestly read `FULLY_COVERED + UNVERIFIED` — built but not proven. This is
  computed in the tests domain; the use-cases and capabilities domains are unmodified.
- **The registry gains first-class polymorphic-target support:** a `to_type` may be a
  `|`-delimited set of node types; `check_registry` validates each part and the
  meta-schema renders one edge per part. This is the generalization the polymorphic
  `verifies` edge required.

Guarded by a non-blocking `make testmap-check` (unknown test_type, unmapped test,
unverified use-case, index drift). Self-registers `tools.testmap` (code node +
`map-the-tests` capability).

## Consequences

- The graph is now a full RTM: intent → capability → code → test, with two independent
  coverage dimensions. Uncovered *and* unverified gaps are both visible.
- **This refines ADR-0021**, which forward-looked to "refine the `FULLY_COVERED`
  predicate" in this round. That one-axis note is superseded in substance by the
  **two-axis** model here — implementation coverage is untouched; verification is a
  separate axis. No use-case was silently reclassified.
- Integration/e2e tests that carry no marker surface as advisory "unmapped" findings —
  the honest nudge to author a `# verifies:` link where one is warranted.
- Reserved for later: `acceptance`/`contract` test types (open set); ingesting real
  pass/fail or line-coverage as edge properties (this round derives verification from
  structure + markers, not pytest execution).

## Alternatives considered

- **Authored Test nodes** (like use-cases) — rejected: infeasible against ~1,600 evolving
  tests; would drift instantly.
- **Refining the single `FULLY_COVERED` predicate to require tests** (ADR-0021's original
  note) — rejected: conflates two distinct questions and silently reclassifies the
  corpus; the orthogonal axis is cleaner and more honest.
- **A single non-polymorphic `to_type`** (e.g. just CodeUnit) — rejected: under-describes
  an edge that genuinely targets three node types; the `|`-delimited generalization keeps
  the registry metadata truthful.
