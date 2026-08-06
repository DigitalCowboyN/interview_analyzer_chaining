# Tests — how to think about this domain

This bundle is the **test suite as a graph node set**, and what each test **verifies** —
the layer that closes the Requirements Traceability Matrix (intent → capability → code →
**test**). The live view is **[index.md](index.md)** (generated). This page is the mental
model.

## Test nodes are derived, never authored

With ~1,600 test functions, `Test` nodes are **derived** per file by scanning `tests/`
(like the code map). `test_type` (`unit | integration | e2e`, an open set) is derived from
path + filename. There are no Test markdown files to maintain.

## What a test verifies — two sources

- **Derived → code:** a unit test in `tests/<pkg>/` verifies the matching code unit, by the
  tests-mirror-source convention. No authoring.
- **Authored → intent:** an integration/e2e test that validates a use-case's acceptance
  criteria or a capability carries a module-level marker:

  ```python
  # verifies: use-cases:correct-what-the-system-got-wrong
  ```

  The `<domain>:<id>` is prefix-resolved; a marker pointing at a nonexistent node is caught
  by `make graph-check`.

## Verification is derived and orthogonal to implementation

A node's **verification** state (`UNVERIFIED | PARTIALLY_VERIFIED | VERIFIED`) is separate
from its **implementation** coverage (`../use-cases/`, `../capabilities/`). A use-case can
read `FULLY_COVERED` + `UNVERIFIED` — built but not yet proven. Verification rolls up
transitively (a use-case through its fulfilling capabilities' tested code) plus any direct
acceptance marker.

## Reconciling

`make testmap-check` (non-blocking) reports: unknown `test_type`; a test that verifies
nothing the graph can see (no target, no marker); `UNVERIFIED` use-cases (the honest gap);
and index drift. Run `make testmap-index` after adding tests or markers.
