---
type: ADR
id: 34
title: UI is project-scoped; projects carry a derived kind
status: accepted
date: 2026-09-27
supersedes: []
superseded_by: []
governs:
  - frontend/src/app/
  - src/ui/project_kind.py
  - tools/dev/
tags: [adr, ui, frontend, navigation, projects, testing]
source: docs/superpowers/specs/2026-09-27-ui-nav-theme-organization-design.md
---
## Context

The M5.0 UI mirrored the backend's CQRS split in its navigation: separate
Workbench and Gallery roots, each asking for the project again, with the gallery
holding its selection in component state. With real data loaded, the owner found
navigation disorienting (Back lost the project), and the project list unusable:
integration tests mint a project per run for isolation, so ~146 projects — almost
all test debris — sat beside the one real project, indistinguishable because
projects carry no name or kind.

## Decision

1. **The UI is project-scoped.** The project is the top-level navigation scope
   (`/projects/:id/...`), selected from a header switcher and carried only in the
   URL. The CQRS split stays a backend concern; the UI presents one surface with
   per-project tabs.
2. **Project kind is derived at read time, not stored.** A single prefix table in
   `src/ui/project_kind.py` classifies a project id as `real` or `test` (with a
   suite). The read API exposes it; the UI groups all test projects into one
   "Test runs" bucket.
3. **Tests keep one project per run** (isolation for assertions and teardown) and
   must tear down what they create.

## Consequences

- Back/Forward/reload restore UI state; old `/workbench` and `/gallery` routes
  redirect.
- A new test suite with a new project-id prefix must be added to the prefix
  table, or its runs appear as real projects.
- No event-schema change or migration; changing classification is a one-table edit.
- `test-runs` becomes a reserved project id.

## Alternatives considered

- **Tests share one project.** Rejected: cross-run interference in project-level
  assertions (personas, persons, counts) and unsafe teardown.
- **Store kind/name via a `ProjectCreated`-style event.** Rejected for now (YAGNI):
  requires an event-schema change and backfill for a distinction a naming
  convention already carries. Revisit if projects gain user-facing names.
- **Keep Workbench/Gallery roots, persist selection in the URL.** Rejected: fixes
  Back but keeps two parallel nav trees and a re-selection per surface.
