# Use-Cases — how to think about this domain

This bundle records **direct user-centered input** — the intents that (had they been
written first) would have *led to* this system: requirements, user stories, features, and
formal use cases. It is the graph's **source layer**, the "why" above the capabilities
(`../capabilities/`). The live corpus is **[index.md](index.md)** (generated — grouped by
category → form, with derived coverage). This page is the mental model; read it before
authoring a node.

## One node, fidelity in `form`

A use-case is one node type at varying fidelity. `form` is an **open set**:
`user-story | feature | requirement | use-case`. A lightweight `user-story` carries only
the core; a full Cockburn `use-case` adds the optional block (`level`, `preconditions`,
`main_scenario`, `extensions`, `end_conditions`). Add a `form` value in
`tools/usecase/reader.py`.

## Coverage is derived, never stored

A use-case has **no status field**. Its coverage — `NOT_COVERED / PARTIALLY_COVERED /
FULLY_COVERED` — is computed from its `fulfilled_by` capabilities and how far *their* code
reaches. An uncovered or partially-covered intent is **legitimate and expected**: it is
the domain surfacing where intent outruns implementation.

## Capabilities are read-only here

Links are authored **on the use-case side** (`fulfilled_by:`), so recording that a
capability serves an intent never edits a capability file. `fulfills` (capability →
use-case) is the derived inverse, read backward in the graph.

## Frontmatter

```yaml
---
type: UseCase
form: user-story | feature | requirement | use-case
category: product | operations | supporting   # reuses the capability axis (open set)
actor: <who wants it — person, operator, or external system>
acceptance_criteria:                          # list of strings; may be omitted / empty
  - "Given …, when …, then …"                 # Given/When/Then or a rule sentence
fulfilled_by: [<capability slugs from ../capabilities/>]   # may be [] — legitimate
level: user-goal | summary | subfunction      # optional (Cockburn use-case form)
# optional: preconditions, main_scenario, extensions, end_conditions
---
Narrative: "As a <actor>, I want <action> so that <benefit>." Reach past the code to the
human problem — never restate a capability.
```

## Reconciling

`make usecase-check` (non-blocking) reports: unknown `form`/`category`; empty
`acceptance_criteria` (advisory — not yet testable); `NOT_COVERED` intents (advisory —
nothing fulfills them); and index drift. Cross-domain endpoint integrity (a `fulfilled_by`
pointing at a nonexistent capability) is covered by `make graph-check`. Run
`make usecase-index` after adding or editing a node.
