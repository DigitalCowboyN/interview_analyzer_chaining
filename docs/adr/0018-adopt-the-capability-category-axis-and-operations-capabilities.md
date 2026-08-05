---
type: ADR
id: 18
title: Adopt the capability category axis and operations capabilities
status: accepted
date: 2026-08-05
supersedes: []
superseded_by: []
tags: [adr, knowledge-management, okf, capabilities, tooling]
source: docs/superpowers/specs/2026-08-05-operations-capabilities-design.md
---
## Context
The capability domain (ADR-0017) modelled only *product* capabilities — what the
Interview Analyzer does for analysts, implemented by `src/`. But the repo also has a
substantial *operations* capability — the guarded-knowledge-graph program itself
(catalogs, guards, ADRs, drift detection, cascade, honesty check), implemented by
`tools/`. Capturing it as capabilities is how the work compounds: the program becomes
self-documenting and discoverable through the same map. This required a way to
classify a capability by what kind of thing it serves.

## Decision
Add a **`category`** axis to capabilities — the industry term for grouping a capability
map — as an **open, ordered set**: `product, operations, strategic, supporting`.
`product` (what the app does) and `operations` (what the repo does to stay correct and
advanceable) are populated; **`strategic` and `supporting` are reserved** — recognized
by the guard and skipped by the renderer until a node uses them, so support tools /
systems can be classified later with a one-line change. `category` is orthogonal to
`tier` (core/enabling) and authored on primaries (children inherit).

Author the operations capabilities: one primary, *maintain-a-guarded-knowledge-graph*,
with nine children mapping 1:1 to the `tools.*` code units (the code map was extended to
`tools/` in a prior round for exactly this). Tighten coverage so `tooling` is
**mandatory** — every `tools/` package must be claimed by an operations capability, the
same drift guarantee product capabilities already have. `implemented_by` remains a
"helps fulfill" edge, not "shares code."

## Consequences
The knowledge-graph program is now first-class in the capability map, and a future tool
added without an operations capability is flagged. The product/operations split keeps
the map honest about who a capability serves; the reserved `strategic`/`supporting`
values mean the axis will not need reopening when we build support systems. Coverage of
shared infrastructure/model units stays advisory. Value statements remain authored and
can drift from code between checks; the guard catches broken links and coverage gaps,
not stale prose. (A real instance surfaced during backfill: the file-access capability
was renamed to match what `src/api/routers/files.py` actually does — read-only output
access, not transcript upload.)

## Alternatives considered
Hardcoding a two-value `product|operations` axis (rejected: the research surfaced
`strategic`/`supporting` as standard categories, so the axis is left extensible from the
start); naming the field `class` (rejected: a Python reserved word) or `realm` (rejected:
non-standard — `category` is the industry term); modelling the operations plumbing (hook
wiring, interpreter script) as separate capabilities (rejected: folded into the relevant
tool — coverage is package-level). Refines ADR-0017; supersedes nothing. Use-case links
remain deferred to round 3.
