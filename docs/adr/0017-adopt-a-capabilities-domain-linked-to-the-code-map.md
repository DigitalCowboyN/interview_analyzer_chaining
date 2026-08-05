---
type: ADR
id: 17
title: Adopt a capabilities domain linked to the code map
status: accepted
date: 2026-08-05
supersedes: []
superseded_by: []
tags: [adr, knowledge-management, okf, capabilities, tooling]
source: docs/superpowers/specs/2026-08-05-capabilities-domain-design.md
---
## Context
The guarded knowledge graph had a *how* layer (`docs/code/`, the package/role map)
but no *what* layer — nothing named the system's capabilities in value terms, or tied
them to the code that implements them. This is round 2 of a planned vertical stack
(code → capabilities → use-cases): capabilities are the stable "what the system can
do," the counterpart to the code map's "how."

## Decision
Adopt `docs/capabilities/` as an OKF bundle of `type: Capability` nodes — flat files
linked by `parent:`, each carrying `kind` (primary/child/variant), `tier`
(core/enabling, on primaries), and `implemented_by:` edges to CodeUnit slugs from the
code map. Definitions are authored (the what/for-whom is human judgment, uncomputable
from code); the guard (`make capability-check`, non-blocking) only reconciles:
`implemented_by` links resolve against the code map's unit registry (reused, not
re-hardcoded); every pipeline-layer/surface code unit is claimed by some capability
(coverage — infrastructure/model units advisory); classification and index-sync. The
domain joins the knowledge cascade + registry. `src/` is never touched — capabilities
point *at* the code map rather than annotating code.

## Consequences
"What will this change touch?" now answers in value terms, and orphaned pipeline/
surface code surfaces as a coverage finding (undocumented capability or dead code).
The core/enabling tier keeps the map honest about analyst value versus substrate.
Coverage for shared infrastructure (`utils`, `models`) and the M3.0 legacy
(`io`, `persistence`) is advisory, so drift there is visible but not flagged. The
capability→code links are authored, so they can drift from reality between checks;
the guard catches broken links and coverage gaps, not stale value statements.

## Alternatives considered
Capabilities mirroring `src/` packages 1:1 (rejected: that is just the code map
re-skinned — capabilities are value-framed and may cross packages, e.g.
`correct-the-analysis` spans `api`/`commands`/`events`); reverse `capability:` markers
in `src/` docstrings (rejected this round: capabilities point at the code map, code
stays untouched — a reverse overlay is a possible later round); a capability→code
Mermaid graph (deferred: the index shows the edges; a general cross-domain graph
renderer is the next queued topic). Use-case links are deferred to round 3.
