---
type: ADR
id: 19
title: Capabilities are durable intent; implementation is a derived, replaceable link
status: accepted
date: 2026-08-06
supersedes: []
superseded_by: []
tags: [adr, knowledge-management, capabilities]
source: docs/superpowers/specs/2026-08-06-capabilities-as-intent-design.md
---
## Context
The capability domain (ADR-0017/0018) authored `implemented_by` on every node, which
quietly implied a capability must be implemented. That conflates two different things: a
capability is an *expectation* of the product; the code is one *iteration* that reaches
toward it. A real intent that isn't built yet (e.g. "import transcripts") then had no
home, and "how far along is this capability" looked like a missing attribute.

## Decision
A capability is **durable intent** — it is never "built," only *currently implemented* by
an iteration that can be replaced wholesale while the capability stands unchanged. The
degree of implementation lives **entirely in the `implemented_by` links and is derived**,
never an authored `status`/maturity attribute; an **empty or partial `implemented_by` is
legitimate** (an intent current code only partly reaches). `primary`, `child`, and
`variant` are **all intent** — broad, narrower, and alternative *what*, respectively;
`parent` is decomposition, not a how-chain. **Code is the only "how"**; the *how-decisions*
live in ADRs and specs, so there is no middle "how-definition" capability layer. The
`implements` inverse is **derived** from `implemented_by` (never authored as markers in
code). Capability↔use-case is indirect and many-to-many (a later round). Refines
ADR-0017/0018; supersedes nothing.

## Consequences
Aspirational and partially-implemented capabilities are first-class — the map can carry a
product's intent ahead of its code, and "what have we promised but not yet reached" is
derivable from the links (a view deferred to the graph-links work) rather than a field
that rots. Nothing in the schema or guard changed — an `implemented_by: []` node already
passes. The risk: because the degree is derived, a stale or aspirational capability is
only visible by reading its (empty) links, not flagged — intent that never gets
implemented can linger silently. The division of labor (capability = what/why, ADR/spec =
how decided, code = current how) keeps each concern in one place.

## Alternatives considered
A `status: planned | partial | realized` maturity field (rejected: a capability is never
"built," so maturity is the wrong frame — the links already carry it); a middle
"how-definition" capability layer between intent and code (rejected: that is the ADR/spec
corpus — duplicating it would drift from both code and ADRs); authoring the `implements`
inverse as `capability:` markers in `src/`/`tools/` (rejected: capabilities point at the
code map, code stays untouched — the inverse is a free derivation); tying capabilities to
use-cases directly (rejected: the relationship is indirect and many-to-many, and some
capabilities fulfill no use-case).
