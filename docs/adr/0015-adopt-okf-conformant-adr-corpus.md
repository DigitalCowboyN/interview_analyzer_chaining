---
type: ADR
id: 15
title: Adopt an OKF-conformant, non-blocking ADR corpus for architectural decisions
status: accepted
date: 2026-07-30
supersedes: []
superseded_by: []
tags: [adr, knowledge-management, okf, tooling]
source: docs/superpowers/specs/2026-07-30-adr-okf-knowledge-system-design.md
---
## Context
Architectural decisions were real but buried in spec prose, discoverable only
by reading milestone specs end to end. One silent supersession already
happened — the M4.6 GraphRAG-ask spec overrode the 2026-07-04 spec's "borrow
neo4j-graphrag-python" line with no back-pointer from the superseded text —
and no agent instruction surface pointed at the decision corpus at all.

## Decision
Adopt `docs/adr/` as an OKF v0.1-conformant ADR bundle: one markdown file per
decision (OKF frontmatter + a short MADR-style body linking out to the
milestone spec), with generated `index.md`/`log.md`. Pair it with a
five-layer, entirely non-blocking knowledge loop: read (inject the ADR index
as context on architectural prompts), capture (nudge after a spec lands),
and guard (`make adr-check` for structural integrity, spec-references-ADR,
and staleness) — none of it ever fails a command, blocks a commit, or stops
a tool call.

## Consequences
Past decisions are discoverable before new ones are made; supersession edges
(like ADR-0008 → ADR-0014) are explicit and bidirectionally checked instead
of silent; adoption depends entirely on visibility — nothing enforces it, so
the corpus can still rot if agents ignore the surfaced context.

## Alternatives considered
Blocking enforcement — hooks or checks that fail commands or fail commits
(rejected: adoption should be by visibility, not gates); machine-generating
ADR content from specs (rejected: backfill is human-curated and one-time, so
ADRs stay accurate rather than mechanically restated).
