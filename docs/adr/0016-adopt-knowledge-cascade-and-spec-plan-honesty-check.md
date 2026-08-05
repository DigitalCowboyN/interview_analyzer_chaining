---
type: ADR
id: 16
title: Adopt knowledge cascade and spec/plan honesty check
status: accepted
date: 2026-08-05
supersedes: []
superseded_by: []
tags: [adr, knowledge-management, okf, context-engineering, tooling]
source: docs/superpowers/specs/2026-08-05-knowledge-cascade-and-honesty-check-design.md
---
## Context
Seven guarded knowledge domains now exist (adr, api, cli, code, glossary,
graph-queries, prompts), each an OKF bundle with a non-blocking
`make <domain>-check`. But only ADR was surfaced to agents — the other six were
pull-only and undiscoverable to a fresh session, so their checks rarely ran at the
moment of need. The naive fix (describe all seven in `CLAUDE.md`) is the "all at
once" failure Anthropic's context-engineering guidance warns against: keep the
always-loaded surface lightweight, and let agents retrieve detail just-in-time via
lightweight identifiers rather than pre-loading it.

## Decision
Make discovery structural, not injected. Author a single OKF cascade root
`docs/index.md` — one row per domain with a one-line description (the ranking
signal) and its check — and point `CLAUDE.md` at it in three lines. Agents land on
the root and disclose into only the domain they are in.

Add a non-blocking **spec/plan honesty check**: a `PostToolUse(Write)` hook nudges a
per-domain review recorded as a `## Knowledge-graph check` addendum in the spec/plan
itself (verdict: clean · reconciled · overridden). When it surfaces a domain that
should have been consulted but was not, the agent reconciles mechanical gaps and
escalates design-affecting ones to the owner (change the design, or record an
override). A thin guard (`tools/knowledge/`, `make knowledge-check`) mechanizes the
only checkable invariants — the addendum is present on new specs/plans (pre-adoption
files grandfathered by date), and the cascade root covers every domain — never its
semantic correctness.

This refines, and does not supersede, ADR-0015's read→capture→guard loop: the
"read" leg is slimmed from injecting the full ADR table to a provisional one-line
pointer (retired once the cascade demonstrably gets ADRs consulted without it), and
the "capture" nudge is generalized from ADR-only to the whole knowledge graph.

## Consequences
All seven domains are discoverable at the moment of need without bloating
always-loaded context; a spec/plan carries a durable, reviewable record of which
domains it touched and how they were reconciled; standing context actually shrinks
(the ADR table becomes a pointer). Adoption still rests on visibility, not gates —
the honesty check guards that a review was recorded, never that it was correct, so a
skipped-but-mentioned check can pass. The ADR pointer is deliberately provisional.

## Alternatives considered
Per-domain injecting hooks that fire on every matching edit (rejected: runs against
the "minimize injection, trust judgment, naming-as-signal" guidance and re-adds the
brittleness a major model version spent removing); seven policy blocks in `CLAUDE.md`
(rejected: the "all at once" failure); a before-hook for every domain (rejected as
scope creep — only ADR, where a missed decision is catastrophic rather than cheap to
fix later, keeps a provisional before-signal); machine-generating `docs/index.md`
(rejected: seven near-static rows whose descriptions are the human-curated ranking
signal — the coverage guard keeps it honest).
