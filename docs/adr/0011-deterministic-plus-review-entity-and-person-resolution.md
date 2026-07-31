---
type: ADR
id: 11
title: Deterministic-plus-review entity and person resolution, auto-link only within project
status: accepted
date: 2026-07-10
supersedes: []
superseded_by: []
tags: [resolution, entity, person, schema-v2]
governs:
  - src/resolution/
source: docs/superpowers/specs/2026-07-10-layer4-schema-v2-design.md
---
## Context
Graph schema v2 needs to canonicalize entity surface forms ("the dashboard" /
"our analytics dashboard") and link speakers across interviews to real
people, without an LLM adjudicating every ambiguous pair.

## Decision
Entity matching is deterministic + review (no LLM adjudication in v1):
normalized-exact matches auto-merge; embedding-similarity pairs at or above
an auto-merge threshold (default 0.92) auto-merge; pairs in a lower
suggestion band (default 0.80–0.92) become worklist suggestions; human events
(`EntityMergeConfirmed`, `EntitySplit`) lock their targets against future
engine re-runs. Person linking auto-links within a project (exact name or
front-matter match) plus review; cross-project person linking is human-only.

## Consequences
Resolution runs are idempotent and safe to re-run (deterministic ids + state
checks, locked items skipped); ambiguous cases surface on `/review/worklist`
instead of being silently guessed; cross-project identity resolution is
explicitly out of v1 scope.

## Alternatives considered
LLM-adjudicated resolution pairs and nickname dictionaries (deferred to the
backlog, not in v1); automatic cross-project person linking (rejected:
fuzzier-than-exact should never auto-link, and cross-project identity is
human-only).
