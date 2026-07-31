---
type: ADR
id: 7
title: Focused calls, not one-shot mega-calls
status: accepted
date: 2026-07-04
supersedes: []
superseded_by: []
tags: [llm, extraction, extensibility]
source: docs/superpowers/specs/2026-07-04-mine-layers-design.md
---
## Context
Each enrichment dimension (purpose, entities, claims, topics, keywords, ...)
could be produced by one large structured LLM call per fragment, or by many
small dimension-specific calls.

## Decision
Each enrichment dimension is its own focused LLM call with its own prompt,
response schema, and confidence (the extractor registry pattern). Expanding
the set of calls is good; collapsing them into do-everything calls is
explicitly rejected.

## Consequences
Per-dimension confidence, failure isolation, and independent
tunability/correctability; adding a new enrichment dimension or lens
extractor means registering a new extractor, not touching a shared
mega-prompt; more total LLM calls per fragment.

## Alternatives considered
Collapsing the calls into one structured mega-call (rejected by the owner:
loses per-dimension confidence, failure isolation, tunability, and
correctability).
