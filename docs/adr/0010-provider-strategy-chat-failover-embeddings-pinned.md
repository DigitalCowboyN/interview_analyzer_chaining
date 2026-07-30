---
type: ADR
id: 10
title: Provider strategy — config-selected chains, chat failover but pinned embeddings
status: accepted
date: 2026-07-05
supersedes: []
superseded_by: []
tags: [llm, provider, embeddings, reliability]
source: docs/superpowers/specs/2026-07-04-mine-layers-design.md
---
## Context
Every model-touching capability (chat/extraction, embeddings) needs a
provider strategy that survives quota/availability failures without silently
corrupting the comparability of results.

## Decision
Chat/extraction (`BaseLLMAgent`) sits behind a config-selected provider chain
(Anthropic primary, OpenAI, Claude Code harness) with per-call failover on
quota/availability errors (429/5xx) — safe because every event records the
model that produced it. Embeddings (`Embedder`) use OpenAI or local
sentence-transformers with NO silent per-call failover: the provider is
config-pinned, every vector is tagged `{model, dim}`, indexes are per-model,
and "falling back" means flipping config and re-running via event replay.

## Consequences
Chat-based enrichment degrades gracefully under provider outages; embedding
vectors never silently mix incomparable spaces; switching embedding
providers is an explicit, replay-driven operation, not an automatic one.

## Alternatives considered
Uniform failover across chat and embeddings (rejected for embeddings:
vectors from different models live in incomparable spaces, making silent
failover a correctness bug rather than a reliability feature).
