---
type: ADR
id: 14
title: Hand-rolled hybrid retrieval instead of adopting neo4j-graphrag-python
status: accepted
date: 2026-07-16
supersedes: [8]
superseded_by: []
tags: [graphrag, ask, retrieval]
source: docs/superpowers/specs/2026-07-16-m46-graphrag-ask-design.md
---
## Context
ADR-0008 (2026-07-04) planned to borrow `neo4j-graphrag-python`'s hybrid
retrievers for GraphRAG ask-the-corpus retrieval. By the time the ask-the-
corpus design (M4.6) was written, the repo had grown its own mature async
query layer, and adding the dependency would mean a sync-driver adapter and
an embedder-interface wrapper for what is roughly three Cypher patterns the
repo already knows how to write and test.

## Decision
Retrieval is hand-rolled in the repo's own idiom: vector, fulltext, and
graph-anchored channels fused with reciprocal-rank fusion (RRF) — an idea
borrowed from `neo4j-graphrag-python`, not the dependency itself. This
supersedes ADR-0008's "borrow neo4j-graphrag-python retrievers" line.

## Consequences
No sync-driver adapter or embedder wrapper is needed; retrieval channels are
plain async functions testable like the rest of the repo's reader modules
(`src/ask/reader.py`); the system carries its own RRF/fusion code instead of
a library dependency.

## Alternatives considered
`neo4j-graphrag-python` as a dependency (rejected: sync-driver adapter +
embedder wrapper + un-idiomatic testing for ~3 Cypher patterns); Text2Cypher
as the primary retrieval mechanism (rejected: ungrounded generated Cypher
against a frozen schema, non-deterministic tests, against the
focused-calls/traceability doctrine); a retrieval-only first slice (rejected:
would defer the part that proves the graph answers questions).
