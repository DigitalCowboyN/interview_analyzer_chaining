---
type: ADR
id: 5
title: Layered Mine architecture (ingestion → enrichment → lens → segment → export)
status: accepted
date: 2026-07-04
supersedes: []
superseded_by: []
tags: [mine, layers, architecture]
source: docs/superpowers/specs/2026-07-04-mine-layers-design.md
---
## Context
The system's goal is to turn an unstructured transcript into a dataset much
larger than the source, mined for purposes not all known up front (persona
research, meeting minutes, usability findings, ...), without hardcoding one
reporting vertical.

## Decision
Analysis is organized as five layers, each adding structure over the raw
transcript without rewriting it: (1) conversation structure — fragments,
speakers, utterances; (2) enrichment — per-dimension analysis, entities,
claims, embeddings; (3) lenses — purpose-built readings via a generic engine;
(4) segments — topic episodes; (5) export — OKF bundles. Each layer is a new
set of event types + projection handlers, shipping independently, and
historical data can be re-enriched by replaying the event log.

## Consequences
A new layer or lens can be added without rewriting earlier layers; every
layer inherits the projection-delivery checklist (handler registration,
subscription allowlist, lane routing); build order became Layer 1 → 2 → 3
(meeting lens) → 5 (OKF export) → 4 refinements (entity resolution) →
GraphRAG retrieval.

## Alternatives considered
One hardcoded reporting vertical (rejected in favor of a purpose-neutral core
plus pluggable lens profiles); a reporting-first thin slice — OKF export over
current data (rejected: would lack speakers/decisions/entities, proving
plumbing without advancing the mine).
