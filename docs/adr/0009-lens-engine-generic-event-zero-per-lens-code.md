---
type: ADR
id: 9
title: Lens engine — one generic event, one generic handler, zero per-lens code
status: accepted
date: 2026-07-09
supersedes: []
superseded_by: []
tags: [lens, projection, extensibility]
source: docs/superpowers/specs/2026-07-04-mine-layers-design.md
---
## Context
A lens (meeting_minutes, persona, ...) needs to project purpose-built node
types (Decision, ActionItem, ...) into the graph. Typed per-node events and
handlers per lens would mean new Python and a new subscription allowlist
entry for every lens added.

## Decision
A single generic `LensExtractionGenerated` event (carrying `{lens,
lens_version, node_type, item_id, fields, supporting_fragment_ids,
confidence, model, provider}`) and a single generic `LensExtractionHandler`
MERGE a node labeled from `node_type`, validated against the lens YAML's
declared `projects_to` set — never raw string interpolation from LLM output.
Adding a lens becomes one YAML + prompts: no new Python, no new allowlist
entries, no new handlers.

## Consequences
New lenses ship without touching projection code or the event allowlist;
lens correctness depends on the YAML's `projects_to` validation rather than
code review of a new handler; re-running a bumped `lens_version` supersedes
prior items cleanly via deterministic item ids (`uuid5(...)`).

## Alternatives considered
Typed per-node events (rejected: structurally defeats "new lens with zero
code"); export-only lens outputs (rejected: nothing queryable, nothing to
correct).
