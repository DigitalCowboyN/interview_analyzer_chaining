---
type: ADR
id: 6
title: Preserve, never rewrite (overlay-not-rewrite)
status: accepted
date: 2026-07-04
supersedes: []
superseded_by: []
tags: [mine, data-model, overlay]
source: docs/superpowers/specs/2026-07-04-mine-layers-design.md
---
## Context
Interpreting a messy, disjointed transcript (speaker attribution, stitching
continuity across interruptions) risks "cleaning up" the source in a way that
hides what actually happened.

## Decision
The transcript is stored verbatim; all interpretation (speakers, stitching,
classifications) is additive, grounded to stable fragment IDs + character
offsets, and correctable — never a rewrite of the source. Stitching, for
example, emits relationship data only; the fragment sequence itself stays
untouched.

## Consequences
The interview can always be viewed as it actually happened (linear
as-spoken) as well as through an interpreted lens (stitched-by-utterance);
every downstream artifact must ground to fragment IDs + offsets rather than
restated text; corrections are new events, not text edits.

## Alternatives considered
Rewriting/reflowing disjointed transcripts into clean utterances (rejected:
destroys the ability to see the interview as it actually happened).
