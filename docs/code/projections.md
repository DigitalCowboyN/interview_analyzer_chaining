---
type: CodeUnit
unit: projections
role: pipeline-layer
key_modules: []
---
Sole Neo4j writer: replays EventStoreDB category subscriptions in commit-position (causal) order across per-lane reorder buffers, parking and redriving events whose referents aren't ready.
