---
type: CodeUnit
unit: resolution.engine
role: pipeline-layer
key_modules: []
---
Layer 4 engine: deterministic ids and aggregate-state checks make re-runs idempotent; locked canonicals and blocked pairs are skipped; suggestions are computed but never auto-persisted as events.
