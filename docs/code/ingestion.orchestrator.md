---
type: CodeUnit
unit: ingestion.orchestrator
role: pipeline-layer
key_modules: []
---
Layer 1 orchestrator: read -> normalize -> speakers (parse or infer) -> fragment events -> stitch overlay -> map file, all through event-sourced repositories.
