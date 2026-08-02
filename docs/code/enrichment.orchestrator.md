---
type: CodeUnit
unit: enrichment.orchestrator
role: pipeline-layer
key_modules: []
---
Layer 2 orchestrator: loads a Layer 1 interview, runs the extractor registry, and emits enrichment events; resume-aware, skips already-analyzed fragments unless forced.
