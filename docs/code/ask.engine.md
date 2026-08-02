---
type: CodeUnit
unit: ask.engine
role: surface
key_modules: []
---
Ask engine: retrieve -> fuse -> assemble -> one synthesis call. A dead embedder drops the vector channel (flagged); zero hits skip the LLM; synthesis failure still carries the full retrieval result.
