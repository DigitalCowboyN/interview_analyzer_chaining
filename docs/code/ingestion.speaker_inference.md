---
type: CodeUnit
unit: ingestion.speaker_inference
role: pipeline-layer
key_modules: []
---
Speaker genesis for unlabeled transcripts: windowed LLM proposals per fragment, reconciled deterministically by majority vote over overlapping windows.
