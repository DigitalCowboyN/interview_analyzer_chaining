---
type: CodeUnit
unit: ingestion
role: pipeline-layer
key_modules: [orchestrator, stitcher, speaker_inference]
---
Layer 1: normalizes a transcript, segments it into offset-grounded fragments, establishes speakers (parsed or inferred), and stitches interrupted utterances back together — all through events.
