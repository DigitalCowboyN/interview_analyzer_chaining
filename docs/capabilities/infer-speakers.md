---
type: Capability
kind: child
parent: ingest-transcripts
implemented_by: [ingestion.speaker_inference, agents]
---
Infer which speaker each fragment belongs to when the transcript doesn't label them, via windowed LLM proposals reconciled by majority vote.
