---
type: Capability
kind: child
parent: ingest-transcripts
implemented_by: [ingestion.orchestrator]
---
Read a transcript file and emit one `SentenceCreated` event per offset-grounded fragment, plus the map file used later for verbatim grounding.
