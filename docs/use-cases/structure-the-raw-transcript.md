---
type: UseCase
form: feature
category: product
actor: analyst
acceptance_criteria:
  - "Given a raw transcript file without speaker labels, when it's ingested, then each fragment is emitted with an offset-grounded position and a speaker inferred from context"
  - "Given an utterance split across an interruption, when ingestion runs, then the pieces are stitched back into one continuous statement without altering the verbatim text"
fulfilled_by: [ingest-transcripts, infer-speakers, parse-fragments, segment-conversation, stitch-utterances]
---
As an analyst handed a raw, unlabeled transcript, I want it turned into clean, speaker-attributed, stitched utterances automatically so I can start reading for meaning instead of first doing manual transcript cleanup.
