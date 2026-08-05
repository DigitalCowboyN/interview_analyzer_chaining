# Capabilities

What the system can do, linked to the code map (`../code/`).

## core

### ask-the-corpus
Answer an analyst's free-form question over the whole corpus with hybrid graph + vector retrieval and one grounded, cited answer.

- **implemented_by:** ask, ask.engine, ask.reader
- cited-synthesis — Turn fused retrieval results into one grounded, cited answer via a single synthesis call. (ask.engine)
- hybrid-retrieval — Retrieve and fuse graph and project-scoped vector search results over Neo4j for a question. (ask.reader, ask.engine)

### correct-the-analysis
Let an analyst correct anything the AI produced — text, speakers, segments, lens items, resolutions — each correction appended as a new event, never a silent rewrite.

- **implemented_by:** api, commands, events
- correct-resolution — Approve, reject, or reverse an automatic entity/person resolution suggestion via the resolution router. (api, commands, events)
- edit-text — Correct verbatim transcript text through the edits router — appended as a reviewable event, never a silent rewrite. (api, commands, events)
- override-lens-items — Override an incorrect lens-extracted item (e.g. a wrong action item) via the lenses router, locking it against future re-runs. (api, commands, events)
- remove-segments — Remove a topic segment from the analysis via the segments router, preserving the full event history. (api, commands, events)
- rename-reattribute-speakers — Rename a speaker or reattribute an utterance to a different speaker via the speakers router. (api, commands, events)

### enrich-fragments
Classify each fragment's function, structure, and purpose — the analytic backbone the workbench and lenses read.

- **implemented_by:** enrichment, enrichment.orchestrator, enrichment.executor, agents, models
- classify-dimensions — Classify each fragment's function, structure, and purpose along fixed analytic dimensions, one focused LLM call per dimension. (enrichment.executor, agents)
- extract-claims — Pull discrete factual and opinion claims out of each interview via a focused, schema-validated LLM call — raw material for lenses and ask. (enrichment.executor, agents)
- tag-topics-keywords — Tag each fragment with topics and keywords for retrieval, browsing, and downstream lenses. (enrichment.executor, agents)

### export-a-portable-bundle
Produce a portable, git-versionable Markdown+YAML bundle of an interview's lens items and transcript, every item grounded back to the verbatim source.

- **implemented_by:** export, export.reader, export.renderer, export.bundler
- assemble-bundle — Guard, read, render, and write the full OKF export bundle in one in-memory pass. (export.bundler, export.reader)
- render-bundle — Render read-side lens items and transcript into Markdown+YAML documents, generically from node type, properties, and the lens profile. (export.renderer)

### extract-insights-via-lenses
Apply a purpose-built reading of an interview (meeting minutes, persona, …) via one generic, profile-driven engine — no per-lens code.

- **implemented_by:** lens, lens.engine, agents
- per-lens-extractors _(variant)_ — Per-lens extractor calls — `meeting_minutes`' objectives/decisions/action items, `persona`'s traits/goals/quotes — configured entirely by profile, no lens-specific code path. (lens.engine, agents)
- run-lens-engine — Drive any lens profile (YAML + prompts) through one generic engine: run extractors, resolve speaker references, and emit the lens events. (lens.engine)

### ingest-transcripts
Turn raw transcript files into structured, speaker-attributed, stitched utterances the rest of the system analyses.

- **implemented_by:** ingestion, ingestion.orchestrator, ingestion.speaker_inference, ingestion.stitcher, agents
- infer-speakers — Infer which speaker each fragment belongs to when the transcript doesn't label them, via windowed LLM proposals reconciled by majority vote. (ingestion.speaker_inference, agents)
- parse-fragments — Read a transcript file and emit one `SentenceCreated` event per offset-grounded fragment, plus the map file used later for verbatim grounding. (ingestion.orchestrator)
- segment-conversation — Normalize the raw transcript and segment it into sentence-level units via spaCy, before fragment and speaker events are emitted. (ingestion)
- stitch-utterances — Merge utterances split across interruptions back into one continuous statement, without touching the verbatim fragment text. (ingestion.stitcher, agents)

### resolve-entities-and-people
Link speakers to canonical Persons across interviews and canonicalize entity surface forms — cross-interview identity for the analyst view.

- **implemented_by:** resolution, resolution.engine
- canonicalize-entities — Consolidate entity surface-form variants into one canonical entity per real-world referent, via human-reviewed suggestions. (resolution.engine)
- merge-split-link-alias — The deterministic, idempotent resolution operations — merge, split, link, alias — computed as suggestions, never auto-persisted as events. (resolution.engine)
- resolve-persons — Link a speaker's identity across separate interviews to one canonical Person. (resolution.engine)

### serve-workbench-and-gallery
Serve the analyst-facing workbench (read + correct) and gallery (browse across interviews), kept current via live SSE notifications.

- **implemented_by:** api, ui, ui.reader
- gallery-read — Serve the read-only gallery of projects and their lens outputs for browsing across interviews. (ui.reader)
- live-notifications — Push surface-tagged live notifications over SSE so the workbench and gallery stay current without a manual refresh. (ui)
- run-read-queries — Serve the Neo4j read queries — interviews, transcript, personas, worklist, lens items — that back the workbench. (api)
- workbench-write — Accept an analyst's correction from the workbench UI and dispatch it as a command. (api, commands)

## enabling

### maintain-event-source-of-truth
Hold the append-only, frozen-format event log that is the system's sole source of truth — every command validates intent, then appends; nothing rewrites history.

- **implemented_by:** commands, events

### project-events-to-graph
Replay the event log in causal order into Neo4j as the sole writer, maintaining the queryable read model.

- **implemented_by:** projections

### provider-strategy-and-focused-calls
Provide configuration-driven, provider-agnostic LLM access — one focused, schema-validated call per task, with automatic failover across providers.

- **implemented_by:** agents
- chat-failover _(variant)_ — Fail a chat call over to the next configured provider (Anthropic → Claude Code → OpenAI) on an availability error, transparent to the caller. (agents)
- pinned-embeddings _(variant)_ — Pin embedding calls to one configured provider/model — never failed over, since vectors from different models aren't comparable. (enrichment)
