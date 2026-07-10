# Design: The Mine — Layered Enrichment & Retrieval Architecture (Layers 0–5)

**Date:** 2026-07-04 (updated 2026-07-05)
**Status:** Approved. **Layer 1 (M4.1) SHIPPED** — merged to main via PR #1 on 2026-07-05.
**Builds on:** M3.0 single-writer event-sourced architecture (ESDB → projection service → Neo4j)

## Purpose

The system's goal: take an unstructured interview or meeting transcript and explode it
into a dataset much larger than the source, so it can be mined, rotated, and filtered
for different reporting purposes (persona research, usability findings, meeting
minutes/decisions/action items, and purposes not yet known).

This design keeps the event-sourced core (its strongest asset) and grows the system in
five layers. Each layer is a new set of event types + projection handlers, so each
ships independently and historical data can be re-enriched by replaying the event log.

## Guiding decisions (agreed with owner)

1. **Lens architecture** — a purpose-neutral core that always runs, plus pluggable
   lens profiles for specific reporting purposes. Not one vertical hardcoded.
2. **Preserve, never rewrite** — the transcript is stored verbatim; all interpretation
   (speakers, stitching, classifications) is additive, grounded, and correctable.
3. **Focused calls, not one-shots** — each enrichment dimension is its own focused
   LLM call with its own prompt, response schema, and confidence. Expanding the set of
   calls is good; collapsing them into do-everything calls is explicitly rejected.
4. **The map is the coordinate system** — every downstream artifact grounds to stable
   fragment IDs + character offsets in the immutable source.
5. **Mixed input formats** — ingestion detects and normalizes; worst case is fully
   unlabeled flat text requiring speaker inference from nothing.
6. **Primary consumption (first phases):** OKF report bundles + richer structured
   queries; ask-the-corpus GraphRAG retrieval follows as a later phase.

## Layer 0: Input boundary (out of scope)

A separate system produces the raw interview text. This system is **not** responsible
for audio processing, transcription, or diarization — text arrives as an input, and we
assume it arrives in a bad state: potentially one long undifferentiated string of
sentences, no speaker labels, dialogue disjointed by crosstalk and interruptions.

Reference example: `data/input/GMT20231026-210203_Recording.txt` — a real transcript
exhibiting exactly this (continuous string, unlabeled interleaved speakers, rapid
overlapping exchanges).

Contract: **text in, nothing assumed.**

## Layer 1: Ingestion, the Map, speaker genesis, and stitching

Pipeline order: normalize → segment fragments with offsets → build map →
speaker inference pass → stitching pass. Each pass emits events; each is
independently correctable by the end user.

### The Map (upgraded conversation map)

- Source text stored immutably, content-hashed.
- Segmentation produces **fragments**: contiguous runs of speech in source order. A
  fragment is often an incomplete sentence in crosstalk-heavy transcripts.
- Every fragment: stable ID + `start_char`/`end_char` offsets into the source.
- `sequence_order` reflects as-spoken order and is immutable; the `FOLLOWS` chain
  always represents verbatim order.
- All downstream extractions reference map IDs + spans, never raw text.

### Format detection & normalization

Labeled formats (Zoom/Teams/Otter exports, `Speaker:` prefixes) are parsed directly —
speaker modeling is a parsing problem. Unlabeled flat text falls through to inference.
Interview front matter (participants, date, project, method) is captured as
OKF-compatible YAML and stored via `InterviewMetadataUpdated`; known participants seed
speaker inference. The Interview aggregate gets a proper UUID identity (replacing
filename-as-key).

### Speaker genesis (inference from nothing)

1. Windowed LLM pass proposes speaker boundaries and provisional handles (S1, S2, …)
   with per-assignment confidence.
2. Whole-document consistency pass reconciles handles across windows using linguistic
   fingerprints (question-asking patterns, first-person content, verbal tics).
3. Events: `SpeakerCreated` (provisional: true), `FragmentAttributed`
   (speaker, confidence, actor: system).

Corrections (follow existing edit-protection rules — user events survive
regeneration): `SpeakerRenamed`, `SpeakerMerged`, `SpeakerSplit`,
`FragmentReattributed`.

### Stitching (overlay, never rewrite)

A second LLM pass identifies utterance continuity across interruptions and emits
**relationship data only** — the fragment sequence is untouched:

- `UtteranceIdentified` — one speaker's continuous thought spanning 1..n fragments.
- Fragment→Utterance membership with ordinal (`PART_OF_UTTERANCE {position}`).
- `INTERRUPTS` — utterance B broke into utterance A between specific fragments.
- `OVERLAPS` — where the source shows simultaneous speech.

This enables two visualizations later: linear as-spoken (interruption edges crossing
the timeline — the realistic view) and stitched-by-utterance (each person's complete
thought). Stitches carry confidence; corrections: `StitchCorrected`, `StitchRemoved`.

### Error handling

Low-confidence attributions and stitches are committed with their confidence values,
not parked — a visible wrong guess the user can correct beats a gap. Confidence
thresholds are config; "all attributions below 0.7" is a queryable review worklist.

### Layer 1 completion notes (2026-07-05, as shipped in M4.1)

Delivered per plan (`docs/superpowers/plans/2026-07-04-layer1-ingestion-map-speakers-stitching.md`),
with these as-built notes that later layers should treat as current reality:

- **Entry point:** `python -m src.ingestion <file>` (IngestionOrchestrator). The legacy
  `src/pipeline.py` flow is untouched and runs in parallel; Layer 2 must decide its fate.
- **Deferred from Layer 1** (unchanged intentions): `OVERLAPS` edges, OKF front-matter
  capture (→ Layer 5 era), `StitchCorrected` as a distinct event (remove + re-identify
  covers v1), LLM-based window reconciliation (deterministic overlap voting shipped),
  live-LLM golden evaluation for prompt tuning.
- **Corrections shipped:** rename/merge/split speakers, reattribute fragments, remove
  stitches — 202+version, `X-User-ID` provenance, human events lock fields.
- **Hard lesson → standing checklist item:** a new event type is NOT delivered until
  (1) handler registered in bootstrap, (2) event type added to the subscription
  allowlists in `src/projections/config.py`, (3) Sentence-stream payloads carry
  `interview_id` for lane routing, (4) handlers raise (not no-op) when cross-stream
  MATCH targets aren't projected yet. A drift-guard unit test now enforces (1)+(2);
  an integration smoke test (`tests/integration/test_layer1_projection_smoke.py`)
  covers the path end-to-end. Every later layer inherits this checklist.
- **Environment note:** the 26 live-LLM integration tests fail on provider quota
  (OpenAI 429 insufficient_quota) as of 2026-07-05 — Layer 2's enrichment work needs
  working provider credit.

## Layer 2: Core enrichment — the Extractor Registry

The existing 7-call pattern is kept and formalized as the template for all enrichment.
Each dimension is a registered, focused extractor:

```yaml
extractor:
  name: sentence_purpose
  prompt_file: prompts/core/purpose.yaml
  response_model: SentencePurposeResponse   # Pydantic, schema-enforced
  context_needs: [observer_context]
  scope: fragment            # fragment | utterance | segment | document
  target_field: purpose
```

### Improvements to the existing 7 (keep every call, strengthen each)

- Schema-enforced structured outputs (API-level) replace "please respond in JSON";
  Pydantic validation becomes a backstop, not a coin flip.
- Numeric confidence (0–1) replaces string confidence.
- Speaker- and utterance-aware context: ContextBuilder upgrade so windows show
  `[S1]:` labels and can supply the stitched utterance a fragment belongs to.
- Optional spaCy cross-check on function/structure types — flags disagreement as a
  review signal; does not replace the LLM calls.

### New focused extractors (expansion, same pattern)

- `entity_mentions` — people, orgs, products, tools; span-grounded (char offsets
  within the fragment, LangExtract-style traceability). Emits `EntitiesExtracted`;
  projects `Entity` nodes + `MENTIONS {start, end}` edges. Canonicalization/resolution
  remains Layer 4; surface form + type stored now.
- `claim_extraction` — utterance-scoped (the stitching payoff): what the speaker
  asserts, commits to, asks. Emits `ClaimExtracted`; projects `Claim` nodes with
  `MADE_BY` → Speaker and `SUPPORTED_BY` → fragments.
- Embeddings per fragment and per utterance (not an LLM call) in Neo4j vector
  indexes — realizes planned M3.1. Vectors ride as `EmbeddingGenerated` events with
  the vector inline (base64 float32): keeps single-writer and replay purity; the
  direct-write alternative was rejected as breaking the architecture's core rule.

Scopes matter: some extractors run per-fragment (as today), some per-utterance
(claims — enabled by stitching), some per-segment/document. The registry handles
fan-out.

### Provider strategy (M4.2 decision — the baseline config pattern)

Every model-touching capability sits behind an interface with a config-selected
provider chain. This generalizes the old single-provider agent factory into the
system's standing pattern:

- **Chat/extraction (`BaseLLMAgent`)** — providers: Anthropic (current primary;
  Haiku-class model for enrichment calls), OpenAI, and **Claude Code harness**
  (headless `claude -p` subprocess as a backend — clear behavior replication; the
  one caveat, accepted by the owner, is that it does not exercise the raw API
  lifecycle in the interface). Per-call failover down the chain on quota/availability
  errors (429/5xx) is safe because every event already records the model that
  produced it.
- **Embeddings (`Embedder`)** — providers: OpenAI (`text-embedding-3-small`) and
  local sentence-transformers. **No silent per-call failover**: vectors from
  different models live in incomparable spaces. The provider is config-pinned,
  every vector is tagged `{model, dim}`, indexes are per-model, and "falling back"
  means flipping config and re-running the embedding extractor via event replay.

### M4.2 scope additions (agreed 2026-07-05)

- API-level structured outputs in the agent layer (`call_model(prompt, schema=None)`;
  OpenAI json_schema response_format, Anthropic forced tool-use) — no M3.2 dependency.
- ContextBuilder v2: contexts rendered with `[S1]:` speaker labels from the Layer 1
  graph, plus `utterance_context` (the full stitched thought).
- Enrichment orchestrator: `python -m src.enrichment <interview_id>` (+ `--enrich`
  one-shot flag on ingestion); resume-aware (skips fragments already analyzed at the
  current extractor version); API `/analysis/` router and Celery task rewired to
  ingest+enrich.
- **Legacy retirement (owner decision):** after a parity check on a sample transcript,
  delete `src/pipeline.py`, `sentence_analyzer.py`, `analysis_service.py`,
  `pipeline_event_emitter.py`, the local .jsonl analysis writers, and their tests.
  The registry becomes the only enrichment path. Sequenced last, gated on parity.
- Every new event type follows the Layer 1 projection-delivery checklist.

## Layer 3: Lens engine

A lens is a declarative YAML profile contributing extractors to the same registry,
plus mapping rules to graph node types:

```yaml
lens: meeting_minutes
version: 2
extractors: [objectives, decisions, action_items, followups]
few_shot_examples: prompts/lenses/meeting/examples.yaml
projects_to: [Objective, Decision, ActionItem, FollowUp]
```

- Core extractors always run; lens extractors run when a lens is applied — at ingest
  or any time later, replayed from the event log (a new lens can run over every
  interview ever ingested with zero upstream re-processing).
- Results emit `LensExtractionGenerated {lens, version, ...}`; lens outputs version
  independently; re-running an improved lens supersedes cleanly.
- User corrections (`LensExtractionOverridden`) follow existing edit-protection rules.

First lens: **meeting_minutes** (exercises speakers hardest — action items need
owners). Second: **persona** (re-expresses the current UX taxonomy plus claims: pain
points, goals, behaviors, notable quotes).

### M4.3 design decisions (agreed 2026-07-09 — "Approach A: fully generic engine")

- **One generic event, one generic handler, zero per-lens code.**
  `LensExtractionGenerated` (Interview stream) carries
  `{lens, lens_version, node_type, item_id, fields: {…}, supporting_fragment_ids,
  confidence, model, provider}`. A single `LensExtractionHandler` MERGEs a node
  labeled from `node_type` — **validated against the lens YAML's declared
  `projects_to` set, never raw string interpolation from LLM output** — sets
  `fields` as flat properties, links `SUPPORTED_BY` → fragments plus lens
  metadata properties. Adding a lens = one YAML + prompts; no Python, no new
  allowlist entries, no new handlers. (Typed per-node events were rejected:
  they structurally defeat "new lens with zero code"; export-only lens outputs
  were rejected: nothing queryable, nothing to correct.)
- **Lens extractors are ordinary `ExtractorSpec`s** run by the M4.2 executor —
  focused calls, own prompts/response models, scopes fragment/utterance/
  **document** (document scope implemented in this milestone; objectives need
  whole-transcript context).
- **Declarative per-type mappings in the lens YAML** cover typed-projection
  needs without code: e.g. `speaker_link: {field: owner, relationship: OWNED_BY}`
  resolves an extracted owner handle/name to the interview's Speaker and links it.
- **Deterministic item ids**: `uuid5(interview:lens:node_type:source_unit:ordinal)`.
  Re-running a bumped `lens_version` supersedes prior items (old nodes for that
  lens+interview replaced at projection); re-running the same version is
  idempotent (skip existing ids).
- **Corrections**: `LensExtractionOverridden` follows the Layer 1 lock pattern
  (human edits to a lens item survive lens re-runs).
- **Trigger surface v1 (owner decision): CLI only** — `python -m src.lens
  <interview_id> <lens_name>`; ingest-flag and API endpoint deferred.
- **Debt first (owner decision):** the M4.2-exit debt list (ROADMAP M4.2
  section) is burned down as the opening tasks of the M4.3 plan, before lens
  work begins.
- meeting_minutes v1 extractors: `objectives` (document scope),
  `decisions` + `action_items` (utterance scope, speaker-linked),
  `followups` (utterance scope).

## Layer 4: Graph schema v2

New projections; existing Sentence/Topic/Keyword nodes remain — fragments are the
grounding layer everything points into.

| Node | Key relationships |
|------|-------------------|
| `Interview` (UUID, front-matter props) | `HAS_PARTICIPANT` → Speaker |
| `Speaker` (provisional flag, display name) | `SPOKE` → Utterance |
| `Utterance` (stitched thought) | `PART_OF_UTTERANCE` ← Fragment; `INTERRUPTS` → Utterance |
| `Fragment` (today's Sentence + offsets) | `FOLLOWS` (as-spoken, unchanged) |
| `Segment` (topic episode) | `CONTAINS` → Fragment |
| `Entity` (canonical) | `MENTIONED_IN` ← Fragment (with span); `ALIAS_OF` |
| `Claim` | `MADE_BY` → Speaker; `SUPPORTED_BY` → Fragment spans |
| Lens nodes (`Decision`, `ActionItem`, …) | `OWNED_BY`/`DECIDED_BY` → Speaker; `SUPPORTED_BY` → Fragment |

Entity resolution (canonicalizing "the dashboard" / "our analytics dashboard") uses
`neo4j-graphrag-python` resolution utilities — a borrowed component, not an adopted
pipeline (adopting its pipeline would bypass event sourcing and is rejected).

## Layer 5: Consumption

1. **OKF exporter** — a projection rendering purpose-filtered subgraphs into Open
   Knowledge Format bundles (Google Cloud OKF v0.1: markdown + YAML frontmatter,
   `type` required; markdown links as graph edges). One directory per export;
   `index.md` carries interview front matter; one file per Decision/ActionItem/
   insight; `SUPPORTED_BY` rendered as quoted verbatim fragments with source
   references. Git-versionable, agent-consumable. OKF's role is confined to front
   matter (ingest) and export bundles (egress) — not internal storage.
2. **Richer REST queries** — decisions-by-meeting, pain-points-by-speaker,
   low-confidence review worklists.
3. **GraphRAG retrieval** (later phase) — `neo4j-graphrag-python` hybrid retrievers
   (vector + fulltext + Cypher) over the vector indexes and graph.

## Build order

Layer 1 → Layer 2 → Layer 3 (meeting lens) → Layer 5 (OKF export) → Layer 4
refinements (entity resolution) → GraphRAG retrieval.

Each phase = new events + projection handlers, independently shippable and testable
in the established unit/integration pattern.

## Testing strategy

- Standard unit + integration coverage per the existing pattern (aggregates, handlers,
  projections).
- **Golden-transcript fixtures** for LLM-dependent passes (speaker genesis,
  stitching): a small corpus of messy crosstalk transcripts (seeded from
  `data/input/GMT20231026-210203_Recording.txt`) with hand-verified expected
  structures, so inference-quality regressions are caught deterministically.
- Extractor registry tested with mocked agents (existing pattern in
  `tests/agents/`).

## Rejected alternatives

- **Adopting `neo4j-graphrag-python`'s construction pipeline wholesale** — writes
  directly to Neo4j, bypassing event sourcing; loses edit protection, replay, and
  lens re-runs. Borrow retrievers/resolution utilities only.
- **Reporting-first thin slice** (OKF export over current data) — exports would lack
  speakers/decisions/entities; proves plumbing without advancing the mine.
- **Collapsing the 7 calls into one structured mega-call** — rejected by owner:
  focused calls give per-dimension confidence, failure isolation, tunability, and
  correctability.
- **Rewriting/reflowing disjointed transcripts** — rejected: interpretation must be
  an overlay so the interview can be seen as it actually happened.

## Research references

- [Google Cloud: Open Knowledge Format](https://cloud.google.com/blog/products/data-analytics/how-the-open-knowledge-format-can-improve-data-sharing) (v0.1, June 2026)
- [LangExtract](https://github.com/google/langextract) — span-grounded structured extraction
- [neo4j-graphrag-python](https://neo4j.com/docs/neo4j-graphrag-python/current/) — retrievers, entity resolution, Neo4j 2026.01 SEARCH support
- [Awesome-GraphRAG](https://github.com/DEEP-PolyU/Awesome-GraphRAG) — survey of the GraphRAG landscape
