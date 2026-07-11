# M4.5 — Layer 4: Graph Schema v2 (design)

**Status:** approved 2026-07-10
**Parent spec:** `docs/superpowers/specs/2026-07-04-mine-layers-design.md` (Layer 4 section)
**Structure:** ONE spec (this document), THREE sequenced implementation plans:
M4.5a (debt + rename) → M4.5b (resolution) → M4.5c (segments). Each plan is
independently shippable (own branch/PR, own review cycle).

## Goal

The full Layer 4 table from the parent spec: the graph becomes schema v2 —
fragments under their honest name, canonical entities with aliases, real
cross-interview people, and topic-episode segments — while the event log stays
immutable and every inference remains a correctable overlay.

Decisions locked during brainstorming: **full schema v2** scope (not
resolution-only); entity matching is **deterministic + review** (no LLM
adjudication in v1); person linking is **auto within project + review**
(fuzzier-than-exact never auto-links; cross-project is human-only); segments
come from a **focused document-scope LLM extractor** (not derived from
per-fragment topics).

## The wire-format line (binds all three plans)

Stored events are immutable history. Therefore, FROZEN FOREVER and documented
as wire format:

- Event type names (`SentenceCreated`, `SentenceEdited`, `AnalysisGenerated`, …)
- `aggregate_type` values (`"Sentence"`, `"Interview"`) inside envelopes
- Stream names (`Sentence-{id}`, `Interview-{id}`)

Everything else — graph labels, Python naming, docs, query surfaces — may
rename. Where a code rename would only create wire-format confusion, a
docstring note ("projects as `:Fragment`") is used instead.

---

## M4.5a: Debt burndown + Fragment rename

### Debt (from the consolidated backlog's "M4.5-entry debt" section — verbatim scope)

1. Markdown escaping for LLM text in rendered link titles / analysis table
   cells (newlines, `]`, `|` corrupt index links and tables).
2. Entity/speaker slug collisions and empty slugs (uniquify with `-2` suffix;
   hash fallback for punctuation-only surfaces).
3. Speaker rollup queries unbounded → push pagination/caps into Cypher.
4. Exporting a never-applied (but valid) lens → 422 "lens never applied"
   instead of a vacuous-guard near-empty bundle.
5. Exports route sync file IO (rmtree/write/zip) → `run_in_executor`.
6. Offsets-invariant-with-front-matter test parametrized for the FLAT path.
7. Small leftovers: rollup substring name-filter behavioral test with
   non-empty data; renderer item_path/title derivation DRY-up; bundler
   rmtree→write staging via temp-dir + rename (closes the non-atomic window).

### Rename (dual-label, migration, code surface)

- **New projections** MERGE `(:Fragment:Sentence {...})` — both labels, the
  pattern the lens engine proved with `(:LensItem:<Label>)`.
- **Migration CLI** (`python -m src.projections.migrate_fragment_label` or a
  make target): one idempotent Cypher —
  `MATCH (s:Sentence) WHERE NOT s:Fragment SET s:Fragment` — safe to re-run,
  run against dev and test DBs; the Layer-4 smoke asserts the dual-label
  invariant (`every :Sentence is :Fragment and vice versa`).
- **Query surface**: every query we own (export reader, projection handlers'
  MATCHes, API routers, integration smokes, docs) moves to `:Fragment`.
- **Deprecation shim**: the `:Sentence` label continues to be written and
  retained through M4.5; dropping it is a backlog item.
- **Code surface**: repositories/new code adopt Fragment naming with thin
  deprecated aliases kept (`get_sentence_repository` aliases
  `get_fragment_repository`; `Sentence = Fragment` class alias for the
  deprecation window); the aggregate class renames `Sentence` → `Fragment`
  with stream-naming constants untouched. Per-model vector index DDL STAYS on
  `:Sentence` for M4.5 — nodes carry both labels so the indexes keep serving;
  re-targeting to `:Fragment` would create duplicate indexes for nothing and
  is deferred to the shim-label drop.
- **Why first**: every M4.5b/c handler and query is born writing `:Fragment`;
  zero rename churn on new code.

---

## M4.5b: Resolution core (canonical entities + Person identity)

### The Project aggregate (new)

Canonical entities and people span interviews → they need a cross-interview
event-sourcing home. New aggregate `Project` (`Project-{project_id}` streams,
`AggregateType.PROJECT = "Project"`), with the full projection-delivery
checklist: handler registration in bootstrap, new `$ce-Project` subscription +
allowlist in `src/projections/config.py`, lane routing, drift-guard pins
updated. The aggregate state holds `canonical_entities: Dict[canonical_id,
{name, entity_type, surfaces, locked}]` and `persons: Dict[person_id,
{display_name, links: [(interview_id, speaker_id)], locked_links}]`.

### Events

- `EntityCanonicalized {canonical_id, name, entity_type, surfaces: [str],
  method: "deterministic"|"human", confidence}` — canonical_id =
  uuid5(`{project_id}:entity:{normalized_name}:{entity_type}`).
- `EntityAliasAdded {canonical_id, surface, method, confidence}`
- `EntityMergeConfirmed {canonical_id, merged_canonical_id}` (human; locks both)
- `EntitySplit {canonical_id, surfaces_removed: [str], new_canonical_id,
  new_name}` (human; locks)
- `PersonIdentified {person_id, display_name}` — person_id =
  uuid5(`{project_id}:person:{normalized_display_name}`).
- `SpeakerLinkedToPerson {interview_id, speaker_id, person_id,
  method: "exact_name"|"front_matter"|"human", confidence}`
- `PersonLinkRemoved {interview_id, speaker_id, person_id, note}` (human;
  locks that speaker against re-linking)

Human events lock their targets: locked canonical entities and locked person
links are skipped by engine re-runs and never overwritten (the M4.3 lens-item
locking discipline).

### ResolutionEngine (`src/resolution/`)

Mirrors LensEngine's shape: `ResolutionEngine(config).apply(project_id,
force=False) -> ResolutionResult`; patchable `_build_embedder` seam; CLI
`python -m src.resolution <project_id> [--force]`; actor SYSTEM
`user_id="resolution"`.

Pipeline per run:
1. Read the project's projected surface entities and speakers from Neo4j via
   `src/resolution/reader.py` (the engine's own input reads — Layer 4
   concern; `src/export/reader.py` remains Layer 5's consumer Cypher home).
2. **Entity candidates:** normalize surfaces (casefold, strip leading
   articles the/a/an + possessives, naive plural-fold); exact-after-
   normalization groups auto-merge. Then embedding cosine similarity over
   surface strings (config-pinned embedder; one embed call per distinct
   surface, cached in-run): pairs ≥ `auto_merge_threshold` (config, default
   0.92) auto-merge; pairs in `[suggest_threshold=0.80, auto)` become
   worklist suggestions; below, ignored.
3. **Person candidates:** within the project, speakers whose display_name
   matches exactly (case-insensitive) or matches a front-matter participant
   auto-link (`method` accordingly); first-name-only or fuzzy → suggestion.
4. Emit events for auto-merges/links not already in aggregate state
   (idempotent re-runs: deterministic ids + state checks, mirroring
   LensEngine); locked items skipped.
5. Suggestions are NOT events — computed on demand (see worklist below); the
   log stays clean until a human confirms or the engine is certain.

### Projections

```
(:CanonicalEntity {canonical_id, name, entity_type, project_id, locked?})
(:Entity)-[:ALIAS_OF]->(:CanonicalEntity)          // surface -> canonical
(:Person {person_id, display_name, project_id, locked?})
(:Speaker)-[:IDENTIFIED_AS {method, confidence}]->(:Person)
```

Handlers follow the M4.3 generic-handler discipline (raise on missing MATCH
targets for cross-stream ordering, `_raise_if_no_writes`).

### Corrections API (`src/api/routers/resolution.py`, speakers-router pattern)

- `POST /resolution/{project_id}/entities/merge` `{surviving_canonical_id,
  merged_canonical_id}` → EntityMergeConfirmed
- `POST /resolution/{project_id}/entities/{canonical_id}/split`
  `{surfaces: [...], new_name}` → EntitySplit
- `POST /resolution/{project_id}/persons/{person_id}/link`
  `{interview_id, speaker_id}` / `.../unlink` → SpeakerLinkedToPerson /
  PersonLinkRemoved
- All 202 `{status, version}` / 404 / 409, actor HUMAN from `X-User-ID`.

### Consumer upgrades

- `GET /review/worklist` gains `entity_merge_suggestions` and
  `person_link_suggestions` sections (computed by the engine's candidate
  logic exposed through a reader function; each row carries what the
  corrections endpoints need).
- `GET /speakers/rollup` groups by Person when links exist; unlinked speakers
  fall back to display-name grouping with `"linked": false` flagged per row.
- OKF bundle: `entities/<slug>.md` keys on CanonicalEntity (name + aliases
  listed + mentions aggregated across its surface entities); speaker files
  link to `/persons/<slug>.md` when identified (new Person concept file,
  `type: Person`).

---

## M4.5c: Segments (topic episodes)

- **Extractor:** `topic_segments`, document scope, in `config/extractors.yaml`
  + `prompts/core_extractors.yaml`; response model
  `SegmentsResult {segments: [{topic: str, start_index: int, end_index: int,
  confidence}]}` (strict-compliant, ints ≥ 0).
- **Orchestrator:** enrichment gains a document pass — one
  `run_spec_on_text(spec, document_text)` (speaker-labeled transcript, the
  LensEngine `_document_text` shape) after the fragment/utterance passes.
  Resume-aware: interviews whose aggregate already has segments skip unless
  forced.
- **Validation at emit time (never a failed enrichment):** indices must be
  existing fragment sequence numbers; ranges ordered, non-overlapping,
  start ≤ end; invalid proposal → drop ALL segments + flag
  `topic_segments_invalid` (stitcher precedent: degrade, don't guess).
- **Events (Interview stream):** `SegmentIdentified {segment_id
  (uuid5 `{interview_id}:segment:{ordinal}`), topic, start_index, end_index,
  confidence}`; `SegmentRemoved {segment_id, reason}` (human correction;
  redraw = remove + re-run). Aggregate state: `segments: Dict[segment_id,
  {topic, start_index, end_index, removed}]`.
- **Projection:** `(:Segment {segment_id, topic, interview_id, confidence})
  -[:CONTAINS]->(:Fragment)` (one edge per fragment in range); SegmentRemoved
  DETACH DELETEs.
- **Consumers:** `transcript.md` renders segment headings (topic as section
  title above the segment's utterances); `GET /interviews/{id}/segments`
  returns segments with fragment ranges; corrections:
  `DELETE /segments/{interview_id}/{segment_id}` (202, X-User-ID).

---

## Error handling summary

| Failure | Behavior |
|---------|----------|
| Migration re-run | Idempotent (label SET is a no-op on labeled nodes) |
| Resolution on unknown project | ValueError / 404 |
| Embedder unavailable during resolution | Run fails cleanly (retryable); events already emitted are valid and the idempotent re-run completes the remainder (enrichment's resume semantics) |
| Human merge of locked/unknown entity | 409 via domain ValueError |
| Segment proposal invalid (bad indices/overlap) | Drop all + `topic_segments_invalid` flag; enrichment continues |
| Segment events for unprojected fragments | Handler raises for retry/park (ordering guard) |

## Testing strategy

- TDD per task, established unit patterns (mocked seams; no API keys).
- M4.5a: migration CLI dual-label invariant test; every moved query has its
  existing test updated to `:Fragment`; all four existing smokes stay green.
- M4.5b: engine unit tests on canned rows (normalization table-driven;
  threshold boundaries; locked-skip; idempotent re-run); aggregate
  replay tests; handler tests; router tests; drift guard pins grow.
- M4.5c: validation table-driven tests; orchestrator document-pass test;
  projection + transcript-rendering tests.
- Integration: a Layer 4 smoke per plan wave, converging on one end-to-end
  smoke (ingest → enrich+segments → lens → resolve → export) with canned LLM;
  all previous smokes stay green throughout.

## Non-goals (M4.5) — onto the backlog

- GraphRAG retrieval (M4.6 — now properly unblocked).
- Dropping the `:Sentence` shim label.
- LLM-adjudicated resolution pairs; nickname dictionaries.
- Cross-project person auto-linking (human-only in v1).
- Segment redraw as a first-class event (remove + re-run covers v1).
- Corpus-level exports.
