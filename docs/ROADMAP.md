# Project Roadmap

> **This is the canonical project roadmap. Update this document when milestone status changes.**

---

## Quick Status

**Last Updated:** 2026-07-15

| Milestone | Status | Description |
|-----------|--------|-------------|
| M1 | ✅ Complete | Core Plumbing (events, ESDB, aggregates) |
| M2.1 | ✅ Complete | Command Layer |
| M2.2 | ✅ Complete | Dual-Write Integration |
| M2.3-M2.5 | ✅ Complete | Projection Infrastructure |
| M2.7 | ✅ Complete | Testing & Validation |
| M2.8 | ✅ Complete | Event-Sourced Architecture (Production Ready) |
| M2.9 | ✅ Complete | User Edit API |
| M3.0 | ✅ Complete | Remove Dual-Write + neo4j 6.x |
| **TC** | ✅ Complete | Test Coverage Improvement (90.1%) + Phase 9 Cleanup |
| **TC.10** | ✅ Complete | Test Fixes + Infrastructure Integration |
| **M4.1** | ✅ Complete | Layer 1: Ingestion, Map, Speaker Genesis & Stitching |
| **M4.2** | ✅ Complete | Layer 2: Extractor Registry, Provider Chain, Entities/Claims/Embeddings |
| **M3.1** | ✅ Complete | Vector Search (delivered as Layer 2 embeddings + per-model indexes) |
| **M4.3** | ✅ Complete | Layer 3: Generic Lens Engine (meeting_minutes first) + debt burndown |
| **M4.4** | ✅ Complete | Layer 5: OKF Export + front-matter capture + richer queries |
| **M4.5** | ⏳ In Progress | Layer 4: schema v2 (a ✅, b ✅, c: segments) |
| M3.2 | 📋 Partial | AI Agent Upgrade (structured outputs landed; openai 2.x SDK bump still pending) |
| M3.3 | 📋 Planned | Infrastructure Upgrades |

**Current Phase:** M4.5c (segments)
**Tests:** 1123 unit passing, 3 skipped | **Coverage:** 91.63% (unit). Legacy `src/io` + long-skipped suites deleted in M4.3.

---

## Milestone Checklist

### M4.5b: Resolution Core ✅ COMPLETE

**Spec:** `docs/superpowers/specs/2026-07-10-layer4-schema-v2-design.md` (M4.5b section)
**Plan:** `docs/superpowers/plans/2026-07-11-m45b-resolution-core.md`

Cross-interview resolution: canonical entities with aliases and real Person
identities, driven by a new event-sourced `Project` aggregate, a
deterministic+embedding `ResolutionEngine`, human corrections API, and
consumer upgrades (worklist suggestions, Person-grouped rollup,
canonical-keyed OKF bundles).

- [x] Task 1: `src/events/project_events.py` — 7 event payload models +
      `project_aggregate_id`/`canonical_entity_id`/`person_id_for` uuid5
      helpers; `AggregateType.PROJECT`
- [x] Task 2: `Project` aggregate (`src/events/aggregates.py`) — canonical
      entities, persons, blocked links, locking discipline
- [x] Task 3: `ProjectRepository` + factory/getter (`Project-{id}` stream)
- [x] Task 4: Projection handlers for the 7 Project events
      (`src/projections/handlers/resolution_handlers.py`) —
      `(:CanonicalEntity)`/`(:Person)` overlay, `ALIAS_OF`/`IDENTIFIED_AS`
- [x] Task 5: Delivery wiring — bootstrap registration, subscription
      allowlist, lane routing, pin tests (bootstrap pin count → 29)
- [x] Task 6: Resolution reader — the engine's own Neo4j input reads
      (`src/resolution/reader.py`)
- [x] Task 7: Candidate logic — pure functions for exact/embedding grouping
      and person linking (`src/resolution/candidates.py`)
- [x] Task 8: `ResolutionEngine` + CLI (`src/resolution/engine.py`,
      `python -m src.resolution`) — idempotent re-runs, locked/blocked skips
- [x] Task 9: Corrections API (`src/api/routers/resolution.py`) — merge,
      split, link, unlink; 409s from domain `ValueError`
- [x] Task 10: Worklist suggestions + Person-grouped speaker rollup
- [x] Task 11: OKF bundle — canonical entities + Person concept files
- [x] Task 12: Layer 4 integration smoke (`test_layer4_resolution_smoke.py`,
      two interviews, overlapping surfaces, fake embedder, idempotence,
      dual-label invariant) + schema v2 docs

**Completed:** 2026-07-15

**Deferred:** M4.5c (topic segments); LLM adjudication of borderline entity
pairs (deterministic + review only, spec non-goal); cross-project person
linking (human-only, future); embedding cache beyond a single run;
suggestion pagination on the worklist; front-matter participant matching
inside on-demand suggestions (engine-only); `FragmentRepository =
SentenceRepository` class alias + call-site/patch-path flips (existing
backlog, rides the alias drop).

---

### M4.5a: Debt Burndown + Fragment Rename ✅ COMPLETE

**Spec:** `docs/superpowers/specs/2026-07-10-layer4-schema-v2-design.md` (M4.5a section)
**Plan:** `docs/superpowers/plans/2026-07-10-m45a-debt-fragment-rename.md`

Debt burndown (M4.4-exit debt, landed first, per the M4.3 pattern):
- [x] Renderer text safety: `_link_text` / `_cell` escape LLM/graph text in
      markdown link labels and table cells; single-pass `(path, title)`
      derivation shared across item files, index sections, speaker back-links
- [x] `_SlugRegistry`: bundle-wide unique speaker/entity slugs (`-2`, `-3`, …
      suffixes; hash fallback for punctuation-only surfaces)
- [x] `LensNeverAppliedError` → 422 (never-applied lens no longer produces a
      near-empty bundle); `speaker_rollup_rows` scan-capped (`LIMIT $scan_cap`,
      default 5000)
- [x] Staged atomic bundle writes off the event loop (`_write_bundle` via
      `asyncio.to_thread`; write to a sibling staging dir, then swap — a
      failure mid-write leaves the old bundle intact)
- [x] Offsets-invariant-with-front-matter test parametrized for the FLAT
      (unlabeled) path alongside LABELED

Sentence → Fragment rename (dual-label overlay + migration CLI; wire format frozen):
- [x] Dual-label writer: `SentenceCreatedHandler`'s create query keeps
      `MERGE (s:Sentence {sentence_id: $sentence_id})` as anchor and adds
      `SET s:Fragment`; `python -m src.projections.migrate_fragment_label`
      one-shot idempotent migration (`{"relabeled": <n>}`)
- [x] All owned reads flip `MATCH (s:Sentence ...)` → `MATCH (s:Fragment ...)`
      across `src/export/reader.py` and every projection handler; vector-index
      DDL stays on `:Sentence` (shim keeps it serving) until the shim drops
- [x] Code-surface rename: `Fragment` aggregate class (was `Sentence`),
      `get_fragment_repository()` (was `get_sentence_repository`) — both with
      deprecated aliases (`Sentence = Fragment`, `get_sentence_repository =
      get_fragment_repository`); event types, `aggregate_type` value
      (`"Sentence"`), and `Sentence-{id}` stream names are frozen wire format
      and unchanged
- [x] Integration smokes (Layers 1/2/3/5) and data-integrity/idempotency/replay
      tests flipped to `:Fragment`; migration idempotence proven live against
      the test DB (second run relabels 0); `docs/architecture/database-schema.md`
      updated (`:Fragment` primary, `:Sentence` shim + frozen-wire-format note)

**Completed:** 2026-07-11

**Deploy prerequisite:** any environment with pre-existing graph data MUST run
`python -m src.projections.migrate_fragment_label` when deploying this branch
(reads already query `:Fragment`); verified idempotent live.

**Deferred to M4.5b (since completed — see M4.5b section above):** Project
aggregate, resolution engine, corrections. Still outstanding: M4.5c (segments);
dropping the `:Sentence` shim label and deprecated code aliases; re-targeting
vector index DDL to `:Fragment` (rides the shim drop).

---

### M4.4: Layer 5 — OKF Export + Front-Matter Capture ✅ COMPLETE

**Spec:** `docs/superpowers/specs/2026-07-10-okf-export-design.md`
**Plan:** `docs/superpowers/plans/2026-07-10-okf-export.md`

Front-matter capture:
- [x] `parse_front_matter` (tolerant; YAML dates normalized to ISO strings so
      event payloads stay JSON-serializable); `normalize()` segments the body
      with offsets kept absolute into the unmodified source
- [x] Orchestrator stores title/started_at/metadata incl. raw block under
      `metadata["front_matter"]` via existing `InterviewCreated` (no new
      event types)
- [x] Participant speaker seeding: labeled speakers matched by full-name or
      unique-first-name become confirmed (`display_name = participant`,
      method `"front_matter"`); unlabeled participants ride into the
      inference prompt as a hint; ambiguity never seeds

OKF export (Approach: read-side exporter over Neo4j, zero per-lens code):
- [x] `src/export/reader.py`: single Cypher layer (transcript/speakers/lens
      items/claims/entities/latest-analysis + worklist + speaker rollup)
      shared by bundle and API
- [x] `src/export/renderer.py`: pure OKF v0.1 renderer — dual-audience
      bundles; frontmatter delimiter-safety (quoted re-dump when values
      contain `---`); reserved `index.md` (no frontmatter, covers all files)
      and bundler-owned `log.md` (entries grouped by ISO date, newest
      first); bundle-absolute links as edges
      (`DECIDED_BY`/`OWNED_BY`/`MADE_BY` → `/speakers/...`); verbatim
      blockquote grounding linking `/transcript.md` anchors
- [x] `src/export/bundler.py` + CLI: `OkfExporter` with projection-lag
      consistency guard (aggregate expected set = current-version items +
      locked items of any version vs projected ids); renders fully in
      memory before writing;
      `python -m src.export <interview_id> <lens_name> [--out exports] [--zip]`
- [x] API: `GET /exports/{interview_id}/{lens_name}` (zip download;
      404/422/409), `GET /interviews/{interview_id}/lenses/{lens}/items`,
      `GET /review/worklist` (low-confidence + unresolved-reference queue),
      `GET /speakers/rollup` (by-display-name across interviews — v1
      limitation, real identity is Layer 4)
- [x] Layer 5 integration smoke (front matter in → OKF bundle out, real
      ESDB+Neo4j); `scripts/test-integration.sh` runner

**Completed:** 2026-07-10

**Deferred:** GraphRAG retrieval; corpus-level bundles; incremental/diff
exports; importing arbitrary OKF bundles; cross-interview speaker identity
resolution (rollup is display-name match, documented).

---

### M4.3: Layer 3 — Generic Lens Engine + Debt Burndown ✅ COMPLETE

**Spec:** `docs/superpowers/specs/2026-07-04-mine-layers-design.md` (Layer 3 + M4.3 design decisions)
**Plan:** `docs/superpowers/plans/2026-07-09-layer3-lens-engine.md`

Debt burndown (M4.2-exit debt, landed first):
- [x] Per-model embedding properties (`embedding_<model>`) with per-model vector
      indexes; embedder dim validation + OpenAI `dimensions` param
- [x] Resilient failover chain construction (unconstructible providers skipped
      with a warning; empty chain raises)
- [x] Span-keyed MENTIONS edges; EntitiesExtracted provider materialized;
      mixed-provider flag; embed only non-failed fragments; batch CLI per-file
      isolation; strict-schema test recursion hardened
- [x] Dead code deleted: `src/io/` + its tests, two long-skipped legacy
      integration suites, dead `classification` config block

Lens engine (Approach A — fully generic, zero per-lens code):
- [x] Executor: document scope + public `run_spec_on_text` (SpecOutcome);
      fragment/utterance paths refactored onto it
- [x] Lens profile model (`LensSpec`/`load_lens`) — one YAML under `lenses/`
      fully describes a lens; labels and node_types validated
- [x] meeting_minutes lens: objectives (document), decisions / action_items /
      followups (utterance) + prompts; strict-compliant response models
- [x] Three generic Interview-stream events: `LensApplied` (supersession),
      `LensExtractionGenerated`, `LensExtractionOverridden` (human lock)
- [x] Generic projection handlers: dual-label `(:LensItem:<Label>)` nodes
      (validated + sanitized labels), `SUPPORTED_BY` grounding, declarative
      speaker links; bootstrap pins 19 → 22; interview allowlist extended
- [x] LensEngine: owner resolution (SELF / handle / display_name), deterministic
      uuid5 item ids, idempotent re-runs, locked overrides survive `--force`;
      `python -m src.lens <interview_id> <lens_name> [--force]`
- [x] Corrections endpoint: `POST /lenses/{interview_id}/items/{item_id}/override`
- [x] Layer 3 lens smoke test (end-to-end through real projection)

**Completed:** 2026-07-10

**Deferred:** persona lens (next); lens apply via ingest flag/API; same-version
`--force` full re-extraction (needs an item-clearing event — CLI documents the
limitation); OKF export of lens outputs (M4.4).

---

### M4.2: Layer 2 — Extractor Registry & Core Enrichment ✅ COMPLETE

**Spec:** `docs/superpowers/specs/2026-07-04-mine-layers-design.md`
**Plan:** `docs/superpowers/plans/2026-07-05-layer2-extractor-registry.md`

- [x] Agent layer: API-level structured outputs (`call_model(prompt, schema=)` —
      OpenAI json_schema, Anthropic forced tool-use)
- [x] Provider strategy: `ClaudeCodeAgent` (headless `claude -p`) + `FailoverAgent`
      chain (Anthropic Haiku → Claude Code → OpenAI); Anthropic primary
- [x] Extractor registry (`config/extractors.yaml`) + ported prompts with numeric
      confidence; the 7 focused calls kept, never merged
- [x] GraphContextBuilder: speaker-labeled + utterance-aware contexts
- [x] spaCy cross-check flags (function/structure disagreement as review signal)
- [x] New extractors: entity mentions (span-grounded), claims (utterance-scoped)
- [x] Embeddings: `Embedder` protocol (OpenAI + local sentence-transformers,
      config-pinned), events with inline base64 vectors, per-model Neo4j vector indexes
- [x] AnalysisGenerated v2 (dimension confidences, flags, provider provenance)
- [x] Projections: Entity/Claim/embedding handlers + allowlists + drift guard
- [x] Enrichment orchestrator (resume-aware) + `python -m src.enrichment` CLI +
      ingest `--enrich` chaining
- [x] API `/analysis/` + Celery task rewired to ingest+enrich
- [x] Legacy pipeline retired (pipeline.py, sentence_analyzer, context_builder,
      analysis_service, pipeline_event_emitter, llm_responses; ~200 tests)
- [x] Golden/parity assertions; Layer 2 projection smoke test

**Completed:** 2026-07-06 (final-review fix wave 2026-07-09)

**Deferred to M4.3 entry (from the M4.2 final review — tracked debt):**
- Per-model embedding property (`embedding_<model>`) or single-index +
  mandatory `embedding_model` filter — today all per-model vector indexes
  target the shared `embedding` property (cross-contamination risk only when
  two models share dimensions).
- FailoverAgent chain construction is all-or-nothing (one missing API key
  kills the whole chain); skip unconstructible providers with a warning.
- Delete now-dead `src/io/local_storage.py` writers + `config.yaml`
  `classification` block ("registry is the only path" made literal).
- Per-dimension provider provenance (currently last-successful-call per unit);
  MENTIONS edge collapses duplicate mentions of one entity per fragment;
  embedder dim not validated against emitted vectors; EntitiesExtracted
  provider not materialized in graph; batch CLI aborts on first failing file.
- From the fix-wave re-review: strict-schema test recursion doesn't descend
  into inline `properties`/`items` (covered today via `$defs` hoisting);
  zero-claim utterances re-extract on every unforced resume (retry semantics,
  but unbounded); fully-failed fragments still consume embedding compute
  before being discarded.

---

### M4.1: Layer 1 — Ingestion, Map, Speaker Genesis & Stitching ✅ COMPLETE

**Spec:** `docs/superpowers/specs/2026-07-04-mine-layers-design.md`
**Plan:** `docs/superpowers/plans/2026-07-04-layer1-ingestion-map-speakers-stitching.md`

- [x] Offset-preserving segmentation (`segment_text_with_offsets`)
- [x] Ingestion package: format detector (labeled/flat) + normalizer producing
      offset-grounded fragments (`source[start:end] == fragment` invariant)
- [x] Sentence aggregate: `start_char`/`end_char` + correctable speaker
      attribution (`SpeakerAttributed`/`SpeakerReattributed`, human lock)
- [x] Interview aggregate: speaker lifecycle (`SpeakerCreated`/`Renamed`/`Merged`)
      and stitching overlay (`UtteranceIdentified`/`InterruptionRecorded`/`StitchRemoved`)
- [x] Windowed LLM speaker inference with deterministic overlap reconciliation
- [x] Stitcher: baseline grouping + LLM refinement (invalid proposals degrade
      to baseline; transcript text never rewritten)
- [x] Ingestion orchestrator + upgraded map (.jsonl with offsets, speaker,
      confidence, utterance per fragment); `python -m src.ingestion <file>`
- [x] Projection handlers: Speaker + Utterance nodes (`HAS_PARTICIPANT`,
      `SPOKEN_BY`, `SPOKE`, `PART_OF_UTTERANCE`, `INTERRUPTS`)
- [x] Correction API: rename/merge/split speakers, reattribute fragments,
      remove stitches (202 + version; human events lock fields)
- [x] Golden crosstalk fixture (deterministic, recorded LLM responses)

**Completed:** 2026-07-04

---

### M2.9: User Edit API ✅ COMPLETE

- [x] Review existing `src/api/routers/edits.py` implementation
- [x] Complete `POST /edits/sentences/{interview_id}/{sentence_index}/edit` endpoint
- [x] Complete `POST /edits/sentences/{interview_id}/{sentence_index}/analysis/override` endpoint
- [x] Complete `GET /edits/sentences/{interview_id}/{sentence_index}/history` endpoint
- [x] Integration with command handlers (SentenceCommandHandler.handle)
- [x] Return 202 Accepted status with version
- [x] E2E tests passing (3 tests)
- [x] Unit tests passing (16 tests)

**Completed:** 2026-01-18

---

### M3.0: Remove Dual-Write ✅ COMPLETE

- [x] Remove direct Neo4j writes from pipeline
- [x] Projection service becomes SOLE writer
- [x] Remove deprecated code paths
- [x] Remove 27+ legacy tests (41 tests deleted)
- [x] Upgrade neo4j 5.28.1 → 6.x (driver) / 5.26.0 (server)
- [x] Update documentation

**Completed:** 2026-01-18

---

### TC: Test Coverage Improvement ✅ COMPLETE

- [x] Phase 0: Projection Infrastructure (150 tests)
  - Handler Registry, Bootstrap, Metrics, Health
  - Subscription Manager, Projection Service
- [x] Phase 1: Event Sourcing Foundation (81 tests)
  - Event Store (92.5%), Repository (94.6%), Command Handlers (89.4%)
- [x] Phase 2: Pipeline Tests (21 tests)
  - Pipeline coverage: 52% → 81.4%
- [x] Phase 4: Parked Events Tests (20 tests)
  - Parked Events coverage: 29% → 100%
  - Fixed production bugs: invalid EventEnvelope.metadata reference
- [x] Phase 5: Critical Gaps (27 tests)
  - run_projection_service.py: 0% → 100%
  - sentence_analyzer.py: 65% → 92%
- [x] Phase 6: Domain Model (96 tests)
  - aggregates.py: 75% → 99%
  - environment.py: 57% → 95%
  - Fixed bug: detect_environment() f.read() called twice
- [x] Phase 7: Integration Test Infrastructure (10 tests)
  - Makefile Python auto-detection (pyenv/Homebrew/system)
  - ESDB environment-aware connection (host/docker/CI)
  - `make test-infra-up/down` and `make test-integration-full` targets
  - 75 integration tests passing, 44 skipped (architectural)
- [x] Fix M3.0 integration test compatibility
- [x] Update TEST_COVERAGE_IMPROVEMENT_PLAN.md
- [x] Phase 9: Architectural test cleanup
  - Deleted 11 deprecated tests (test_neo4j_analysis_writer_lifecycle.py)
  - Refactored 8 data integrity tests for projection pattern
  - Documented skipped tests with architectural rationale

**Results:**
- Tests: 554 → 1027 (+473 tests)
- Coverage: 66.8% → 90.1% (+23.3%)
- Skipped tests: Reduced from 44 architectural skips to 33 (11 deleted, 8 refactored)

**Completed:** 2026-01-25

---

### TC.10: Test Fixes + Infrastructure Integration ✅ COMPLETE

**Goal:** Fix remaining test issues and enable infrastructure-dependent integration tests

- [x] Fix projection handler unit tests (mock bug: Neo4jConnectionManager.get_session)
- [x] Fix pipeline conftest fixture (mock bug: classify_sentence vs analyze_sentence)
- [x] Add integration markers to live API tests (26 tests now properly marked)
- [x] Fix test_projection_rebuild.py infrastructure test
  - Made Neo4jMapStorage.initialize() a no-op for M3.0 single-writer architecture
  - Added `make test-rebuild` target
- [x] Fix InterviewCreatedData to include project_id in event data
  - Handlers now correctly read project_id from event payload
- [x] Fix test infrastructure EventStore connection (ESDB_CONNECTION_STRING override)
  - Tests now use EVENTSTORE_TEST_* variables following Neo4j pattern
- [x] Fix test_wait_for_ready_timeout flaky test
  - Now uses definitely-unavailable port instead of assuming production Neo4j absent

**Test Status:**
- Unit tests (`-m "not integration"`): 977 passed, 3 skipped
- Integration tests: 119 passed, 12 skipped (architectural)
- Full suite: All tests passing

**Completed:** 2026-01-28

---

### M3.1: Vector Search 📋 PLANNED

> Note: vector search builds on the Layer 1 fragment/utterance nodes (M4.1)
> and is expected to fold into Layer 2's embedding extractors (M4.2).

- [ ] Store sentence embeddings in Neo4j
- [ ] Semantic similarity search endpoints
- [ ] Vector-based clustering for topics
- [ ] Enhanced keyword/topic extraction
- [ ] Update 11 data integrity tests for eventual consistency (`test_neo4j_data_integrity.py`)

**Dependencies:** M3.0 complete (neo4j 6.x required)

---

### M3.2: AI Agent Upgrade 📋 PLANNED

- [ ] Upgrade openai 1.93.3 → 2.x
- [ ] Refactor `src/agents/` implementations
- [ ] Evaluate OpenAI Agents SDK
- [ ] Update anthropic SDK

**Dependencies:** M3.0 complete

---

### M3.3: Infrastructure Upgrades 📋 PLANNED

- [ ] Upgrade pytest 8.3.3 → 9.x
- [ ] Upgrade pytest-cov 6.0.0 → 7.x
- [ ] Upgrade redis 6.2.0 → 7.x
- [ ] Upgrade isort 5.13.2 → 7.x
- [ ] Re-establish performance baselines for M3.0 single-writer architecture

**Dependencies:** M3.0 complete

---

## Completed Milestones

<details>
<summary>M1: Core Plumbing ✅</summary>

- Event envelope and domain events
- EventStoreDB client and connection management
- Repository pattern for aggregates
- Interview and Sentence aggregates

</details>

<details>
<summary>M2.1: Command Layer ✅</summary>

- Command base classes and handlers
- Interview and Sentence commands
- Actor tracking and correlation IDs

</details>

<details>
<summary>M2.2: Dual-Write Integration ✅</summary>

- Event-first dual-write pattern
- Pipeline emits events before Neo4j writes
- Event failures abort operations (correct behavior)

</details>

<details>
<summary>M2.3-M2.5: Projection Infrastructure ✅</summary>

- Lane Manager with 12 configurable lanes
- Subscription Manager for ESDB persistent subscriptions
- Projection handlers for Interview and Sentence events
- Monitoring and health checks

</details>

<details>
<summary>M2.7: Testing & Validation ✅</summary>

- Integration tests for event-sourced processing
- E2E pipeline tests
- 72% code coverage

</details>

<details>
<summary>M2.8: Event-Sourced Architecture ✅</summary>

- Dynamic event versioning
- Edit protection across regeneration
- Cardinality enforcement at source
- Deprecation warnings for legacy paths
- **Completed:** 2026-01-17 (Production Ready)

</details>

<details>
<summary>M2.9: User Edit API ✅</summary>

- Edit sentence endpoint: `POST /edits/sentences/{id}/{index}/edit`
- Override analysis endpoint: `POST /edits/sentences/{id}/{index}/analysis/override`
- History endpoint: `GET /edits/sentences/{id}/{index}/history`
- Returns 202 Accepted with version
- 16 unit tests + 3 E2E tests
- **Completed:** 2026-01-18

</details>

<details>
<summary>TC: Test Coverage Improvement ✅</summary>

- Phase 0: Projection Infrastructure (150 tests, 78-100% coverage)
- Phase 1: Event Sourcing Foundation (81 tests, 89-95% coverage)
- Phase 2: Pipeline Tests (21 tests, 81.4% coverage)
- Phase 4: Parked Events Tests (20 tests, 100% coverage)
- Phase 5: Critical Gaps (27 tests) - run_projection_service 100%, sentence_analyzer 92%
- Phase 6: Domain Model (96 tests) - aggregates 99%, environment 95%
- Phase 7: Integration Test Infrastructure (10 tests) - ESDB environment detection, Makefile improvements
- Fixed M3.0 integration test compatibility
- Fixed production bugs in parked_events.py (invalid metadata reference)
- Fixed bug in environment.py (detect_environment f.read() called twice)
- Coverage: 66.8% → 90.1% (+23.3%)
- Tests: 554 → 1027 (+473 tests)
- **Completed:** 2026-01-25

</details>

---

## Dependency Upgrade Schedule

| Package | Current | Target | Milestone | Rationale |
|---------|---------|--------|-----------|-----------|
| neo4j | ~~5.28.1~~ 6.x | ✅ Done | **M3.0** | Vector types; single write path |
| openai | 1.93.3 | 2.x | **M3.2** | Agents SDK; function outputs |
| anthropic | >=0.39.0 | Latest | **M3.2** | Keep in sync |
| pytest | 8.3.3 | 9.x | M3.3 | Dev tooling |
| pytest-cov | 6.0.0 | 7.x | M3.3 | Dev tooling |
| redis | 6.2.0 | 7.x | M3.3 | Performance |
| isort | 5.13.2 | 7.x | M3.3 | Dev tooling |

---

## Technical Debt

### Deferred Backlog (consolidated 2026-07-10 — the "come back to it" list)

**M4.5-entry debt (from the M4.4 final review — burn down first, per the M4.3 pattern):**
- [ ] Markdown escaping for LLM text in rendered link titles / analysis table cells
      (newlines, `]`, `|` corrupt index links and tables)
- [ ] Entity/speaker slug collisions and empty slugs (uniquify with `-2` suffix;
      hash fallback for punctuation-only surfaces)
- [ ] Speaker rollup queries unbounded (push pagination/caps into Cypher)
- [ ] Exporting a never-applied lens passes the guard vacuously → give it a clear
      422 "lens never applied" instead of a near-empty bundle
- [ ] Exports route does sync file IO (rmtree/write/zip) on the event loop
- [ ] Offsets-invariant-with-front-matter test parametrized for the FLAT path
- [ ] Rollup substring name-filter behavioral test with non-empty data;
      renderer item_path/title derivation DRY-up; bundler rmtree→write window
      not atomic on OS/IO failure (consider temp-dir + rename staging)

**From M4.5a final review (2026-07-11):**
- [ ] Flip repository getter/factory call sites + test patch paths to
      `get_fragment_repository`/`create_fragment_repository` together when the
      deprecated aliases drop (post-M4.5)
- [x] Dual-label invariant assertion (every `:Sentence` is `:Fragment` and vice
      versa) → closed by `tests/integration/test_layer4_resolution_smoke.py`
      (M4.5b Task 12)
- [ ] `_link_text` backslash hardening in `src/export/renderer.py` (raw
      trailing `\` can still escape a link's closing bracket; escape
      backslashes first or rstrip after truncation)
- [ ] `FragmentRepository = SentenceRepository` class alias (+ edits.py local
      dependency naming) so M4.5b code doesn't annotate against the old class
      name
- [ ] Concurrent exports of same interview+lens share one fixed `.staging`
      path (`bundler.py`) — pre-existing race, not a regression
- [ ] Unit-test gap: projection-handler read-MATCH label shapes covered only
      by integration smokes
- [ ] Migration CLI uses deprecated `CALL {} IN TRANSACTIONS` syntax
      (non-fatal notice; CLI deleted at shim drop — fix only if it outlives
      that)
- [ ] Claim `(path, title)` derivation duplicated between render_bundle
      back-links and `_render_claim` (DRY)

**From M4.5b (2026-07-15):**
- [ ] Dockerized projection-service has never worked in this environment: its
      Neo4j target (dev `neo4j` compose service) has been stopped for months
      and until 278902f subscription groups were created without
      resolve_links (all $ce- events parked). resolve_links is fixed; still
      needed: point the service at a live Neo4j (or drop it from the default
      stack) and add a deployed-path smoke.

**From M4.5b final review (2026-07-15):**
- [ ] Engine-deferred merge pairs (auto-band, two existing canonicals) never
      appear on the worklist — surface them in compute_suggestions
      (`src/resolution/suggestions.py` discards the auto band)
- [ ] No human path to add an alias to a locked canonical: corrections API
      lacks add-alias, and skipped_locked surfaces are invisible on the
      worklist (engine counter only)
- [ ] `PersonLinkRemovedHandler` parks on duplicate delivery (removed==0
      guard) — NOTE: the guard is load-bearing for parked-event ordering
      (prevents a replayed parked link from resurrecting a removed edge);
      any fix must keep that property
- [ ] `suggestions.py` docstring overclaims: entity-merge rows are NOT
      actionable before the first engine run (confirm_entity_merge 404/409s
      until canonicals exist in the aggregate)
- [ ] Worklist GET degrades hard when the embedder is unavailable (quota) —
      500s the whole worklist; add graceful degradation (companion to the
      deferred embedding cache)
- [ ] `person_rows` lacks a `sp.merged_into IS NULL` filter; broader:
      SpeakerMerged x IDENTIFIED_AS/person-link interaction is unhandled
      (stale links after Layer-1 merges)
- [ ] Consider exempting PERSON-type surfaces from the plural fold in
      normalize_surface ("Jenkins"→"jenkin") — derivation is wire-adjacent
      once minted
- [ ] Carried task-review minors: `_cid_for_key` checks only the first
      surface of a group; no two-speakers-one-person render test; aliases
      frontmatter order asserted as set not list; T9 error-detail/tuple
      conventions

**Feature deferrals (each waits for a real need or its milestone):**
- [ ] Persona lens — second lens, proves zero-per-lens-code for real (YAML + prompts)
- [ ] Lens apply via ingest flag / API endpoint
- [ ] Same-version `--force` full re-extraction (needs an item-clearing event;
      CLI documents the limitation)
- [ ] GraphRAG retrieval (next after Layer 4 — borrow `neo4j-graphrag-python`
      retrievers only, never its construction pipeline)
- [ ] Corpus-level / multi-interview OKF bundles; incremental/diff exports;
      importing arbitrary OKF bundles (ingest reads front matter only)
- [ ] Layer-1 leftovers: `OVERLAPS` edges; `StitchCorrected` as a distinct event;
      LLM-based window reconciliation; live-LLM golden evaluation for prompt tuning

### Post-M3.0 Cleanup ✅ DONE
- [x] Remove 27 legacy tests (test_neo4j_analysis_writer_legacy.py)
- [x] Remove deprecated Neo4jMapStorage direct write code
- [x] Remove deprecated Neo4jAnalysisWriter direct write code
- [x] Remove graph_persistence tests (14 tests)
- [ ] Update 11 data integrity tests for eventual consistency (M3.1)
- [x] Delete dead `src/io/` (legacy storage protocols/writers) and its tests (M4.3)

### Future Improvements (Unprioritized)
- [ ] Rewrite fault-tolerance suite for EventStoreDB (legacy `test_neo4j_fault_tolerance.py` deleted in M4.3 — it imported the removed `src/io` modules; an ESDB-native rewrite is fresh work)
- [ ] Re-baseline performance benchmarks for single-writer (legacy `test_neo4j_performance_benchmarks.py` deleted in M4.3 for the same reason)
- [ ] Prometheus metrics exporter (currently in-memory)
- [ ] WebSocket for real-time Neo4j updates
- [ ] CLI tool for replaying parked events
- [ ] Event schema versioning and migration
- [ ] OpenTelemetry distributed tracing
- [ ] Neo4j query optimization for bulk operations
- [ ] Circuit breaker for Neo4j connection failures
- [ ] Event archival/compaction strategy

---

## Skipped Tests Inventory

**Total skipped: 3** (3 unit + 0 integration)

### Unit (3 skipped)

| Test | Reason | Milestone |
|------|--------|-----------|
| `test_helpers.py` (2 tests) | `openpyxl` not installed | N/A — optional dependency |
| `test_text_processing.py` (1 test) | Import-time exception logging untestable without reload | N/A — test limitation |

### Integration (0 skipped)

The long-skipped legacy suites (`test_neo4j_fault_tolerance.py`, `test_neo4j_performance_benchmarks.py`) were deleted in M4.3 — they imported the removed `src/io` modules. ESDB-native rewrites are tracked under Future Improvements.

---

## Architecture Overview

```
Current State (M3.0 - Single-Writer) ✅
───────────────────────────────────────
User Upload / Edit API
    ↓
Pipeline / Command Handlers
    └──→ EventStoreDB (events only) ← Source of Truth

EventStoreDB
    ↓
Projection Service (12 lanes)
    ↓
Neo4j (sole writer, materialized view)
```

---

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-07-10 | M4.4 (Layer 5) complete: read-side exporter over Neo4j | One reader (`src/export/reader.py`) serves both the OKF bundle and the query endpoints — no second query path to keep in sync |
| 2026-07-10 | Front matter round-trips through the Interview aggregate, not a new event | Ingest front matter → aggregate metadata (`InterviewCreated.metadata["front_matter"]`) → rendered `interview.md` header; no projection, no new event types |
| 2026-07-10 | M4.3 (Layer 3) complete: generic lens engine, Approach A | A lens is one YAML + prompts; three generic events + one generic handler set serve every lens — zero per-lens code |
| 2026-07-10 | M4.2-exit debt burned down before lens work | Per-model embedding isolation, resilient failover construction, dead `src/io` deleted, provenance/edge minors |
| 2026-07-09 | Dynamic node labels validated at emit AND sanitized at handler | LLM output never reaches Cypher as a label; `projects_to` keys are the only legal labels |
| 2026-07-06 | M4.2 (Layer 2) complete; legacy pipeline retired | Registry is the sole enrichment path; ~200 legacy tests removed |
| 2026-07-06 | Generalized provider strategy (interface + config chain) | Anthropic Haiku primary → Claude Code harness → OpenAI; embeddings config-pinned, model-tagged, never silently switched |
| 2026-07-06 | Embeddings ride as events with inline base64 vectors | Preserves single-writer + replay purity; direct Neo4j writes rejected |
| 2026-07-06 | Bumped anthropic model claude-3-haiku-20240307 → claude-haiku-4-5-20251001 | Old model retired (not_found_error); verified live |
| 2026-07-06 | Vector-index DDL runs in its own auto-commit session | Neo4j forbids schema DDL inside a data-write transaction |
| 2026-07-04 | M4.1 (Layer 1) complete: speakers, utterances, offset-grounded map | Spec: docs/superpowers/specs/2026-07-04-mine-layers-design.md |
| 2026-07-04 | Stitching is an overlay, never a rewrite | Interview must be viewable as-spoken; interpretation is additive + correctable |
| 2026-07-04 | Speaker inference reconciles windows by deterministic overlap voting | LLM-based reconciliation deferred until golden evaluation demands it |
| 2026-01-31 | Unskipped 4 integration tests | Fixed Neo4j connection (NEO4J_URI override), .env loading, M3.0 test update |
| 2026-01-31 | Fixed performance test Neo4j connection | Override NEO4J_URI in setup_test_environment so handlers use test DB |
| 2026-01-31 | Added .env loading to root conftest | API keys from .env now available for all tests without manual export |
| 2026-01-31 | Updated test_concurrent_file_processing for M3.0 | Added projection handler replay before Neo4j verification |
| 2026-01-28 | TC.10 complete | All infrastructure tests passing, event env vars fixed |
| 2026-01-28 | Added project_id to InterviewCreatedData | Handler needs project_id in event data, not just envelope |
| 2026-01-28 | Made Neo4jMapStorage.initialize() no-op | M3.0 single-writer: projection service is sole Neo4j writer |
| 2026-01-28 | Use EVENTSTORE_TEST_* vars in test fixtures | Follow Neo4j pattern; ignore production .env values |
| 2026-01-26 | TC.10 phase for infrastructure tests | Projection rebuild test needs dedicated Make target and infra |
| 2026-01-26 | Fixed mock bugs in unit tests | Neo4j session mock used wrong pattern; pipeline fixture mocked wrong method |
| 2026-01-26 | Added integration markers to 26 live API tests | Proper skipping when running unit tests without API keys |
| 2026-01-25 | Integration test infrastructure complete | ESDB environment detection, Make targets for test orchestration |
| 2026-01-24 | Test coverage 90.1% achieved | 1000 tests, added Phase 5-6, fixed environment.py bug |
| 2026-01-24 | Test coverage milestone extended | 86.0% coverage achieved, 943 tests, fixed parked_events.py bugs |
| 2026-01-19 | Test coverage milestone complete | 84.2% coverage achieved (target 80%), 902 tests |
| 2026-01-18 | M3.0 complete: single-writer architecture | Removed 41 legacy tests, all direct Neo4j writes eliminated |
| 2026-01-18 | Bundle neo4j 6.x with M3.0 | Single write path simplifies migration |
| 2026-01-18 | Separate openai 2.x to M3.2 | Orthogonal to event-sourcing; needs dedicated focus |
| 2026-01-18 | Defer pytest/redis to M3.3 | No immediate benefit; low priority |
| 2026-01-18 | Plan vector search for M3.1 | Requires neo4j 6.x vector types |

---

## How to Update This Document

1. **When starting a milestone:** Change status from 📋 to ⏳, update "Current Phase"
2. **When completing tasks:** Check off items in the milestone checklist
3. **When completing a milestone:** Change status to ✅, move to "Completed" section
4. **When making decisions:** Add entry to Decision Log

**Document Owner:** Engineering Team
**Review Cadence:** Update after each milestone completion
