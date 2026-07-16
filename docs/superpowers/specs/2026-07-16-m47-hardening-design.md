# M4.7 — Hardening & Operational Readiness (design)

**Status:** draft 2026-07-16 (awaiting owner review)
**Sources:** ROADMAP Deferred Backlog (owner-approved candidates + judgment picks,
2026-07-16); M4.5b/M4.5c/M4.6 final-review findings.

## Goal

Close the operational gaps the M4.x milestones deferred: make the deployed
projection path actually work, give the graph a programmatic schema (indexes and
constraints applied by code, not documentation), harden the ask and resolution
surfaces against the failure modes their reviews identified, and prove the
zero-per-lens-code claim with a second lens. No new wire format: one workstream
reuses an existing frozen event (`EntityAliasAdded`); everything else is
read-side, config, infra, or prompt/YAML assets.

## Scope decision (owner, 2026-07-16)

All four shortlisted candidates are in — deployed-path fix, overlay indexes,
persona lens, ask-channel hardening — plus judgment-picked backlog items that
fit the hardening theme. Priority order below is execution order.

## Workstreams (priority order)

### W1 — Programmatic schema: `ensure_schema`

The only DDL the code applies today is per-model vector indexes
(`embedding_handlers._ensure_vector_index`) and the M4.6 fulltext index. The
canonical index/constraint list lives only in
`docs/architecture/database-schema.md` ("Schema Creation Script") — it has never
been executed programmatically, and it is missing every Layer-4 overlay key the
handlers MERGE by (`:Segment(segment_id)`, `:CanonicalEntity(canonical_id)`,
`:Person(person_id)`), plus the other handler MERGE keys that grew since M2
(`:Interview(interview_id)`, `:Utterance`, `:Entity`, `:Speaker`, `:Project`).

- New `src/projections/schema.py`: one authoritative, idempotent DDL list
  (`CREATE CONSTRAINT/INDEX ... IF NOT EXISTS`) covering **every MERGE key used
  by a projection handler** (the plan enumerates them by reading the handlers)
  plus the existing documented script entries. Plain async function taking a
  session — fake-session testable with query-text pins, the reader-module idiom.
- Applied at projection-service startup (`run_projection_service.main()` before
  `service.start()`) and exposed as `python -m src.projections.ensure_schema`
  for one-off/deploy use.
- `docs/architecture/database-schema.md`'s script section becomes a pointer to
  the module (single source of truth in code).
- The `:Sentence` shim indexes stay until the shim drops (see Non-goals).

### W2 — Deployed projection path (fix + prove)

The dockerized projection service has never worked here: its Neo4j target (the
dev `neo4j` compose service) has been stopped for months, `neo4j` has no
healthcheck (commented out), and `projection-service` gates only on
`service_started`. Right now the container is up and consuming events with a
dead write target (lane-queue-depth warnings during today's smoke run) — worst
of both worlds: it drains subscriptions it can't project.

- Restore a real healthcheck on the `neo4j` service (cypher-shell/HTTP probe,
  the eventstore healthcheck pattern) and gate `projection-service`, `app`, and
  `worker` on `neo4j: service_healthy`.
- Projection-service startup runs W1's `ensure_schema` against its target, so a
  fresh dev Neo4j comes up with the schema.
- Fail fast: if Neo4j is unreachable at startup, the service exits non-zero
  (compose `restart: unless-stopped` handles retry) instead of consuming with a
  dead target.
- **Deployed-path smoke**: a docker-gated integration test (own pytest marker +
  make target, NOT in the default suites) that brings up / verifies the real
  compose stack, writes one interview's events through the API-side command
  path, and asserts the node materializes in the **dev** Neo4j via bolt :7687 —
  the first end-to-end proof of the deployed path.

### W3 — Ask-channel hardening (M4.6 final-review findings)

- **Project-aware over-fetch**: vector and fulltext queries run a global top-k
  then filter to the project, so recall decays as other projects grow. Fix with
  a fixed over-fetch multiplier (`CHANNEL_K * OVERFETCH_FACTOR` candidates
  fetched, post-filtered to `CHANNEL_K`) — simplest correct step; index-side
  scoping stays backlogged.
- **Per-index vector degradation**: split the single try/except so the fragment
  and utterance index queries fail independently; both failing keeps
  `flags["vector_unavailable"]`, one failing sets a partial flag and the other
  channel leg still contributes.
- **Fulltext ensure-once**: `ensure_fulltext_index` (incl. `db.awaitIndexes`)
  memoized per process instead of running on every ask. (W1 also applies it at
  service startup.)
- **Unicode-safe sanitizer**: `sanitize_fulltext_query` currently strips
  non-ASCII word characters, so accented names never reach the fulltext channel;
  make it Unicode-aware (escape/strip Lucene specials rather than whitelisting
  ASCII).
- **CLI validation + coverage**: `--top-k` validated (`>= 1`; today `-1`
  silently slices `scored[:-1]`); CLI tests including the non-zero-exit
  (`SynthesisUnavailable`) leg.
- **Interview title**: the reader already returns it in context rows unused —
  surface it in the context-block header line (cheap prompt-quality win).
- **Test strength**: engine tests assert the distinct
  ValidationError-vs-generic-Exception synthesis error messages.
- **Spec amendment**: M4.6 spec's "three channels, concurrent" wording corrected
  to sequential-by-design (embed once, then per-channel queries).

### W4 — Resolution worklist operational fixes (M4.5b final-review findings)

- **Embedder-unavailable degradation**: worklist GET currently 500s when the
  embedder fails (quota). Catch the failure in `compute_suggestions`, return
  the deterministic (exact-key + participant) suggestions with a
  `flags`-style marker — the ask surface's degradation doctrine applied here.
- **Stale-speaker filter**: `person_rows` gains `sp.merged_into IS NULL` so
  Layer-1 speaker merges stop surfacing stale links. (The broader
  SpeakerMerged × IDENTIFIED_AS reconciliation stays backlogged.)
- **Surface engine-deferred merge pairs**: the auto-band pairs where BOTH
  surfaces already belong to existing canonicals are engine-skipped and
  currently invisible; `compute_suggestions` emits them as actionable
  entity-merge rows.
- **Add-alias correction**: `POST /resolution/{project_id}/entities/{canonical_id}/aliases`
  records the existing frozen `EntityAliasAdded` event through the aggregate —
  the missing human path to extend a locked canonical. Follows the corrections
  router's 202/404/409 conventions.
- **Docstring fix**: `suggestions.py` overclaims that entity-merge rows are
  actionable before the first engine run; correct it.

### W5 — Persona lens (zero-per-lens-code proof)

Second lens, YAML + prompts only: `lenses/persona.yaml` +
`prompts/lens_persona.yaml`. Speaker-scoped persona dimensions (traits, goals,
pain points, notable quotes — exact dimensions pinned in the plan following the
`meeting_minutes` YAML shape). Deliverable is the **proof**: zero new Python;
the existing generic engine, events, projection, export, and API serve it
unchanged. Verified by extending the lens smoke (or an e2e-smoke leg) to apply
`persona` with canned extractions through the real registry.

### W6 — Minor consistency sweep

- `GET /interviews/{id}/segments` 404s for unknown interviews (today: 200 +
  empty list, while the DELETE leg 404s).
- `_link_text` backslash hardening in `src/export/renderer.py` (raw trailing
  `\` can escape a link's closing bracket).
- Concurrent exports of the same interview+lens share one fixed `.staging`
  path — make the staging dir unique per export.
- e2e smoke's canned embedder gets its "values unused, only counts asserted"
  clarifying comment.

## Error handling

Follows the established degradation doctrine: external-dependency failure
degrades a channel/surface with an explicit flag, never a 500 (W3 vector/
fulltext, W4 worklist); infra misconfiguration fails fast and loud (W2 startup);
validation errors reject at the boundary (W3 CLI). No silent fallbacks.

## Testing

- Unit: `schema.py` query-text pins (fake session); ask reader/engine/CLI new
  legs; suggestions degradation + deferred-pair rows; add-alias router tests;
  renderer/staging/segments-404 legs. All in the default `./scripts/test.sh`
  gate.
- Integration: deployed-path smoke is docker-gated (own marker + make target,
  excluded from default suites — it manages real containers); persona-lens leg
  rides the existing in-process smoke pattern; all existing smokes stay green.
- No live-LLM dependencies anywhere.

## Non-goals (stay backlogged)

- **`:Sentence` shim drop** + deprecated alias flips + vector-index retarget +
  migration-CLI deletion — a deliberate, mechanical migration; natural M4.8
  candidate now that M4.5 is complete.
- Text2Cypher ask channel; index-side project scoping for vector search.
- `PersonLinkRemovedHandler` duplicate-delivery guard rework (load-bearing for
  parked-event ordering).
- `EnrichmentResult.flags` scoping/rename (waits for a second document-scope
  extractor).
- `normalize_surface` PERSON plural-fold exemption (wire-adjacent once minted).
- SpeakerMerged × IDENTIFIED_AS full reconciliation (W4 ships the read-side
  filter only).
- Corpus-level bundles, Layer-1 leftovers, Prometheus/OTel/etc. (Future
  Improvements list).
