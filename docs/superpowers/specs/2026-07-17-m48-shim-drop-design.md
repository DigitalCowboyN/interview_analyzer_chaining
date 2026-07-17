# M4.8 — `:Sentence` Shim Drop (design)

**Status:** draft 2026-07-17 (awaiting owner review)
**Parent:** M4.5a (`docs/superpowers/specs/2026-07-10-layer4-schema-v2-design.md`)
deferred this: "dropping the `:Sentence` shim label and deprecated code aliases;
re-targeting vector index DDL to `:Fragment` (rides the shim drop)."

## Goal

Retire the M4.5a transition shim completely: `:Fragment` becomes the only label
on fragment nodes, the deprecated code aliases disappear, and the graph carries
no dead vocabulary into the UI arc. Pure code-surface + graph-label work.

## Frozen (untouched — wire format)

- Event type names (`SentenceCreated`, …), `AggregateType.SENTENCE`'s value
  `"Sentence"`, and `Sentence-{uuid}` stream names.
- Handler/module FILE names (`sentence_handlers.py`, `sentence_events.py`, …)
  — they mirror the frozen event vocabulary; renaming them is churn with no
  value.
- Payload fields (`sentence_id` property stays `sentence_id`).

## Changes

### 1. Write path (projection handler)

`SentenceCreatedHandler`'s create query anchor flips:
`MERGE (s:Sentence {sentence_id: …}) SET s:Fragment` →
`MERGE (s:Fragment {sentence_id: …})` (SET dropped). Deploy-order safe:
existing nodes are dual-labeled, so a `:Fragment` MERGE matches them; new
nodes are single-labeled from the moment the code deploys.

### 2. Code aliases (drop, and finish the rename)

- `src/events/aggregates.py`: delete `Sentence = Fragment`.
- `src/events/repository.py`: rename class `SentenceRepository` →
  `FragmentRepository` (the class itself still carries the old name — M4.5a
  only planned this); delete the `create_sentence_repository` /
  `get_sentence_repository` aliases.
- Flip every consumer call site (`src/ingestion/orchestrator.py`,
  `src/lens/engine.py`, `src/enrichment/orchestrator.py`,
  `src/commands/handlers.py`, `src/api/routers/edits.py`,
  `src/api/routers/speakers.py`) AND every test that patches the old dotted
  paths (~12 files) **in the same commit** — the alias and its patch paths
  must flip together or the suite breaks; this coupling is why the drop was
  deferred as one atomic milestone.

### 3. Vector indexes (the one subtlety)

Fragment-embedding vector indexes are created with label `Sentence` but NAMED
`fragment_embedding_{model}` (`embedding_handlers._ensure_vector_index`).
Stripping `:Sentence` labels would silently EMPTY those indexes, and the lazy
ensure would no-op on the name collision (`IF NOT EXISTS`) — breaking the ask
vector channel. Therefore:

- `_ensure_vector_index` DDL label flips `"Sentence"` → `"Fragment"`.
- The migration (below) drops each existing `:Sentence`-label
  `fragment_embedding_*` index and recreates it same-named on `:Fragment`
  (same dimensions/options); Neo4j repopulates it from the node properties.
  Utterance indexes are label-`:Utterance` and unaffected.

### 4. One-shot migration CLI (replaces `migrate_fragment_label`)

Delete `src/projections/migrate_fragment_label.py` (its job is done and its
deprecated `CALL {} IN TRANSACTIONS` syntax dies with it — closes that backlog
item). New `python -m src.projections.migrate_shim_drop`, idempotent:

1. Recreate each `fragment_embedding_*` vector index on `:Fragment` (discover
   existing ones via `SHOW INDEXES`, drop the `:Sentence`-label version,
   create the `:Fragment` version, `db.awaitIndexes()`).
2. Strip the `:Sentence` label from all nodes (batched `REMOVE s:Sentence`).
3. Drop the three shim btree indexes (`sentence_sentence_id`,
   `sentence_lookup`, `sentence_sequence`) with `IF EXISTS` semantics.
4. Print a JSON summary (`{"relabeled": n, "vector_indexes": [...],
   "btree_dropped": [...]}`); second run reports zeros.

**Deploy order (documented in README + schema doc):** deploy code first, then
run the migration. (Both orders are safe — dual labels make the new MERGE
match old nodes, and old indexes keep working until stripped — but one order
is pinned to keep runbooks deterministic.)

### 5. Schema DDL + docs

- `SCHEMA_DDL` drops its three `:Sentence` shim entries (26 → 23 statements);
  the shim-window comment goes with them.
- `docs/architecture/database-schema.md`: shim notes removed; `:Fragment` is
  simply the label.
- ROADMAP: backlog items closed (shim drop, alias flips, vector retarget,
  migration-CLI deletion/deprecated-syntax item); M4.8 milestone section.

## Testing

- The flip itself is the test: with aliases deleted, any straggler import or
  patch path fails loudly at collection. Full unit gate at HEAD.
- Dual-label invariant assertions in smokes retire to single-label
  (`:Fragment`-only, and explicitly assert NO `:Sentence` label on new nodes).
- Migration proven live against the test Neo4j: seed dual-labeled data +
  a `:Sentence`-label vector index (the pre-drop world), run the CLI, assert
  labels stripped / index retargeted / ask vector query still answers; run
  twice, second run reports zeros.
- All layer smokes + e2e + ask smoke green; one `make deployed-smoke` run at
  the end (the write anchor changed — prove the deployed path still delivers).

## Non-goals

- Renaming frozen wire surfaces or handler file names (above).
- `edits.py`'s local `EditSentenceRequest`/route naming — API paths are a
  public surface; renaming them is a UI-arc decision, not this milestone's.
- Any new feature work; this closes debt only.
