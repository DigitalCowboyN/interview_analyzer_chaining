# C2 (embedding capability) + C3 (connect every test) — design

**Status:** approved by owner 2026-08-07 (brainstorm dialogue).
**Two independent cleanups, brainstormed and shipping together.** C2 restructures the
capability tree so *embedding* is its own capability (not a flavor of provider-strategy).
C3 connects the test suite's currently-"unmapped" tests to what they verify — because
every test tests *something*; an unconnected test is a knowledge gap, not noise.

They touch different domains (capabilities vs tests) and are independent; one branch, two
clearly-separated parts.

---

## Part C2 — split "embedding" from "provider"

### The problem
`pinned-embeddings` is filed as a **variant of `provider-strategy-and-focused-calls`**. That
conflates two distinct concerns: *embedding* (a capability — turn fragments into vectors)
and *the provider* (a separate capability that supplies chat **and** embedding calls, and
could be swapped or self-hosted). The embedder lives in `src/enrichment/embedder.py` and
the vector index in `src/projections`; the provider is `agents`.

### The fix
- **`provider-strategy-and-focused-calls` stays = the provider capability** (config-driven,
  provider-agnostic LLM access with failover). `chat-failover` stays its variant. No change.
- **New capability `embed-fragments`** — a **primary, enabling, product** capability:
  > *Turn each fragment into a vector embedding so fragments can be found by meaning, not
  > just by keyword.*
  `implemented_by: [enrichment, projections]` (the embedder + the vector-index projection).
- **`pinned-embeddings` moves out of provider-strategy** → becomes a **child of
  `embed-fragments`** (`kind: variant → child`, `parent: provider-strategy-and-focused-calls
  → embed-fragments`; `implemented_by` unchanged). Pinning to one model is an *embedding*
  concern (comparability), not a provider-strategy one.
- **No new capability→capability edge.** "Embedding depends on the provider" is an
  implementation fact already captured in the code dependency graph (`enrichment` →
  `agents`); capabilities are intent, and the model deliberately keeps how-dependencies in
  code. The restructure is purely about placing the *intent* correctly.

### Footprint
- Create `docs/capabilities/embed-fragments.md`; edit `docs/capabilities/pinned-embeddings.md`
  (kind + parent). Regenerate `docs/capabilities/index.md` + `docs/graph/{index,graph}.md`
  (new node + moved `child_of` edge + new `implements` edges). `capability-check` clean
  (new primary classified; `enrichment`/`projections` now claimed by `embed-fragments`).
- No new ADR — this is a corpus correction within ADR-0019's capabilities-as-intent model.

---

## Part C3 — connect every test to what it verifies

### The problem
`testmap-check` flags **29 "unmapped" tests** (no derived code target, no `# verifies:`
marker). Verified: **no test imports another test module**, so there is no test→test
composition to capture. The 29 break down as:
- **4 `api_surface.*`** — they test `tools.api` (the api-surface catalog tool); the path
  segment `api_surface` just doesn't match `api`/`tools.api_surface`. → **resolution fix.**
- **19 `integration/*`** + **6 root-level** (`test_config`, `test_celery_app`, `test_tasks`,
  `test_anthropic_agent_response`, `test_openai_agent_response`, `test_prompts`) —
  cross-cutting; the graph can't derive their target. → **authored `# verifies:` markers.**

### The fix
1. **Resolution alias (mechanical).** In `tools/testmap/reader.py`, add a small
   `_TARGET_ALIASES = {"api_surface": "tools.api"}` that `_target` consults before giving up.
   Resolves the 4 `api_surface` tests to `code:tools.api`. (Extensible: one dict entry per
   known dir↔unit naming mismatch.)
2. **Marker pass (authored).** Add a module-level `# verifies: <domain>:<id>` to each
   remaining test, targeting the most honest node it exercises end-to-end. A marker may
   target **code** (the unit a smoke test drives), a **use-case**, or a **capability**:
   - layer smoke tests → the layer's code unit: `test_layer1_projection_smoke` →
     `code:projections`; `test_layer2_enrichment_smoke` → `code:enrichment`;
     `test_layer3_lens_smoke` → `code:lens`; `test_layer4_resolution_smoke` →
     `code:resolution`; `test_layer5_export_smoke` → `code:export`.
   - `test_ask_smoke` → `use-cases:get-a-grounded-answer-from-my-corpus`.
   - provider/agent tests (`test_anthropic_api_messages`, `test_openai_api_responses`,
     `test_multi_provider_api_live`, `test_prompt_validation_live`,
     `test_anthropic_agent_response`, `test_openai_agent_response`) → `code:agents`.
   - `test_api_calls` → `code:api`; `test_neo4j_*` → `code:persistence`;
     `test_projection_*` / `test_idempotency` / `test_migrate_shim_drop_live` →
     `code:projections`; `test_live_feed_smoke` → `code:ui`; `test_prompts` →
     `code:tools.prompts`.
   - **Every marker is authored by reading the test and confirming it genuinely exercises
     that target** — no guessing. The exact target for each is decided in implementation
     (the list above is the starting map, verified per file).
3. **The honest residual (a signal, not noise).** A few root tests exercise infrastructure
   with **no node to point at** — `test_config` (`src/config.py`, a top-level module, not a
   code-map unit), `test_celery_app` / `test_tasks` (the Celery task queue). Per the owner's
   principle these are not "noise" — they **reveal a gap**: the graph has no node for
   configuration or the task queue. Implementation surfaces each such test explicitly and
   the owner decides per case: (a) add the missing node (e.g. an operations capability for
   the task queue, or document the module as a code unit) and mark to it, or (b) consciously
   accept it as a documented residual. The aim is zero unmapped; where a test can't reach
   zero, the reason is recorded, not shrugged off.

### Footprint
- Edit `tools/testmap/reader.py` (`_TARGET_ALIASES` + `_target`); add markers to ~25 test
  files (comment-only, no logic touched); regenerate `docs/tests/index.md` +
  `docs/graph/{index,graph}.md` (new `verifies` edges). `graph-check` clean (every marker
  target resolves). `testmap-check` unmapped count drops from 29 toward 0 (residual
  documented). No new ADR — extends the tests domain within ADR-0022.

---

## Non-goals (both parts)
- **A capability→capability "depends_on" edge** (C2) — the dependency lives in code; not modeled as intent.
- **Name-based derivation heuristics for integration tests** (C3) — rejected in brainstorm; authored markers are honest and don't drift.
- **Test→test composition edges** (C3) — none exist in the suite.
- **Forcing a node for genuine infra tests** (C3) — surfaced for an owner decision, not fabricated.
- **Blocking** on any finding.

## Testing
- **C2 unit:** `capability-check` clean after the restructure (new primary classified;
  `pinned-embeddings` resolves its new parent `embed-fragments`; coverage of
  `enrichment`/`projections`). Graph: `embed-fragments` appears as a Capability node; the
  `child_of` edge points `pinned-embeddings → embed-fragments`; `implements` edges to
  enrichment/projections resolve.
- **C3 unit (`tests/testmap/`):** `_target` resolves an `api_surface`-segment path to
  `tools.api` via the alias; a fixture test file carrying a `# verifies: code:enrichment`
  marker harvests that edge; existing derivation unaffected.
- **Smoke:** regenerate all affected indexes; `make capability-check` / `make graph-check`
  / `make code-check` / `make testmap-check` clean-or-expected (testmap unmapped count near
  zero with any residual documented); `make health`; `python -m pytest tests/testmap
  tests/capability tests/graph`.

## Knowledge-graph check
Reviewed against `docs/index.md` on 2026-08-07.

| domain | touched? | consulted / reconciled | notes |
| --- | --- | --- | --- |
| capabilities | yes (C2) | new `embed-fragments`; reparent `pinned-embeddings`; regen index | corpus correction, ADR-0019 model |
| tests | yes (C3) | `_TARGET_ALIASES` in reader; ~25 authored markers; regen index | ADR-0022 model |
| graph | yes | regen: new Capability node + `child_of`/`implements` + `verifies` edges; endpoints checked | derived |
| code | yes (read-only) | marker targets resolve against code units; C2 implemented_by uses documented units | possible new node if owner documents config/celery |
| use-cases | yes (read-only) | some markers target use-cases (e.g. get-a-grounded-answer) | no node change |
| adr | no | no new ADR (C2 within 0019, C3 within 0022) | — |
| cli / knowledge / glossary / api / prompts / graph-queries | no | — | unaffected |

**Verdict:** reconciled — capabilities (C2) + tests (C3) are the subjects; graph regenerated;
code/use-cases consulted read-only; any infra-test residual surfaced for an owner decision.
