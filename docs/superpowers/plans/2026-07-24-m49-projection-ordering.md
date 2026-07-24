# M4.9 — Projection Ordering & Recovery Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make projection of an ingested interview reliable and order-independent — fragments and speakers always project — by processing each interview's events in ESDB `commit_position` order, plus fixing the data-losing park bug and the silent-no-op handler guards.

**Architecture:** Keep the three M4.7 category subscriptions and their ack/checkpoint mechanics untouched. Propagate `commit_position` (already on every delivered event, currently discarded) onto the envelope; add a shared watermark tracker + a per-interview-lane reorder buffer that releases events in `commit_position` order; fix park to `StreamState.ANY`; make `SentenceCreated`/`SpeakerCreated` raise `ReferentNotReadyError` on a missing referent; add a re-drive CLI.

**Tech Stack:** Python 3.10, esdbclient persistent subscriptions, asyncio, Neo4j projection handlers.

**Spec:** `docs/superpowers/specs/2026-07-24-m49-projection-ordering-design.md` (binding).

## Global Constraints

- Wire format FROZEN: no event-type/`aggregate_type`/stream-name/`sentence_id`/handler-file-name changes. `commit_position` is a READ-SIDE transient envelope field populated at delivery — never serialized into stored event metadata.
- Do NOT alter the M4.7 delivery mechanics: sync esdbclient subscription iteration via `asyncio.to_thread`, `resolve_links=True` on `$ce-` streams, acks via `event.ack_id` (not `event.id`), existing checkpoint callback wiring. Reordering happens AFTER delivery, inside the lane; acks are invoked when a buffered event is finally released+processed (deferred, not dropped).
- Commit-order == causal-order is verified (referents commit before dependents); processing in `commit_position` order therefore makes every `MATCH` handler find its referent. The `ReferentNotReadyError` guard + retry + fixed park are the backstop for any residual out-of-order release.
- Determinism: the reorder buffer releases strictly by ascending `commit_position`; a full replay yields the same projection.
- All timing values (`max_hold_ms`, watermark) injectable; defaults pinned by tests.
- Tests: unit via `./scripts/test.sh <paths> -q --no-cov` (agents run targeted files; full suite is CONTROLLER-ONLY). The live projection smoke is env-gated, its own make target, NOT in default suites.
- Reader anchor unchanged: `src/ui/reader.py::transcript_line_rows` matches `(Interview)-[:HAS_SENTENCE]->(Fragment)`; success = it returns every seeded line with its speaker.

---

### Task 1: Propagate `commit_position` onto the envelope

**Files:**
- Modify: `src/events/envelope.py` (add field), `src/events/store.py::_recorded_event_to_envelope` (~line 310, populate it)
- Test: `tests/events/` (envelope + store conversion tests — find the existing store/envelope test file and mirror its style)

**Interfaces (produced — Tasks 4/5 consume):**
- `EventEnvelope.commit_position: Optional[int] = Field(default=None, ...)` — read-side global order key; None for freshly-created (un-delivered) events.
- `_recorded_event_to_envelope` sets `commit_position=recorded_event.commit_position` (the `RecordedEvent` field — verified to exist).

Behavior spec (binding): a freshly constructed envelope (producer side) has `commit_position=None`. An envelope built from a delivered `RecordedEvent` carries that event's `commit_position` (an int). The field is NOT written into the ESDB metadata dict in `append_events` (grep-confirm the metadata dict is unchanged) — it is transient/read-side only.

- [ ] Step 1: Failing tests: envelope accepts/defaults `commit_position=None`; `_recorded_event_to_envelope` (fake `RecordedEvent` with `commit_position=12345`) yields envelope with `commit_position==12345`; `append_events` metadata dict does NOT include `commit_position` (round-trip: a re-read event's stored metadata has no such key).
- [ ] Step 2: Run; confirm failures.
- [ ] Step 3: Implement (one field + one assignment; leave `append_events` metadata untouched).
- [ ] Step 4: `./scripts/test.sh <the two test files> -q --no-cov` green.
- [ ] Step 5: Commit `feat(events): carry commit_position on delivered envelopes (read-side)`.

---

### Task 2: Park with `StreamState.ANY` (stop data loss)

**Files:**
- Modify: `src/projections/parked_events.py` (the `park_event` `append_events` call, ~line 147)
- Test: existing parked-events test file under `tests/projections/` (find it; mirror style)

**Interfaces:** none new; `park_event` now passes `expected_version=StreamState.ANY` to `event_store.append_events`.

Behavior spec (binding): parking N events for the SAME `aggregate_type` (same `parked-<type>` stream) all succeed — the second and later parks must NOT raise `WrongCurrentVersion`/`ConcurrencyError`. `StreamState.ANY` is imported from esdbclient (see how `src/events/store.py` imports `StreamState`).

- [ ] Step 1: Failing test: park two events of the same aggregate_type via a real-ish or faked event store; assert BOTH append calls are issued with `expected_version=StreamState.ANY` (and, against a fake store that models NO_STREAM semantics, that the second does not raise). Include a test that the parked payload/tags are unchanged.
- [ ] Step 2: Run; confirm failure (today the second park raises / is swallowed).
- [ ] Step 3: Implement (one-line `expected_version=StreamState.ANY`).
- [ ] Step 4: `./scripts/test.sh <parked-events test file> -q --no-cov` green.
- [ ] Step 5: Commit `fix(projections): park events append-only (StreamState.ANY) — stop parked-event loss`.

---

### Task 3: Consistent `ReferentNotReadyError` guards

**Files:**
- Modify: `src/projections/handlers/sentence_handlers.py` (SentenceCreated — the HAS_SENTENCE/Interview MATCH), `src/projections/handlers/speaker_handlers.py` (SpeakerCreated, and confirm the shared error type) — READ these first; the speaker/utterance handlers already have a `_raise_if_no_writes`-style guard to mirror.
- Create (if not present): a typed `ReferentNotReadyError` in the handlers' shared module (`base_handler.py` or wherever the existing not-ready raise lives — find it first and reuse; do NOT invent a second error type).
- Test: `tests/projections/test_projection_handlers_unit.py`

**Interfaces (produced):**
- `ReferentNotReadyError(Exception)` — the existing/typed signal that a handler's referenced node is not yet projected (reuse the existing one the speaker/utterance handlers raise; only formalize a name if it's currently a bare `ValueError`). Base-handler retry-to-park already treats this as retriable — verify and preserve.

Behavior spec (binding): when `SentenceCreated`'s handler cannot find the `Interview` node (so the `HAS_SENTENCE` edge would not be created), it RAISES `ReferentNotReadyError` instead of silently completing. Same for `SpeakerCreated` if it depends on a not-yet-projected referent (verify whether it actually does; if SpeakerCreated has no cross-stream referent, note that in the report and leave it — do not manufacture a dependency). The Fragment node itself must STILL be created (its own data) — only the cross-referent EDGE creation is what gates the raise; confirm the current handler order so the node write isn't lost when the edge can't be made. If node-write and edge-write are one query, split so the node persists and only the missing edge triggers the retry (this matters: the reorder buffer makes this rare, but the backstop must not drop the fragment).

- [ ] Step 1: Failing tests (fake session): SentenceCreated with NO Interview node present → raises `ReferentNotReadyError` AND the Fragment node write was still issued; SentenceCreated WITH Interview present → creates HAS_SENTENCE, no raise. Mirror for SpeakerCreated only if it has a real referent dependency.
- [ ] Step 2: Run; confirm failures (today it silently no-ops).
- [ ] Step 3: Implement the guard(s), preserving the node write.
- [ ] Step 4: `./scripts/test.sh tests/projections/test_projection_handlers_unit.py -q --no-cov` green.
- [ ] Step 5: Commit `fix(projections): raise ReferentNotReadyError on missing referent — no silent orphan projections`.

---

### Task 4: Watermark tracker + per-lane reorder buffer (the core)

**Files:**
- Create: `src/projections/reorder_buffer.py` (WatermarkTracker + the buffer logic), `tests/projections/test_reorder_buffer.py`
- Modify: `src/projections/subscription_manager.py` (~line 165-200: record watermark before routing), `src/projections/lane_manager.py` (Lane holds a reorder buffer; release in commit_position order)

**Interfaces (produced):**
```python
class WatermarkTracker:
    def record(self, subscription_name: str, commit_position: int) -> None   # updates that sub's high-water
    def low_watermark(self) -> Optional[int]   # min high-water across REGISTERED subs; None until all registered subs have delivered ≥1
    def register(self, subscription_name: str) -> None
```
- Lane gains a reorder buffer (min-heap / sorted by `commit_position`). Its process loop: pull `(event, checkpoint_callback)` from the queue into the buffer; then DRAIN — release (process + ack) buffered entries in ascending `commit_position` while the lowest satisfies `commit_position <= watermark.low_watermark()` OR it has been buffered ≥ `max_hold_ms`. Injectable `max_hold_ms` (default e.g. 250) and the shared `WatermarkTracker`.

Behavior spec (binding):
- Events entering a lane out of `commit_position` order are PROCESSED in ascending `commit_position` order (the whole point). Test: enqueue events with commit_position [3,1,2] for one interview → handlers invoked in order [1,2,3].
- Release gate 1 (watermark): an event releases once `low_watermark >= its commit_position` (safe: no earlier event can still arrive). Test with a tracker whose watermark advances.
- Release gate 2 (max_hold): if the watermark is stuck (an idle subscription), an event still releases after `max_hold_ms` so the lane never stalls. Test with a never-advancing watermark + injected clock/short hold → event eventually releases.
- Ack timing: `checkpoint_callback` fires only when the buffered event is RELEASED+processed (deferred), never on buffering. A buffered-but-unreleased event is NOT acked. Test asserts the callback is not called until release.
- Determinism: given the same set of events, release order is by `commit_position` regardless of arrival order.
- subscription_manager records `watermark.record(self.name, recorded_event.commit_position)` for every delivered event before `route_event` (so the shared watermark reflects all three subscriptions); each subscription `register`s itself at startup.

Do NOT block the event loop: max_hold uses a bounded `asyncio.wait_for`/timeout on the drain, not a busy-wait. Preserve the existing per-lane in-order guarantee within one commit_position sequence.

- [ ] Step 1: Failing unit tests for `reorder_buffer.py` (pure, no ESDB/Neo4j): WatermarkTracker min-across-subs + None-until-all-registered; buffer releases [3,1,2]→[1,2,3]; watermark-gated release; max_hold flush on stuck watermark (injected clock); deferred-ack (callback not called until release).
- [ ] Step 2: Run; confirm failures. Implement `reorder_buffer.py`.
- [ ] Step 3: Failing integration-of-units tests: Lane with the buffer processes out-of-order enqueues in commit_position order and defers acks; subscription_manager records watermark before routing (fake lane_manager captures calls).
- [ ] Step 4: Implement Lane + subscription_manager wiring. `./scripts/test.sh tests/projections/test_reorder_buffer.py tests/projections/<lane + subscription tests> -q --no-cov` green, no event-loop-block warnings.
- [ ] Step 5: Commit `feat(projections): commit_position reorder buffer + watermark — ordered per-interview projection`.

---

### Task 5: Re-drive CLI for parked events

**Files:**
- Create: `src/projections/redrive.py` (`python -m src.projections.redrive`), `tests/projections/test_redrive.py`

**Interfaces:** a CLI entrypoint that reads parked events (via the existing `parked_events.get_parked_events`) for given aggregate type(s) and re-dispatches them through the real handler registry (idempotent; handlers MERGE, so replay is safe). Prints a JSON summary (counts re-driven / still-failing).

Behavior spec (binding): re-driving a parked event whose referent NOW exists succeeds and applies the projection; one whose referent still missing raises `ReferentNotReadyError` again and is reported as still-parked (not crashed). Idempotent: re-driving an already-applied event is a no-op (MERGE semantics). Reuse `create_handler_registry()` (bootstrap) — do not hand-roll dispatch.

- [ ] Step 1: Failing tests: redrive applies a parked event when its referent exists; reports still-failing when not; idempotent double-redrive. (Fake/real handler registry per the existing projection test idiom.)
- [ ] Step 2: Run; confirm failures. Implement.
- [ ] Step 3: `./scripts/test.sh tests/projections/test_redrive.py -q --no-cov` green.
- [ ] Step 4: Commit `feat(projections): redrive CLI — replay parked events after referents land`.

---

### Task 6: Live projection smoke, docs, gates; then final review + finish

**Files:**
- Create: `tests/integration/test_projection_ordering_smoke.py`
- Modify: `Makefile` (`projection-smoke` target mirroring `deployed-smoke`), `docs/ROADMAP.md`, `docs/` schema/projection notes if present
- Test: the smoke itself

Behavior spec (binding):
- Env-gated `PROJECTION_SMOKE=1` (mirror `DEPLOYED_SMOKE` skip idiom), own make target `projection-smoke` (compose up dev stack + pyenv pin + ESDB_CONNECTION_STRING=localhost override, exactly like `deployed-smoke`/`live-feed-smoke`). Seed an interview through the real dockerized stack via `IngestionOrchestrator`; poll the DEV Neo4j until projection settles; assert EVERY seeded fragment is present with its `HAS_SENTENCE` edge AND its speaker (`SPOKEN_BY`) — i.e. `transcript_line_rows` returns all lines with non-null speakers. Run the seed+assert loop REPEATEDLY (e.g. 5 interviews) to prove the flakiness is gone (today this fails intermittently).
- ROADMAP: add an M4.9 completed section (Spec/Plan links, one line per task, Completed date) per the M4.7/M4.8 idiom; note it unblocks M5.1 T6; Current Phase → back to `M5.1 (Live workbench — resume T6)`.
- Gates (FOREGROUND): backend targeted `./scripts/test.sh tests/events/ tests/projections/ -q --no-cov`; `make projection-smoke` witnessed passing across repeated runs. Full Python suite controller-only.

- [ ] Step 1: Write the smoke + make target; run `make projection-smoke`; WITNESS repeated clean runs. If it still flakes after genuine debugging, report DONE_WITH_CONCERNS/BLOCKED — do NOT commit a smoke you never saw reliably green.
- [ ] Step 2: Docs (ROADMAP + any projection/schema notes).
- [ ] Step 3: Gates green.
- [ ] Step 4: Commit `docs+test: M4.9 complete — projection-ordering smoke, ROADMAP`.

---

## Self-review notes (writing-plans checklist)

- **Spec coverage:** commit_position propagation → T1; park ANY → T2; consistent guards → T3; reorder buffer + watermark (the root fix) → T4; re-drive CLI → T5; live smoke + docs → T6. Non-goals respected (no $all redesign, no standing worker, no wire-format change, no stub-projection).
- **Type consistency:** `commit_position` (T1) consumed by T4/T5; `WatermarkTracker`/`ReferentNotReadyError` names stable across T3/T4/T5; `max_hold_ms` default pinned by T4 tests.
- **Judgment points (deliberate):** exact heap vs sorted-list in the buffer, the watermark's idle-detection details, and smoke poll timings are implementer-authored against the binding behavior specs.
