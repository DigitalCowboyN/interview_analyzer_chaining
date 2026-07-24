# M4.9 — Projection Ordering & Recovery Hardening (design)

**Status:** approved by owner 2026-07-24 (brainstorm dialogue)
**Parent:** surfaced by M5.1 Task 6 (the live ui-smoke exposed it); confirmed
PRE-EXISTING on `main`. Its own milestone off `main`; M5.1 rebases onto it.

## Problem

Projection of a freshly-ingested interview is **flaky and lossy**: fragments
go orphaned (missing `(Interview)-[:HAS_SENTENCE]->(Fragment)`) and speakers
go null, at random, depending on subscription-delivery timing. The workbench
transcript therefore renders empty or partial. Confirmed reproducible on
`main` (no M5.x code). Three adversarial analyses (`.superpowers/sdd/
proj-analysis-{A,B,C}-*.md`) triangulated the cause.

### Root cause (three stacked defects)

1. **Order discarded.** Events for one interview arrive via THREE independent
   ESDB persistent subscriptions (`$ce-Interview`, `$ce-Sentence`,
   `$ce-Project`). The `LaneManager` routes to a lane keyed by `interview_id`
   but processes in **arrival order**, not commit order. ESDB stamps every
   event with a globally-monotonic `commit_position`, but it is thrown away
   in `store.py::_recorded_event_to_envelope`. Dependent handlers `MATCH`
   their referenced nodes and, if the referent isn't projected yet, either
   **silently no-op** (`SentenceCreated`, `SpeakerCreated` — checkpoint as
   success → permanent orphan) or raise → retry → park.
2. **Parking loses data.** `parked_events.park_event` calls
   `store.append_events` with no `expected_version` → defaults to
   `StreamState.NO_STREAM` → only the FIRST park to each `parked-<type>`
   stream succeeds; every subsequent park raises `WrongCurrentVersion` and
   the event is lost. No re-drive of parked events exists.
3. **Inconsistent guards.** The `_raise_if_no_writes` "referent-not-ready"
   signal is present in speaker/utterance handlers but ABSENT in
   `SentenceCreated`/`SpeakerCreated` — the very handlers behind the dominant
   orphaned-fragment symptom.

### Verified linchpin: commit-order == causal-order

The ingestion producer commits referents before dependents (each `save`
assigns monotonic `commit_position`):
- t1 `InterviewCreated`+`SpeakerCreated` (Interview stream, orchestrator
  `save`@124) → t2 `SentenceCreated`+`SpeakerAttributed` (Sentence streams,
  `save`@259) → t3 `UtteranceIdentified` (Interview stream, `save`@135).
- Interview(t1) < HAS_SENTENCE(t2); Speaker(t1) < SpeakerAttributed(t2);
  Fragment(t2) < UtteranceIdentified(t3).

Therefore **processing in `commit_position` order is a complete fix** for the
race — no stub nodes, no timing guesses. (Stub-projection was rejected: two
of three analyses judged it "invisible corruption" — it surfaces blank
Interview/Speaker nodes to readers as real data and breaks the
"node exists ⇒ event projected" invariant.)

## Architecture (layered fix; M4.7 delivery path untouched)

Keep the three subscriptions and their hard-won M4.7 mechanics (sync
iteration via `asyncio.to_thread`, `resolve_links=True`, `ack_id` acks,
checkpointing) **unchanged**. Four additive changes:

### 1. Propagate `commit_position`

Populate `commit_position` on the delivered envelope in
`store.py::_recorded_event_to_envelope` (it is available on the ESDB
`RecordedEvent` at read/subscribe time). This is a **read-side, transient**
field — never stored, no event-payload change. **Wire format stays frozen**
(event type names, `aggregate_type` "Sentence", `Sentence-{uuid}` streams,
`sentence_id`, handler file names all unchanged).

### 2. Per-lane reorder buffer (the root fix)

Each interview lane releases its buffered events in **`commit_position`
order**, gated by a **shared global low-watermark** = the minimum, across the
three subscriptions, of each subscription's highest contiguously-delivered
`commit_position`. A buffered event is released when EITHER:
- `commit_position <= global_watermark` (safe — no earlier event can still
  arrive), OR
- it has been held longer than a bounded `max_hold` (covers an **idle
  subscription** whose watermark would otherwise never advance and stall
  release; also bounds worst-case latency).

Because commit-order == causal-order, in-order release means every `MATCH`
handler finds its referent already projected. On full replay/rebuild the
buffer orders identically → **deterministic projection**.

Parallelism preserved: lanes remain independent; only the shared watermark is
cross-cutting. Values (`max_hold`, watermark source) injectable; defaults
pinned by tests.

### 3. Park `StreamState.ANY`

`park_event` appends with `expected_version=StreamState.ANY` — parked streams
are append-only logs with no concurrency invariant. Stops the active
data-loss; every park succeeds.

### 4. Consistent not-ready guards + deferral + re-drive

- Make `SentenceCreated`/`SpeakerCreated` raise a typed
  `ReferentNotReadyError` on a missing referent (matching the speaker/
  utterance handlers) — no more silent-success orphans. Defense-in-depth
  behind the buffer.
- `ReferentNotReadyError` is retriable; only genuinely-stuck events reach
  parking (now working).
- A **CLI** `python -m src.projections.redrive` replays parked events (idempotent,
  version-guarded). Not a standing worker — with ordering in place parked
  events are rare; a background loop is YAGNI.

## Testing

- **Unit:** the reorder buffer in isolation — out-of-order in → in-order out;
  watermark advances and releases; `max_hold` flush when a subscription is
  idle; `commit_position` populated on the envelope. Handler unit tests for
  the new `ReferentNotReadyError` guards. Park append uses `ANY` (fake-store
  assertion). Re-drive CLI replays a parked event.
- **Live acceptance (the milestone's proof):** an env-gated projection smoke
  that seeds an interview through the real dockerized dev stack and asserts
  **all** fragments AND speakers project reliably across repeated runs — the
  exact flakiness that exists today. Mirrors `deployed-smoke`'s gating/pyenv
  conventions; its own make target.

## Non-goals

- Single-`$all` subscription redesign (rejected: largest regression risk to
  the M4.7 delivery path).
- A standing background re-drive worker (CLI suffices).
- Any change to the frozen wire format or the M4.7 ack/checkpoint mechanics.
- Handler logic beyond the not-ready guards (no stub-projection).
- M5.1 work (resumes after this merges).
