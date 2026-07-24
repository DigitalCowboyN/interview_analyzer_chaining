# M5.1 — Live Workbench Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The workbench transcript and interview list update live — line items appear/resequence and corrections show up without manual refresh — via a backend SSE bridge over ESDB catch-up subscriptions, after two M5.0 hygiene riders land.

**Architecture:** No projection-service changes. `src/ui/notifications.py` hosts a NotificationHub (in-process pub/sub) and EsdbWatcher (three ephemeral catch-up subscriptions on `$ce-Interview`/`$ce-Sentence`/`$ce-Project`, from end of stream, lazy lifecycle); one SSE route on the existing `/ui` router streams thin surface-tagged notifications. Frontend: one `useLiveInvalidation` hook maps surface tags to existing `queryKeys` and invalidates with debounce + trailing re-invalidate; intent settle polling is untouched.

**Tech Stack:** FastAPI `StreamingResponse` (SSE), esdbclient catch-up subscriptions (sync iterator via `asyncio.to_thread`), TanStack Query invalidation, native browser `EventSource`.

**Spec:** `docs/superpowers/specs/2026-07-24-m51-live-workbench-design.md` (binding).

## Global Constraints

- Wire format FROZEN: no new event types, no payload changes, no stream-name changes. This milestone only READS events.
- Loose coupling: browser never sees event types/stream names/ESDB concepts — the SSE payload is exactly `{"surface": "transcript"|"interviews"|"project", "interview_id"?: str, "project_id"?: str}` plus the distinguished `{"surface": "resync"}` message. Components never touch `EventSource` directly; layering api → hooks → components → app holds.
- Intent settle mechanism (bounded confirm-poll in `frontend/src/hooks/mutations.ts`) is NOT modified by the live layer (Task 1 hygiene edits excepted).
- SSE endpoint degrades, never 500s for watcher trouble; UI without SSE behaves exactly like M5.0; never a broken page.
- Debounce policy (binding): first notification for a query key → invalidate immediately; further notifications coalesce; ALWAYS schedule a trailing re-invalidate 2000ms after the most recent notification (timer resets per notification). Values injectable for tests; defaults pinned by a test.
- Heartbeat: SSE comment line every 15s. Headers: `Cache-Control: no-cache`, `X-Accel-Buffering: no`, media type `text/event-stream`.
- Frontend gates (in `frontend/`): `npx vitest run`, `npx tsc --noEmit`, `npm run lint`. Backend targeted tests per task via `./scripts/test.sh <paths> -q --no-cov`. Full Python suite is CONTROLLER-ONLY (never run it in a task).
- Sync esdbclient iteration MUST run via `asyncio.to_thread` (event-loop starvation lesson, M4.7); category streams need `resolve_links=True`.

---

### Task 1: Rider A — intent-pattern hygiene

**Files:**
- Create: `frontend/src/components/NoticeText.tsx`
- Modify: `frontend/src/components/LineDetailPanel.tsx` (remove `CorrectionNoticeBanner` ~line 25 AND byte-identical `PersonNoticeBanner` ~line 310), `frontend/src/components/PersonPicker.tsx` (~line 133), `frontend/src/components/WorklistRows.tsx` (~line 31), `frontend/src/hooks/mutations.ts`, `src/commands/handlers.py`
- Test: existing colocated frontend tests; `tests/commands/test_command_handlers_unit.py`

**Interfaces:**
- Produces: `NoticeText` — props `{ notice: { kind: string; message: string } | null; className?: string }`, rendering the same alert/notice markup the four near-copies render today (preserve `role="alert"` semantics and kind→color mapping exactly; allow the small `mt-1`/`mt-2` variance via `className`).
- Produces: `encodePath(template: string, ...ids: string[])` OR equivalent helper in `frontend/src/api/client.ts` used by all mutation paths in `mutations.ts` (every raw `${id}` interpolation at ~lines 261, 289, 322, 355, 388, 430, 463, 526, 569 goes through `encodeURIComponent`).
- `_handle_override` in `src/commands/handlers.py` gains the same `except ValueError → raise CommandValidationError` wrap around ONLY the aggregate call, exactly mirroring `_handle_edit`'s existing wrap.

Behavior spec (binding): five hygiene items, no behavior changes beyond them —
1. ONE shared notice renderer replaces all four copies; all existing tests still pass (update imports/test ids only as needed).
2. Absence predicates in `mutations.ts` confirm-polling (~lines 74–82 and the worklist predicates) treat `getQueryData(...) === undefined` as NOT settled (return false) instead of a false settle.
3. All mutation URL paths percent-encode interpolated ids.
4. When a text-edit intent settles, additionally invalidate `queryKeys.sentenceHistory(interviewId, sequenceOrder)` so an open history panel refreshes.
5. `_handle_override` ValueError→CommandValidationError parity + a test asserting a same-value override raises `CommandValidationError` (not `CommandExecutionError`), colocated with the existing `_handle_edit` test.

- [ ] Step 1: Write/adjust failing tests first: NoticeText render test (kinds → classes), absence-predicate `undefined` guard test (fake QueryClient returning undefined mid-poll → poll continues), encoding test (id with `/` produces `%2F` in fetch URL), sentenceHistory invalidation test (settled edit → both transcript AND history keys invalidated), backend parity test.
- [ ] Step 2: Run them; confirm each fails for the expected reason.
- [ ] Step 3: Implement all five items.
- [ ] Step 4: Frontend gates + `./scripts/test.sh tests/commands/test_command_handlers_unit.py -q --no-cov` — all green, output pristine.
- [ ] Step 5: Commit `refactor: intent-pattern hygiene — shared notice, poll guards, url encoding, history invalidation, override parity`.

---

### Task 2: Rider B — interview-metadata projection

**Files:**
- Modify: `src/projections/handlers/interview_handlers.py` (the `metadata_diff` branch, currently a literal `pass` at ~line 120), `src/ui/reader.py::interview_metadata` (~line 83)
- Test: existing handler test file for interview handlers (find it under `tests/projections/`); `tests/ui/` reader tests

**Interfaces:**
- Interview node gains STRING property `metadata_json` (JSON object serialized with `json.dumps(..., sort_keys=True)`). No other new properties.
- `interview_metadata` reader now returns `metadata` as a parsed dict: `json.loads` of `metadata_json` when present, `{}` when absent or unparsable (log-and-empty, never raise). HTTP contract to the frontend unchanged (it already expects a metadata object).

Behavior spec (binding): when the event payload contains `metadata_diff` (a dict), the handler reads the interview's current `metadata_json` (may be NULL), merges the diff into the parsed dict (top-level key merge: diff keys overwrite, `None` values delete the key), and writes back the serialized result in the same handler execution. Read-merge-write is safe — lanes serialize per interview (document this in a comment). Wire format untouched. No migration (spec: old interviews backfill via replay if ever needed).

- [ ] Step 1: Failing handler tests (fake session): (a) diff onto empty node sets full dict; (b) diff merges over existing keys; (c) `None` value deletes a key; (d) query-text pin includes `metadata_json`. Failing reader tests: parses JSON; absent → `{}`; malformed JSON → `{}` without raising.
- [ ] Step 2: Run; confirm failures.
- [ ] Step 3: Implement handler merge + reader parse.
- [ ] Step 4: `./scripts/test.sh <the two test files> -q --no-cov` green, pristine.
- [ ] Step 5: Commit `feat(projections): project interview front-matter — metadata_diff merge to metadata_json, reader parse`.

---

### Task 3: NotificationHub + scope mapping

**Files:**
- Create: `src/ui/notifications.py`, `tests/ui/test_notifications.py`

**Interfaces (produced — Tasks 4–6 depend on these exact names):**
```python
@dataclass(frozen=True)
class Notification:
    surface: str                     # "transcript" | "interviews" | "project" | "resync"
    interview_id: Optional[str] = None
    project_id: Optional[str] = None

class NotificationHub:
    def subscribe(self, interview_id: Optional[str], project_id: Optional[str]) -> "Subscription": ...
        # Subscription has .queue (asyncio.Queue[Notification]) and .close()
    def publish(self, notification: Notification) -> None   # fan out to matching subscribers
    def broadcast_resync(self) -> None                      # Notification(surface="resync") to ALL
    @property
    def subscriber_count(self) -> int

def scope_notifications(stream_name: str, payload: dict) -> list[Notification]: ...
```

Behavior spec (binding) for `scope_notifications` (pure function, the loose-coupling translation layer):
- `Sentence-*` streams → `[Notification("transcript", interview_id=payload["interview_id"])]`; missing `interview_id` → `[]` (log-and-skip is the caller's job; here just return empty).
- `Interview-{iid}` streams → transcript notification with `iid` parsed from the stream name; ADDITIONALLY an `interviews` notification when the payload carries `project_id` (e.g. InterviewCreated — verified in spec).
- `Project-*` streams → `[Notification("project", project_id=payload["project_id"])]`; missing → `[]`.
- Any other stream → `[]`.

Matching in `publish`: a subscriber receives a notification when (`surface == "transcript"` and ids match its `interview_id`) or (`surface in ("interviews","project")` and ids match its `project_id`). `resync` goes to everyone. Full queues: drop-oldest (bounded queue, maxsize 64) — a browser that can't keep up self-heals via the next notification; never block the watcher.

- [ ] Step 1: Failing tests: scope mapping for all four stream cases + missing-key cases; hub fan-out (two subscribers different interviews — each gets only its own); resync reaches all; close() unregisters (subscriber_count drops); full-queue drop-oldest doesn't raise.
- [ ] Step 2: Run; confirm failures.
- [ ] Step 3: Implement (pure Python, no ESDB imports in this task).
- [ ] Step 4: `./scripts/test.sh tests/ui/test_notifications.py -q --no-cov` green.
- [ ] Step 5: Commit `feat(ui): notification hub + surface scope mapping`.

---

### Task 4: EsdbWatcher + SSE endpoint

**Files:**
- Modify: `src/ui/notifications.py` (add watcher + module accessor), `src/api/routers/ui.py` (add SSE route)
- Test: `tests/ui/test_notifications.py` (watcher), `tests/api/test_ui_router.py` (SSE route)

**Interfaces (produced):**
```python
class EsdbWatcher:
    def __init__(self, hub: NotificationHub, event_store: Optional[EventStoreClient] = None,
                 backoff_seconds: Sequence[float] = (1, 2, 5, 10)): ...
    async def ensure_started(self) -> None    # idempotent; spawns watch tasks on first call
    async def stop(self) -> None

def get_live_feed() -> tuple[NotificationHub, EsdbWatcher]   # module-level lazy singleton pair
```
- SSE route: `GET /ui/streams/events?interview_id=…&project_id=…` (both optional, at least one required else 422). Response: `text/event-stream`, headers `Cache-Control: no-cache`, `X-Accel-Buffering: no`. Each notification → `data: {json}\n\n` (the Notification's non-None fields). Heartbeat `: keep-alive\n\n` every 15s of quiet. On client disconnect: subscription closed; when `subscriber_count` hits 0 the route calls `watcher.stop()` (lazy lifecycle both directions).

Behavior spec (binding) for the watcher:
- Three catch-up subscriptions — `$ce-Interview`, `$ce-Sentence`, `$ce-Project` — via the project's `EventStoreClient` (`src/events/store.py`) using `client.subscribe_to_stream(stream, from_end=True, resolve_links=True)`.
- Sync iterator consumed with the M4.7 sentinel pattern: `await asyncio.to_thread(next, iterator, sentinel)` per pull — study `src/projections/subscription_manager.py::_run_subscription` FIRST and mirror its idiom; never iterate a sync esdbclient subscription directly on the event loop.
- Per event: JSON-decode `event.data` (malformed → log at debug, skip, keep looping), call `scope_notifications(event.stream_name, payload)`, `hub.publish` each. Use the RESOLVED event's stream name (link-resolved semantics as in M4.7).
- Subscription death (any exception): close iterator, backoff through `backoff_seconds` (repeat last value), resubscribe `from_end=True`, then `hub.broadcast_resync()` — clients refetch what they missed while the watcher was down.
- `ensure_started` is idempotent under concurrency (guard with an asyncio.Lock); `stop` cancels tasks and closes iterators cleanly.

- [ ] Step 1: Failing watcher tests with a FAKE client (queue-driven iterator; no real ESDB): events flow → hub publishes mapped notifications; malformed event skipped, loop continues; iterator raises → after fake backoff a new subscription is created AND resync broadcast; ensure_started twice → one set of tasks; stop cancels.
- [ ] Step 2: Failing route tests (httpx `stream()` against the app with a stubbed hub/watcher injected): first data line arrives and parses to the contract shape; both params absent → 422; disconnect → subscription closed and `watcher.stop()` called at zero subscribers; heartbeat emitted when notification-quiet (inject a short heartbeat interval — make it a route/module constant with an injectable override, defaults pinned by test).
- [ ] Step 3: Run; confirm failures. Implement.
- [ ] Step 4: `./scripts/test.sh tests/ui/test_notifications.py tests/api/test_ui_router.py -q --no-cov` green, pristine.
- [ ] Step 5: Commit `feat(ui): SSE live feed — esdb catch-up watcher, lazy lifecycle, resync on reconnect`.

---

### Task 5: Frontend — useLiveInvalidation + LiveIndicator + page wiring

**Files:**
- Create: `frontend/src/hooks/useLiveInvalidation.ts`, `frontend/src/components/LiveIndicator.tsx`, colocated `__tests__`
- Modify: `frontend/src/app/workbench/[projectId]/[interviewId]/page.tsx`, `frontend/src/app/workbench/[projectId]/page.tsx`

**Interfaces:**
- Consumes: SSE endpoint from Task 4 (`/api/ui/streams/events?...` through the Next proxy — note `EventSource` cannot set headers; the stream needs no identity); `queryKeys` builders from `frontend/src/hooks/queryKeys.ts`.
- Produces:
```ts
type LiveStatus = "live" | "offline" | "idle";
function useLiveInvalidation(scopes: {
  interviewId?: string; projectId?: string;
}, timing?: { coalesceMs?: number; trailingMs?: number }): LiveStatus
// defaults coalesceMs=500, trailingMs=2000 — pinned by a test
```
- `LiveIndicator` — props `{ status: LiveStatus }`; subtle dot + accessible label ("live updates on/off"); rendered by the two workbench pages near their breadcrumbs.

Behavior spec (binding):
- One `EventSource` per mounted hook; URL built from the provided scopes; closed on unmount. `onopen → "live"`, `onerror → "offline"` (browser auto-retries; status flips back on reopen). No scopes → `"idle"`, no connection.
- Notification handling: `transcript` → invalidate `queryKeys.transcript(interviewId)`; `interviews` → invalidate `queryKeys.interviews(projectId)`; `project` → invalidate `queryKeys.transcript(interviewId)` when an interviewId scope is present (person links affect the open transcript) and `queryKeys.persons(projectId)`; `resync` → invalidate ALL keys this hook watches.
- Debounce policy per Global Constraints: immediate first invalidate per key, coalesce within `coalesceMs`, trailing re-invalidate `trailingMs` after the most recent notification (timer resets). Implemented per query key.
- Transcript page: `useLiveInvalidation({ interviewId, projectId })`; interview-list page: `useLiveInvalidation({ projectId })`. Rendering the indicator is the ONLY visual change; existing tests keep passing (mock EventSource globally in test setup if needed).

- [ ] Step 1: Failing Vitest tests with a mocked EventSource class (capture instances; fire `onopen`/`onmessage`/`onerror`): scope→key mapping per surface; debounce with fake timers (burst of 5 → 1 immediate + 1 trailing per key; timer reset verified); resync invalidates all watched; status transitions; unmount closes; defaults pinned.
- [ ] Step 2: Run; confirm failures. Implement hook + indicator + page wiring.
- [ ] Step 3: Frontend gates green, output pristine (no act() warnings).
- [ ] Step 4: Commit `feat(ui): live invalidation — SSE hook, debounce+trailing refetch, live indicator`.

---

### Task 6: Live integration test, ui-smoke extension, docs, gates

**Files:**
- Create: `tests/integration/test_live_feed_smoke.py`
- Modify: `frontend/e2e/smoke.spec.ts`, `README.md` (frontend section: live updates note), `docs/ROADMAP.md` (M5.1 section per M5.0 idiom; row → ✅; add an `M5.1b` planned row "Gallery liveness — fast-follow scopes on the live pipeline"; Current Phase → `M5.1b (Gallery liveness)`)
- Test: the new files themselves

Behavior spec (binding):
- Integration test (deployed-smoke family, env-gated `LIVE_FEED_SMOKE=1`, NOT in default suites, own make target `live-feed-smoke` mirroring `deployed-smoke`'s structure and pyenv pin): start the app in-process (httpx ASGI streaming), open the SSE stream for a fresh interview id, ingest a transcript via `IngestionOrchestrator` with that project, assert a `{"surface": "transcript"}` notification for that interview arrives within a generous timeout; assert heartbeat framing parses. Uses shared ESDB per `tests/integration/test_deployed_projection_smoke.py` conventions (ESDB_CONNECTION_STRING localhost override documented in the header).
- ui-smoke extension: with the transcript page OPEN, append/ingest a new line server-side and assert it appears WITHOUT any user action (the M5.1 money shot). Reuse the existing seeding helper; keep the existing edit-settle journey passing. Generous timeout in the deployed-smoke spirit.
- Docs: README gains a short "Live updates" paragraph (SSE through the proxy, indicator meaning, degrade-to-M5.0 behavior); ROADMAP M5.1 section one line per task + Completed date; stats line left for the controller.
- Gates (FOREGROUND): backend targeted `./scripts/test.sh tests/ui/ tests/api/test_ui_router.py tests/commands/test_command_handlers_unit.py -q --no-cov`; frontend `npm run lint && npm run typecheck && npm test`; `npm run build`; `make ui-smoke` (with the new assertion) witnessed passing; `make live-feed-smoke` witnessed passing. Full Python suite controller-only.

- [ ] Step 1: Write the integration test + make target; run `make live-feed-smoke`; witness pass.
- [ ] Step 2: Extend `smoke.spec.ts`; run `make ui-smoke`; witness pass.
- [ ] Step 3: Docs edits.
- [ ] Step 4: All gates green.
- [ ] Step 5: Commit `docs+test: M5.1 complete — live-feed smoke, ui-smoke live assertion, README, ROADMAP`.

---

## Self-review notes (writing-plans checklist)

- **Spec coverage:** riders → T1/T2; hub+mapping → T3; watcher+SSE+lazy lifecycle+resync → T4; hook+debounce+indicator+page wiring → T5; live proof+docs → T6. Non-goals respected (no gallery scopes, no WebSocket, no replay, no lifespan/startup coupling — lazy start only).
- **Type consistency:** `Notification`/`NotificationHub`/`scope_notifications`/`get_live_feed` names used identically in T3→T4; `useLiveInvalidation`/`LiveStatus` in T5→T6; timing defaults (500/2000/15s) consistent everywhere.
- **Judgment points (deliberate):** exact SSE framing helpers, indicator styling, and fake-EventSource test utilities are implementer-authored against the binding behavior specs.
