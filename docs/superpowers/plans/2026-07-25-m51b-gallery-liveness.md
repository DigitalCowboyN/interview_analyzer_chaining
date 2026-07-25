# M5.1b — Gallery Liveness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the gallery surfaces (persons grid, person detail, personas grid, persona detail, worklist) update live off the existing M5.1 SSE feed, with one additive backend change so persona-lens re-runs also reach the gallery.

**Architecture:** Reuse the M5.1 pipeline unchanged (EsdbWatcher → NotificationHub → `scope_notifications` → SSE route → `useLiveInvalidation`). Persons/worklist/persona-composition are already live-capable because `Project-*` events carry `project_id` and already emit the `project` surface — they only need frontend wiring. Persona-lens *content* liveness needs the Interview aggregate to carry `project_id` and stamp it onto lens events, which the watcher decodes from event metadata and `scope_notifications` routes to the gallery via the same `project` surface. The browser contract (`{surface, interview_id?, project_id?}`) does not change.

**Tech Stack:** Python 3.10 (pyenv `~/.pyenv/versions/3.10.7/bin/python`), pytest, FastAPI/esdbclient/EventStoreDB; Next.js App Router + React + TanStack Query + Vitest/Testing Library.

**Spec:** `docs/superpowers/specs/2026-07-25-m51b-gallery-liveness-design.md`

## Global Constraints

- **Wire format is FROZEN.** Event type names, `AggregateType.SENTENCE` value `"Sentence"`, `Sentence-{uuid}` stream names, `sentence_id`, and handler file names never change. `project_id` is an existing OPTIONAL envelope/metadata field (already serialized into ESDB event metadata by `src/events/store.py` and already present on all `Project-*` events) — stamping it onto lens events is ADDITIVE, not a frozen-format change.
- **No backfill.** Only *newly-emitted* lens events carry `project_id`. Historical lens events do not; liveness only needs new events. Do not migrate or rewrite stored events.
- **Coarse by design.** Any `Interview-*` stream event with a resolvable `project_id` nudges the gallery via the single `project` surface. Per-event-type precision is a deliberate non-goal — the `useLiveInvalidation` debounce (500ms coalesce) + trailing re-invalidate (2000ms) absorb enrichment bursts.
- **Browser contract unchanged.** The SSE message shape stays `{surface, interview_id?, project_id?}` with `surface ∈ {transcript, interviews, project, resync}`. No new surface tags (finer per-surface tags are a non-goal). `personId` refines only which client query keys the `project` surface invalidates; it is NOT sent to the server and NOT added to the SSE URL.
- **Backend tests need env:** `set -a; source .env; set +a` before pytest. Unit tests (Tasks 1–2) are env-light but run under the same harness. The live smoke (Task 5) is env-gated and run via `make live-feed-smoke`, which sets `LIVE_FEED_SMOKE=1` and `ESDB_CONNECTION_STRING=esdb://localhost:2113?tls=false` (the committed `.env` points ESDB at the docker-internal `eventstore` hostname, unresolvable from a host-run process).
- **Run Python via the pyenv interpreter:** `~/.pyenv/versions/3.10.7/bin/python -m pytest ...`.
- **Frontend commands run from `frontend/`:** `npm run test`, `npm run lint`, `npm run build`.

---

### Task 1: Interview aggregate carries and stamps `project_id`

**Files:**
- Modify: `src/events/aggregates.py` (`Interview.__init__` ~line 173; `Interview._apply_interview_created` ~line 241; `Interview.apply_lens` ~line 695; `Interview.record_lens_extraction` ~line 713; `Interview.override_lens_extraction` ~line 764)
- Test: `tests/events/test_aggregates_unit.py`

**Interfaces:**
- Consumes: `Interview.create(..., project_id=...)` already writes `project_id` into BOTH `event.data["project_id"]` and the envelope `event.project_id` (verified: `aggregates.py:425-433`). On rebuild from history, `EventEnvelope.project_id` is repopulated from stored metadata (`src/events/store.py:341`), so `event.project_id` is available both at create-time apply and replay-time apply.
- Produces: `Interview.project_id: Optional[str]` attribute (default `None`), set on `InterviewCreated`. `LensApplied` / `LensExtractionGenerated` / `LensExtractionOverridden` events now carry `project_id` in their envelope (hence ESDB metadata) whenever the interview was created with one. Later tasks (watcher) read this from event metadata.

**Design note:** the three lens methods forward `**envelope_kwargs` to `_add_event`. Use `envelope_kwargs.setdefault("project_id", self.project_id)` so an explicit caller-supplied `project_id` still wins, and a `None` self.project_id (interview created before this change, or without a project) is a harmless no-op (identical to not passing it).

- [ ] **Step 1: Write the failing tests**

Add to `tests/events/test_aggregates_unit.py` (match the file's existing import style — `from src.events.aggregates import Interview`, `from src.events.envelope import Actor, ActorType`; interview ids must be valid UUIDs per the envelope validator):

```python
def test_interview_stores_project_id_from_created_event():
    iid = str(uuid.uuid4())
    interview = Interview(iid)
    interview.create(title="T", source="s", project_id="proj-42")
    assert interview.project_id == "proj-42"


def test_interview_project_id_defaults_to_none_without_project():
    iid = str(uuid.uuid4())
    interview = Interview(iid)
    interview.create(title="T", source="s")
    assert interview.project_id is None


def test_apply_lens_stamps_project_id_onto_event():
    iid = str(uuid.uuid4())
    interview = Interview(iid)
    interview.create(title="T", source="s", project_id="proj-42")
    event = interview.apply_lens("persona", 1)
    assert event.project_id == "proj-42"


def test_record_lens_extraction_stamps_project_id_onto_event():
    iid = str(uuid.uuid4())
    interview = Interview(iid)
    interview.create(title="T", source="s", project_id="proj-42")
    interview.apply_lens("persona", 1)
    event = interview.record_lens_extraction(
        lens="persona", lens_version=1, node_type="Trait", item_id=str(uuid.uuid4()),
        fields={"text": "Decisive"}, supporting_fragment_ids=[], speaker_links=[],
        confidence=0.9, model="haiku", provider="anthropic",
    )
    assert event.project_id == "proj-42"


def test_override_lens_extraction_stamps_project_id_onto_event():
    iid = str(uuid.uuid4())
    interview = Interview(iid)
    interview.create(title="T", source="s", project_id="proj-42")
    interview.apply_lens("persona", 1)
    item_id = str(uuid.uuid4())
    interview.record_lens_extraction(
        lens="persona", lens_version=1, node_type="Trait", item_id=item_id,
        fields={"text": "Decisive"}, supporting_fragment_ids=[], speaker_links=[],
        confidence=0.9, model="haiku", provider="anthropic",
    )
    event = interview.override_lens_extraction(item_id, fields_overridden={"text": "Very decisive"})
    assert event.project_id == "proj-42"


def test_explicit_project_id_kwarg_wins_over_aggregate_value():
    iid = str(uuid.uuid4())
    interview = Interview(iid)
    interview.create(title="T", source="s", project_id="proj-42")
    event = interview.apply_lens("persona", 1, project_id="override-proj")
    assert event.project_id == "override-proj"
```

Ensure `import uuid` is present at the top of the test file (it almost certainly is; add it if not).

- [ ] **Step 2: Run tests to verify they fail**

Run: `~/.pyenv/versions/3.10.7/bin/python -m pytest tests/events/test_aggregates_unit.py -k "project_id" -q --no-cov`
Expected: FAIL — `AttributeError: 'Interview' object has no attribute 'project_id'` (and the lens-event assertions fail because `event.project_id is None`).

- [ ] **Step 3: Implement the aggregate changes**

In `Interview.__init__` (after `super().__init__(aggregate_id)`, alongside the other attribute initializers ~line 176):

```python
        self.project_id: Optional[str] = None
```

In `Interview._apply_interview_created` (~line 241), add after `data = event.data`:

```python
        self.project_id = event.project_id or data.get("project_id")
```

In `Interview.apply_lens` (~line 695), immediately before the `return self._add_event(` call:

```python
        envelope_kwargs.setdefault("project_id", self.project_id)
```

Add the identical `envelope_kwargs.setdefault("project_id", self.project_id)` line immediately before the `return self._add_event(` call in `Interview.record_lens_extraction` (~line 758) and in `Interview.override_lens_extraction` (~line 779).

- [ ] **Step 4: Run tests to verify they pass**

Run: `~/.pyenv/versions/3.10.7/bin/python -m pytest tests/events/test_aggregates_unit.py -q --no-cov`
Expected: PASS (new tests green, no regressions in the file).

- [ ] **Step 5: Commit**

```bash
git add src/events/aggregates.py tests/events/test_aggregates_unit.py
git commit -m "feat(events): Interview aggregate carries and stamps project_id on lens events"
```

---

### Task 2: Watcher decodes metadata `project_id`; `scope_notifications` emits `project` for Interview-stream events

**Files:**
- Modify: `src/ui/notifications.py` (`scope_notifications` ~line 42; `EsdbWatcher._handle_event` ~line 327)
- Test: `tests/ui/test_notifications.py` (`FakeRecordedEvent` ~line 206; add pure-mapping and watcher tests)

**Interfaces:**
- Consumes: lens events on `Interview-*` streams now carry `project_id` in ESDB metadata (Task 1). The watcher's `RecordedEvent` exposes `.metadata` (bytes, JSON-encoded — the same dict `src/events/store.py:156-172` writes, which includes `"project_id"`).
- Produces: `scope_notifications(stream_name: str, payload: dict, *, event_project_id: Optional[str] = None) -> List[Notification]`. For `Interview-*` events it emits `Notification("project", project_id=...)` (in addition to the existing `transcript` and conditional `interviews`) whenever a `project_id` is resolvable from either the payload or `event_project_id`. `_handle_event` decodes `event.metadata` best-effort and passes the extracted `project_id` as `event_project_id`.

**Design notes:**
- Keep the existing `interviews` notification exactly as-is (fires only when the *payload* carries `project_id` — effectively `InterviewCreated`, which puts `project_id` in `data`). The new `project` emission is additive and uses the resolved id `payload.get("project_id") or event_project_id`.
- Metadata decoding must be best-effort inside the existing `_handle_event` try/except: a malformed or missing metadata blob yields `event_project_id=None` and must never raise (the method already logs-and-skips on any exception; keep that guarantee). `RecordedEvent.metadata` may be absent/empty on a fake or a metadata-less event — guard with `getattr(event, "metadata", None)` and treat empty/unparseable as `None`.

- [ ] **Step 1: Write the failing tests (pure mapping)**

Add to `tests/ui/test_notifications.py`, alongside the existing `scope_notifications` tests (module already defines `IID`, `PID`, and imports `scope_notifications`, `Notification`):

```python
def test_interview_stream_with_event_project_id_adds_project_notification():
    result = scope_notifications(f"Interview-{IID}", {}, event_project_id=PID)
    assert result == [
        Notification("transcript", interview_id=IID),
        Notification("project", project_id=PID),
    ]


def test_interview_stream_with_payload_project_id_adds_interviews_and_project():
    # InterviewCreated carries project_id in the payload: interviews (existing)
    # AND project (new) both fire, resolved from the payload.
    result = scope_notifications(f"Interview-{IID}", {"project_id": PID})
    assert result == [
        Notification("transcript", interview_id=IID),
        Notification("interviews", project_id=PID),
        Notification("project", project_id=PID),
    ]


def test_interview_stream_without_any_project_id_emits_only_transcript():
    result = scope_notifications(f"Interview-{IID}", {})
    assert result == [Notification("transcript", interview_id=IID)]


def test_payload_project_id_takes_precedence_over_event_project_id():
    result = scope_notifications(
        f"Interview-{IID}", {"project_id": PID}, event_project_id="other-proj"
    )
    assert Notification("project", project_id=PID) in result
    assert Notification("project", project_id="other-proj") not in result


def test_project_stream_unaffected_by_event_project_id_kwarg():
    # A Project-* event still maps to exactly one project notification.
    result = scope_notifications(f"Project-{PID}", {"project_id": PID}, event_project_id="ignored")
    assert result == [Notification("project", project_id=PID)]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `~/.pyenv/versions/3.10.7/bin/python -m pytest tests/ui/test_notifications.py -k "event_project_id or without_any_project_id or payload_project_id or project_stream_unaffected" -q --no-cov`
Expected: FAIL — `scope_notifications()` got an unexpected keyword argument `event_project_id`.

- [ ] **Step 3: Implement `scope_notifications`**

Replace the `Interview-` branch and the signature in `src/ui/notifications.py`:

```python
def scope_notifications(
    stream_name: str, payload: dict, *, event_project_id: Optional[str] = None
) -> List[Notification]:
    """Map one core-domain stream event to zero or more surface notifications.

    Pure function: no I/O, no hub access. Returns [] when the stream isn't
    one we notify on, or when a required id is missing from the payload
    (log-and-skip for the missing-key case is the caller's job, not ours).

    `event_project_id` carries a project_id resolved from the event's
    metadata envelope (Interview-stream lens events stamp it there but not in
    their data payload); it lets an Interview-* event reach the gallery via
    the `project` surface without a per-event DB lookup.
    """
    if stream_name.startswith("Sentence-"):
        interview_id = payload.get("interview_id")
        if not interview_id:
            return []
        return [Notification("transcript", interview_id=interview_id)]

    if stream_name.startswith("Interview-"):
        interview_id = stream_name[len("Interview-") :]
        if not interview_id:
            return []
        notifications = [Notification("transcript", interview_id=interview_id)]
        payload_project_id = payload.get("project_id")
        if payload_project_id:
            notifications.append(Notification("interviews", project_id=payload_project_id))
        # Coarse by design: any Interview-stream event with a resolvable
        # project (payload OR metadata) nudges the gallery via `project`.
        resolved_project_id = payload_project_id or event_project_id
        if resolved_project_id:
            notifications.append(Notification("project", project_id=resolved_project_id))
        return notifications

    if stream_name.startswith("Project-"):
        project_id = payload.get("project_id")
        if not project_id:
            return []
        return [Notification("project", project_id=project_id)]

    return []
```

- [ ] **Step 4: Run the mapping tests to verify they pass**

Run: `~/.pyenv/versions/3.10.7/bin/python -m pytest tests/ui/test_notifications.py -k "scope or project_id or transcript or interview" -q --no-cov`
Expected: PASS (new mapping tests green; existing `scope_notifications` tests still green — the `interviews` behavior is unchanged, and the existing `test_interview_stream_with_project_id_adds_interviews_notification` now also sees a trailing `project` notification, so **update that existing test's expected list** to include `Notification("project", project_id=PID)` as its third element).

- [ ] **Step 5: Write the failing watcher test (metadata decode)**

First, extend `FakeRecordedEvent` (~line 206) to carry metadata without breaking its existing positional constructions:

```python
@dataclass
class FakeRecordedEvent:
    """Stands in for esdbclient.RecordedEvent: the fields the watcher reads
    -- resolved stream name, raw data bytes, and raw metadata bytes."""

    stream_name: str
    data: bytes
    metadata: bytes = b"{}"
```

Then add a watcher test (mirrors the existing `test_watcher_publishes_notifications_from_subscribed_events` idiom):

```python
@pytest.mark.asyncio
async def test_watcher_emits_project_notification_from_lens_event_metadata():
    hub = NotificationHub()
    sub = hub.subscribe(interview_id=None, project_id=PID)

    client = FakeEventStoreDBClient()
    interview_sub = FakeSubscription()
    # A lens event: project_id lives in metadata (envelope), NOT in data.
    interview_sub.push_event(
        FakeRecordedEvent(
            f"Interview-{IID}",
            json.dumps({"lens": "persona", "lens_version": 1}).encode(),
            json.dumps({"project_id": PID}).encode(),
        )
    )
    client.queue_subscription("$ce-Interview", interview_sub)

    watcher = make_watcher(client, hub)
    await watcher.ensure_started()
    try:
        notification = await asyncio.wait_for(sub.queue.get(), timeout=2.0)
        assert notification == Notification("project", project_id=PID)
    finally:
        await watcher.stop()


@pytest.mark.asyncio
async def test_watcher_tolerates_malformed_metadata_on_interview_event():
    # Bad metadata must not raise: the transcript notification still lands.
    hub = NotificationHub()
    sub = hub.subscribe(interview_id=IID, project_id=None)

    client = FakeEventStoreDBClient()
    interview_sub = FakeSubscription()
    interview_sub.push_event(
        FakeRecordedEvent(f"Interview-{IID}", json.dumps({}).encode(), b"not-json")
    )
    client.queue_subscription("$ce-Interview", interview_sub)

    watcher = make_watcher(client, hub)
    await watcher.ensure_started()
    try:
        notification = await asyncio.wait_for(sub.queue.get(), timeout=2.0)
        assert notification == Notification("transcript", interview_id=IID)
    finally:
        await watcher.stop()
```

- [ ] **Step 6: Run watcher tests to verify they fail**

Run: `~/.pyenv/versions/3.10.7/bin/python -m pytest tests/ui/test_notifications.py -k "metadata" -q --no-cov`
Expected: FAIL — the project notification never arrives (the watcher does not yet decode metadata, so `event_project_id` is never passed).

- [ ] **Step 7: Implement `_handle_event` metadata decode**

Replace `EsdbWatcher._handle_event` (~line 327) in `src/ui/notifications.py`:

```python
    def _handle_event(self, event: Any) -> None:
        """Decode one resolved event and publish its mapped notifications.
        Malformed `event.data` (unparseable JSON, or valid JSON that isn't
        the dict-shaped payload scope_notifications expects) is logged at
        debug and skipped. Event metadata is decoded best-effort to recover a
        `project_id` an Interview-stream lens event stamps in its envelope but
        not its data payload -- a metadata surprise never aborts handling."""
        try:
            payload = json.loads(event.data)
            event_project_id = self._project_id_from_metadata(event)
            notifications = scope_notifications(
                event.stream_name, payload, event_project_id=event_project_id
            )
        except Exception:
            logger.debug("EsdbWatcher: skipping malformed event on stream '%s'", event.stream_name)
            return

        for notification in notifications:
            self._hub.publish(notification)

    @staticmethod
    def _project_id_from_metadata(event: Any) -> Optional[str]:
        """Best-effort project_id from the event's metadata envelope. Returns
        None on absent/empty/unparseable/non-dict metadata -- never raises."""
        raw = getattr(event, "metadata", None)
        if not raw:
            return None
        try:
            meta = json.loads(raw)
        except (ValueError, TypeError):
            return None
        if isinstance(meta, dict):
            return meta.get("project_id")
        return None
```

- [ ] **Step 8: Run the full notifications test file to verify it passes**

Run: `~/.pyenv/versions/3.10.7/bin/python -m pytest tests/ui/test_notifications.py -q --no-cov`
Expected: PASS (all mapping + watcher tests, including the two new metadata tests, green).

- [ ] **Step 9: Commit**

```bash
git add src/ui/notifications.py tests/ui/test_notifications.py
git commit -m "feat(ui): scope Interview-stream events to project surface via metadata project_id"
```

---

### Task 3: Frontend `keysForSurface` gallery keys + optional `personId` scope

**Files:**
- Modify: `frontend/src/hooks/useLiveInvalidation.ts` (`LiveInvalidationScopes` ~line 23; `keysForSurface` ~line 62; `useLiveInvalidation` destructure + `onmessage`/`onopen` + effect deps ~line 159)
- Test: `frontend/src/hooks/__tests__/useLiveInvalidation.test.tsx`

**Interfaces:**
- Consumes: `queryKeys` already exports `personas(projectId)`, `persona(projectId, personId)`, `persons(projectId)`, `person(projectId, personId)`, `worklist(projectId)`, `transcript(interviewId)`, `interviews(projectId)` (`frontend/src/hooks/queryKeys.ts`) — no changes needed there.
- Produces: `LiveInvalidationScopes` gains optional `personId?: string`. `keysForSurface("project", scopes)` returns `[transcript(interviewId)?, persons, personas, worklist, persona?, person?]`; `keysForSurface("resync", scopes)` returns the union of everything the hook watches. `buildStreamUrl` is unchanged (never sees `personId`).

**Design note:** `personId` refines only `keysForSurface`. `buildStreamUrl({ interviewId, projectId })` must keep its exact existing call sites — do NOT pass `personId` into the SSE URL (the server scopes by interview/project only).

- [ ] **Step 1: Write the failing tests**

Add to `frontend/src/hooks/__tests__/useLiveInvalidation.test.tsx` (the file already imports `keysForSurface` and `queryKeys`, and has a `keysForSurface` describe block — add there):

```typescript
describe("keysForSurface — gallery keys (M5.1b)", () => {
  it("project surface returns persons, personas, and worklist for a project scope", () => {
    const keys = keysForSurface("project", { projectId: "p1" });
    expect(keys).toEqual(
      expect.arrayContaining([
        queryKeys.persons("p1"),
        queryKeys.personas("p1"),
        queryKeys.worklist("p1"),
      ]),
    );
  });

  it("project surface adds persona+person detail keys only when personId is scoped", () => {
    const withPerson = keysForSurface("project", { projectId: "p1", personId: "per1" });
    expect(withPerson).toEqual(
      expect.arrayContaining([
        queryKeys.persona("p1", "per1"),
        queryKeys.person("p1", "per1"),
      ]),
    );
    const withoutPerson = keysForSurface("project", { projectId: "p1" });
    expect(withoutPerson).not.toContainEqual(queryKeys.persona("p1", "per1"));
    expect(withoutPerson).not.toContainEqual(queryKeys.person("p1", "per1"));
  });

  it("resync surface includes the gallery keys too", () => {
    const keys = keysForSurface("resync", { projectId: "p1", interviewId: "i1", personId: "per1" });
    expect(keys).toEqual(
      expect.arrayContaining([
        queryKeys.transcript("i1"),
        queryKeys.interviews("p1"),
        queryKeys.persons("p1"),
        queryKeys.personas("p1"),
        queryKeys.worklist("p1"),
        queryKeys.persona("p1", "per1"),
        queryKeys.person("p1", "per1"),
      ]),
    );
  });
});
```

Also add a hook-level test that a `project` notification with a `personId` scope invalidates the detail key (mirror the existing invalidation-test idiom in the file — grab `latestSource()`, call `send("project")`, advance timers, assert `invalidateQueries` was called with `queryKeys.persona(...)`). Model it on whatever the file's existing "project surface invalidates persons" test looks like; if none exists, spy on `client.invalidateQueries`:

```typescript
it("project notification invalidates the persona detail key when personId is scoped", () => {
  const { client, Wrapper } = makeWrapper();
  const spy = vi.spyOn(client, "invalidateQueries");
  renderHook(() => useLiveInvalidation({ projectId: "p1", personId: "per1" }), { wrapper: Wrapper });
  act(() => {
    latestSource().onopen?.();
    send("project");
  });
  expect(spy).toHaveBeenCalledWith({ queryKey: queryKeys.persona("p1", "per1"), exact: true });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run (from `frontend/`): `npm run test -- useLiveInvalidation`
Expected: FAIL — `keysForSurface("project", ...)` does not yet return `personas`/`worklist`/detail keys.

- [ ] **Step 3: Implement the scope + mapping changes**

In `frontend/src/hooks/useLiveInvalidation.ts`, extend the scope interface (~line 23):

```typescript
export interface LiveInvalidationScopes {
  interviewId?: string;
  projectId?: string;
  personId?: string;
}
```

Replace the `project` and `resync` cases in `keysForSurface` (~line 62). Update the docstring's mapping summary to match:

```typescript
export function keysForSurface(surface: string, scopes: LiveInvalidationScopes): QueryKey[] {
  const keys: QueryKey[] = [];
  const { interviewId, projectId, personId } = scopes;

  switch (surface) {
    case "transcript":
      if (interviewId) keys.push(queryKeys.transcript(interviewId));
      break;
    case "interviews":
      if (projectId) keys.push(queryKeys.interviews(projectId));
      break;
    case "project":
      if (interviewId) keys.push(queryKeys.transcript(interviewId));
      if (projectId) {
        keys.push(queryKeys.persons(projectId));
        keys.push(queryKeys.personas(projectId));
        keys.push(queryKeys.worklist(projectId));
        if (personId) {
          keys.push(queryKeys.persona(projectId, personId));
          keys.push(queryKeys.person(projectId, personId));
        }
      }
      break;
    case "resync":
      if (interviewId) keys.push(queryKeys.transcript(interviewId));
      if (projectId) {
        keys.push(queryKeys.interviews(projectId));
        keys.push(queryKeys.persons(projectId));
        keys.push(queryKeys.personas(projectId));
        keys.push(queryKeys.worklist(projectId));
        if (personId) {
          keys.push(queryKeys.persona(projectId, personId));
          keys.push(queryKeys.person(projectId, personId));
        }
      }
      break;
    default:
      break;
  }
  return keys;
}
```

In `useLiveInvalidation` (~line 159), thread `personId` through — but NOT into `buildStreamUrl`:

```typescript
export function useLiveInvalidation(
  scopes: LiveInvalidationScopes,
  timing?: LiveInvalidationTiming,
): LiveStatus {
  const { interviewId, projectId, personId } = scopes;
```

The effect's idle guard stays keyed on interview/project only (`if (!interviewId && !projectId)`) — `personId` alone never opens a stream. Keep `new EventSource(buildStreamUrl({ interviewId, projectId }))` unchanged. Update the two `keysForSurface(...)` call sites inside `onopen` (resync) and `onmessage` to pass `{ interviewId, projectId, personId }`. Add `personId` to the effect dependency array:

```typescript
  }, [interviewId, projectId, personId, coalesceMs, trailingMs, queryClient]);
```

- [ ] **Step 4: Run tests to verify they pass**

Run (from `frontend/`): `npm run test -- useLiveInvalidation`
Expected: PASS (new gallery-key and personId tests green; existing tests unchanged — the existing `project`-surface test still sees `persons(projectId)` among the returned keys).

- [ ] **Step 5: Commit**

```bash
git add frontend/src/hooks/useLiveInvalidation.ts frontend/src/hooks/__tests__/useLiveInvalidation.test.tsx
git commit -m "feat(ui): map project surface to gallery query keys; optional personId scope"
```

---

### Task 4: Wire `useLiveInvalidation` + `LiveIndicator` into the 5 gallery pages

**Files:**
- Modify: `frontend/src/app/gallery/personas/[projectId]/page.tsx`
- Modify: `frontend/src/app/gallery/personas/[projectId]/[personId]/page.tsx`
- Modify: `frontend/src/app/gallery/persons/[projectId]/page.tsx`
- Modify: `frontend/src/app/gallery/persons/[projectId]/[personId]/page.tsx`
- Modify: `frontend/src/app/gallery/worklist/page.tsx`
- Test: the co-located `__tests__/page.test.tsx` for each of the five

**Interfaces:**
- Consumes: `useLiveInvalidation(scopes)` → `LiveStatus`; `LiveIndicator({ status })`. Wiring pattern (verified in `frontend/src/app/workbench/[projectId]/page.tsx`): `const liveStatus = useLiveInvalidation({ projectId }); ... <LiveIndicator status={liveStatus} />`.
- Route params per page: personas grid → `{ projectId }` (`useParams`); persona detail → `{ projectId, personId }` (`useParams`); persons grid → `{ projectId }` (`useParams`); person detail → `{ projectId, personId }` (`useParams`); worklist → `projectId` from `searchParams.get("project") ?? ""` (falsy `""` → hook returns `"idle"`, no stream — correct).

**Design note:** each page already has its route params in scope. Add the hook call near the top of the component and render `<LiveIndicator>` in the header area next to the `<h1>` (match the workbench placement/markup). Grids pass only `{ projectId }`; detail pages pass `{ projectId, personId }`; worklist passes `{ projectId }`.

- [ ] **Step 1: Write the failing tests**

For each of the five `__tests__/page.test.tsx`, add an assertion that the live indicator renders. The `LiveIndicator` renders the text `"Live updates off"` in a non-connected test environment (no real EventSource → status stays `idle`/`offline`). Mirror each test file's existing render helper (they already render the page with a QueryClient wrapper and mocked hooks). Add:

```typescript
it("renders the live-updates indicator", async () => {
  renderPage(); // use whatever the file's existing render helper is called
  expect(await screen.findByText(/Live updates/i)).toBeInTheDocument();
});
```

If a test file stubs `EventSource`, keep the assertion on the `/Live updates/i` text (matches both `"on"` and `"off"`). Ensure `screen` is imported from `@testing-library/react` (it already is in these files).

- [ ] **Step 2: Run tests to verify they fail**

Run (from `frontend/`): `npm run test -- gallery`
Expected: FAIL — no element with text `Live updates` yet on the gallery pages.

- [ ] **Step 3: Wire the personas grid page**

In `frontend/src/app/gallery/personas/[projectId]/page.tsx`, add imports:

```typescript
import { useLiveInvalidation } from "@/hooks/useLiveInvalidation";
import { LiveIndicator } from "@/components/LiveIndicator";
```

After `const { projectId } = useParams...`:

```typescript
  const liveStatus = useLiveInvalidation({ projectId });
```

Render the indicator beside the heading (replace the `<h1>` line with a flex row that keeps the existing `<h1>` and adds the indicator, matching the workbench's header treatment):

```tsx
      <div className="flex items-center justify-between">
        <h1 className="text-lg font-semibold">Personas</h1>
        <LiveIndicator status={liveStatus} />
      </div>
```

- [ ] **Step 4: Wire the persona detail page**

In `frontend/src/app/gallery/personas/[projectId]/[personId]/page.tsx`, add the same two imports, then:

```typescript
  const liveStatus = useLiveInvalidation({ projectId, personId });
```

Render `<LiveIndicator status={liveStatus} />` in the heading row next to the existing `<h1>` (same flex-row treatment as Step 3).

- [ ] **Step 5: Wire the persons grid page**

In `frontend/src/app/gallery/persons/[projectId]/page.tsx`, add the imports, then:

```typescript
  const liveStatus = useLiveInvalidation({ projectId });
```

Render `<LiveIndicator status={liveStatus} />` in the heading row.

- [ ] **Step 6: Wire the person detail page**

In `frontend/src/app/gallery/persons/[projectId]/[personId]/page.tsx`, add the imports, then:

```typescript
  const liveStatus = useLiveInvalidation({ projectId, personId });
```

Render `<LiveIndicator status={liveStatus} />` in the heading row.

- [ ] **Step 7: Wire the worklist page**

In `frontend/src/app/gallery/worklist/page.tsx`, add the imports, then (after `const projectId = searchParams.get("project") ?? "";`):

```typescript
  const liveStatus = useLiveInvalidation({ projectId });
```

Render `<LiveIndicator status={liveStatus} />` in the heading row next to the worklist's `<h1>`. (When `projectId === ""` the hook returns `"idle"` and the indicator reads "Live updates off" — correct: no project selected, no stream.)

- [ ] **Step 8: Run tests to verify they pass**

Run (from `frontend/`): `npm run test -- gallery`
Expected: PASS (all five pages render the indicator; existing page tests unaffected).

- [ ] **Step 9: Commit**

```bash
git add frontend/src/app/gallery
git commit -m "feat(ui): wire live invalidation + indicator into the five gallery pages"
```

---

### Task 5: Live smoke — gallery `project` notifications (resolution + persona lens)

**Files:**
- Modify: `tests/integration/test_live_feed_smoke.py` (add one new test function; reuse the module's helpers)

**Interfaces:**
- Consumes: the same in-process technique the file already uses — call `stream_events(...)` directly and iterate its `StreamingResponse.body_iterator`. Emit real ESDB events via aggregates + repositories (no HTTP server): `IngestionOrchestrator` for the interview; `Project.identify_person` via `get_project_repository()` for the resolution leg; `LensEngine().apply(interview_id, "persona")` with a mocked executor for the lens leg.
- Produces: a project-scoped SSE subscriber that receives `Notification("project", project_id=...)` frames for both a `Project-*` resolution event and stamped `Interview-*` persona-lens events.

**Design notes (grounded deviations from the spec's illustrative wording):**
- The spec illustrates leg (b) as `python -m src.lens <iid> persona`; that CLI drives a live LLM (the project's OpenAI quota is exhausted). Use `LensEngine().apply(interview_id, "persona")` with the executor mocked to canned outcomes — exactly the pattern `tests/integration/test_layer3_lens_smoke.py::test_persona_lens_projects_dual_label_nodes_with_links_and_grounding` uses. This emits the *same real, stamped* `LensApplied`/`LensExtractionGenerated` events to ESDB, LLM-free — which is precisely the mechanism under test.
- The spec illustrates leg (a) as "link a speaker→person via the API". Emit the resolution event directly through the `Project` aggregate + `get_project_repository().save(...)` (the same calls the resolution router makes) rather than standing up an HTTP client alongside the direct `stream_events` driver. Same ESDB `PersonIdentified` event; the API is just one caller of `identify_person`.
- Open the SSE stream **project-scoped** (`project_id=P`, no `interview_id`) — the gallery's scope. `NotificationHub._matches` routes `project`-surface notifications by `project_id`, so both legs' notifications reach this subscriber. No `_FixedFirstUUID4` interview-id pinning is needed (we control `project_id` directly).
- Assert on `payload.get("surface") == "project" and payload.get("project_id") == P`, skipping `": keep-alive"` and `": connected"` frames, with the module's existing `NOTIFICATION_TIMEOUT_S` deadline loop. The `SUBSCRIPTION_SETTLE_S` buffer after the first chunk still applies (the watcher's `from_end` subscriptions must be live before events are emitted).

- [ ] **Step 1: Write the failing test**

Add to `tests/integration/test_live_feed_smoke.py` (reuse `_bare_request`, `HEARTBEAT_TEST_SECONDS`, `SUBSCRIPTION_SETTLE_S`, `NOTIFICATION_TIMEOUT_S`, `LABELED`). Add imports at the top of the function as shown:

```python
@pytest.mark.asyncio
async def test_live_feed_delivers_project_notifications_for_resolution_and_lens(
    tmp_path, monkeypatch
):
    """Gallery liveness (M5.1b): a project-scoped SSE subscriber receives a
    `project` notification for (a) a Project-stream resolution event and
    (b) stamped Interview-stream persona-lens events. Leg (b) proves the
    aggregate-stamp mechanism end-to-end: the persona lens run emits events
    whose project_id lives only in ESDB metadata, and the watcher recovers it
    there and routes them to the gallery."""
    from unittest.mock import AsyncMock, MagicMock

    from src.api.routers.ui import stream_events
    from src.events.aggregates import Project
    from src.events.project_events import project_aggregate_id
    from src.events.repository import get_project_repository
    from src.ingestion.orchestrator import IngestionOrchestrator
    from src.lens.engine import LensEngine

    project_id = f"live-gallery-smoke-{uuid_mod.uuid4()}"

    monkeypatch.setattr("src.api.routers.ui.HEARTBEAT_SECONDS", HEARTBEAT_TEST_SECONDS)
    monkeypatch.setattr(
        "starlette.requests.Request.is_disconnected", AsyncMock(return_value=False)
    )

    # Ingest a labeled interview into this project (Layer 1) BEFORE opening the
    # stream -- the persona lens re-run below is the event we watch for, not
    # this ingest (the watcher is from_end).
    input_file = tmp_path / "gallery_smoke.txt"
    input_file.write_text(LABELED)
    ingest = IngestionOrchestrator(project_id=project_id, map_dir=tmp_path / "maps")
    ingest_result = await ingest.ingest_file(input_file)
    interview_id = ingest_result.interview_id

    response = await stream_events(_bare_request(), interview_id=None, project_id=project_id)
    assert response.media_type == "text/event-stream"
    generator = response.body_iterator

    async def next_project_notification():
        deadline = asyncio.get_running_loop().time() + NOTIFICATION_TIMEOUT_S
        while True:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                pytest.fail(f"No project notification for {project_id} within {NOTIFICATION_TIMEOUT_S}s.")
            chunk = await asyncio.wait_for(generator.__anext__(), timeout=remaining)
            if chunk in (": keep-alive\n\n", ": connected\n\n"):
                continue
            assert chunk.startswith("data: ") and chunk.endswith("\n\n")
            payload = json.loads(chunk[len("data: "):].strip())
            if payload.get("surface") == "project" and payload.get("project_id") == project_id:
                return payload

    try:
        first_chunk = await asyncio.wait_for(
            generator.__anext__(), timeout=HEARTBEAT_TEST_SECONDS + 10
        )
        assert first_chunk == ": connected\n\n"
        await asyncio.sleep(SUBSCRIPTION_SETTLE_S)

        # --- Leg (a): a Project-stream resolution event -> project notification.
        repo = get_project_repository()
        project = Project(project_aggregate_id(project_id))
        project.identify_person(project_id, str(uuid_mod.uuid4()), "Jane Doe")
        await repo.save(project)
        assert await next_project_notification() == {"surface": "project", "project_id": project_id}

        # --- Leg (b): a persona-lens run -> stamped Interview-stream events ->
        # project notification (mechanism proof; executor mocked, LLM-free).
        executor = MagicMock()
        executor.run_spec_on_text = AsyncMock(
            side_effect=lambda spec, text, ctx=None: SpecOutcome(
                data={spec.name: []}, provider="anthropic", model="haiku"
            )
        )
        monkeypatch.setattr(LensEngine, "_build_executor", lambda self, lens: executor)
        await LensEngine().apply(interview_id, "persona")
        assert await next_project_notification() == {"surface": "project", "project_id": project_id}
    finally:
        await generator.aclose()
```

Add `from src.enrichment.executor import SpecOutcome` to the module's top-level imports (mirroring `test_layer3_lens_smoke.py`).

- [ ] **Step 2: Bring up the dev stack and run the smoke to verify leg (b) fails without the backend changes present**

This test depends on Tasks 1–2 (the aggregate stamp + watcher metadata decode). If run on this branch after Tasks 1–2 are committed, it should PASS; to confirm it is a real end-to-end check, temporarily stash the Task-1 `setdefault` lines and observe leg (b) time out. Restore before continuing. Run:

```bash
make live-feed-smoke
```

Expected (with Tasks 1–2 present): PASS. Expected (Task-1 stamp removed): leg (b) fails with the "No project notification" timeout, proving the assertion exercises the stamp. Leg (a) passes regardless (Project-stream events are unaffected by M5.1b).

- [ ] **Step 3: Run the full smoke to verify it passes**

Run: `make live-feed-smoke`
Expected: PASS — both the existing transcript smoke and the new gallery smoke green.

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_live_feed_smoke.py
git commit -m "test(integration): live smoke — gallery project notifications for resolution and lens"
```

---

## Post-Implementation Verification

Run the full backend and frontend suites (not just the touched files) before the whole-branch review:

- Backend unit: `set -a; source .env; set +a; ~/.pyenv/versions/3.10.7/bin/python -m pytest tests/ui tests/events -q`
- Frontend: from `frontend/` — `npm run test`, then `npm run lint`, then `npm run build`
- Live smoke (env-gated, dev stack): `make live-feed-smoke`

Update `.superpowers/sdd/progress.md` with the M5.1b SHIPPED line after the whole-branch review is clean and the merge is approved.
