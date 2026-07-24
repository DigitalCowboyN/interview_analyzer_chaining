# M5.1 — Live Workbench: Real-Time Projection Feed (design)

**Status:** approved by owner 2026-07-24 (brainstorm dialogue)
**Parent:** ROADMAP "Upcoming — the UI arc"; builds on M5.0 (PR #11, main
`81235aa`). Real-time is a committed owner requirement, not optional.

## Goal

The workbench transcript becomes a dynamic surface: line items appear and
resequence live (by `sequence_order`) as ingestion/enrichment/lens/resolution
events process, and corrections made by any user show up without manual
refresh. The workbench interview list updates live too (new interviews
appear, counts tick). Two M5.0 final-review riders land first so the
real-time layer builds on a clean foundation.

**Delivery split (owner decision 2026-07-24):** the owner wants full liveness
(gallery/worklist included) eventually, but this milestone ships transcript +
interview list to keep the risky-infrastructure PR reviewable; gallery
liveness is a committed thin fast-follow PR on the proven pipeline ("add
subscription scopes", no rework).

## Architecture (chosen: backend SSE bridge)

No projection-service changes. No new infrastructure. The FastAPI backend
grows a notification path; browsers connect over SSE through the existing
Next.js `/api/*` proxy.

- **`src/ui/notifications.py`** (all logic, unit-testable — the reader
  idiom) + one SSE route on the existing `/ui` router.
- **EsdbWatcher** — one background task in the API process holding three
  ephemeral **catch-up** subscriptions (`$ce-Interview`, `$ce-Sentence`,
  `$ce-Project`), subscribed **from the current end of stream** — no replay,
  no consumer group, nothing persistent. Reuses the M4.7 conventions: sync
  esdbclient iteration via `asyncio.to_thread`, link resolution on category
  streams. Lazy lifecycle: starts on first SSE connection, stops when the
  last one closes. ESDB drop → backoff reconnect; on reconnect every
  connected client receives one `resync` message.
- **NotificationHub** — in-process pub/sub: each SSE connection registers an
  `asyncio.Queue` with scope filters; the watcher maps events to scopes and
  fans out. Single-uvicorn assumption is deliberate and documented (dev
  scale; multi-process fan-out is a future concern, YAGNI).
- **Scope mapping preserves loose coupling:** the browser NEVER sees event
  types, stream names, or any ESDB concept. The watcher translates
  server-side to surface tags — the entire browser contract:
  - Sentence + Interview events → `{surface: "transcript", interview_id}`
  - Interview events whose payload carries `project_id` (e.g.
    `InterviewCreated` — verified) additionally → `{surface: "interviews",
    project_id}`; Interview events without it emit the transcript scope only
  - Project events → `{surface: "project", project_id}` (person links
    affect open transcripts)
- **SSE endpoint:** `GET /ui/streams/events?interview_id=…&project_id=…` —
  one connection per open screen, server-side filtered. Heartbeat comment
  ~15s; `Cache-Control: no-cache` and `X-Accel-Buffering: no` so the Next
  proxy streams. `resync` message = "invalidate everything you watch once."

### Consistency caveat (accepted by owner, with escalation path)

The watcher is a separate consumer from the Neo4j writer, so a notification
can arrive before the projection lands — a naive refetch could be stale.
Mitigation (binding): the frontend debounces invalidations and always issues
a **trailing re-invalidate ~2s after the last notification in a burst**. If
this heuristic ever proves insufficient, the escalation path is a
post-projection emitter (projection service notifies after write+ack) —
a targeted upgrade that does not change the browser contract.

## Frontend: live invalidation layer

- **One new hook, `useLiveInvalidation(scopes)`** — opens the EventSource
  through the `/api` proxy, maps surface tags to the existing `queryKeys`
  builders, invalidates with debounce + trailing re-invalidate (~2s). No
  component touches EventSource directly (api → hooks → components → app
  layering holds).
- Transcript page subscribes `{transcript: interviewId, project:
  projectId}`; the interview-list page subscribes its project. Live
  appearance/resequencing falls out of refetch — the reader already orders
  by `sequence_order`.
- **Intent settle logic untouched:** the bounded confirm-poll remains the
  settle mechanism (must work with SSE down); push is additive freshness
  only.
- **Live/offline indicator:** a subtle header dot — EventSource open =
  live; closed/erroring = offline with the browser's native silent retry.
  SSE failure degrades to exactly M5.0 behavior; never a broken page.

## Riders (land FIRST, before real-time wires into these layers)

- **Rider A — intent-pattern hygiene** (M5.0 final-review batch, one task):
  extract ONE shared notice renderer (4 near-copies today, 2 byte-identical
  in `LineDetailPanel.tsx`); guard absence predicates against `undefined`
  query data mid-poll; route mutation URL paths through an encoding helper;
  invalidate `sentenceHistory` when a text edit settles; give
  `_handle_override` the same `ValueError → CommandValidationError` wrap as
  `_handle_edit` (parity + test).
- **Rider B — interview-metadata projection:** implement the
  `metadata_diff` branch in `InterviewMetadataUpdatedHandler` (today a
  literal `pass`): merge the diff into a **JSON-string property**
  `metadata_json` on the Interview node (Neo4j stores no nested maps;
  read-merge-write is safe — lanes serialize per interview). The `/ui`
  reader parses it back to a dict — frontend contract unchanged (it already
  expects a metadata object). Wire format untouched (event + payload already
  exist and flow). No migration: old interviews backfill via ESDB replay if
  ever needed; new ingests project immediately. The metadata panel then has
  real content — and updates live like everything else.

## Error handling

- ESDB unavailable at watcher start / mid-stream: backoff reconnect;
  clients stay connected and get `resync` when the watcher recovers. The
  SSE endpoint never 500s for watcher trouble — degradation, not breakage.
- SSE unsupported/blocked/closed: UI silently stays on M5.0
  fetch-on-navigation behavior; indicator shows offline.
- Malformed/unknown events in the watcher: log-and-skip (never kill the
  watcher loop).

## Testing

- **Backend unit:** hub fan-out + scope mapping with fake events; SSE
  endpoint via httpx streaming against a stubbed hub (first events,
  heartbeat, resync); watcher lifecycle (lazy start, last-client stop,
  reconnect → resync) with a fake ESDB client. Rider B: handler merge with
  fake sessions (query-text pins) + reader parse.
- **Frontend unit (Vitest):** `useLiveInvalidation` with a mocked
  EventSource — scope→queryKey mapping, debounce + trailing re-invalidate
  (fake timers), resync invalidates all watched keys, indicator states.
- **Live integration:** one deployed-smoke-family test — open the SSE
  stream against real API + ESDB, ingest a fragment, assert a `transcript`
  notification arrives.
- **Playwright:** extend `ui-smoke` with the milestone's proof — transcript
  page open, a new line ingested server-side **appears without any user
  action**.

## Non-goals (M5.1)

- Gallery/worklist liveness — the committed fast-follow PR on this
  pipeline.
- WebSockets; event replay/history over SSE (reconnect = one resync;
  initial state always from `/ui` reads).
- Auth (dev identity switcher stands).
- Multi-process/scaled fan-out (single-uvicorn hub by design).
- Edit observability (M5.2).
