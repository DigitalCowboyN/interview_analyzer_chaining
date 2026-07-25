# M5.1b — Gallery Liveness (design)

**Status:** approved by owner 2026-07-25 (brainstorm dialogue)
**Parent:** M5.1 (PR #13, main `ad53bb8`) deferred "gallery liveness — the
committed fast-follow PR on this pipeline." This is that follow-up.

## Goal

Extend the M5.1 SSE live feed so the **gallery** surfaces update live, with no
manual refresh: the persons grid and person detail as speakers are identified/
linked, the worklist as suggestions/canonicalizations land, and the personas
grid/detail as their composition changes AND as the persona lens re-runs.
Reuses the M5.1 pipeline (backend SSE bridge + frontend `useLiveInvalidation`
+ debounce/trailing re-invalidate) — mostly consumer-side wiring, plus one
small additive backend change so persona-lens events can reach the gallery.

## What is already live for free (no backend change)

The `Project` aggregate already stamps `project_id` on every resolution event
(`PersonIdentified`, `SpeakerLinkedToPerson`, `PersonLinkRemoved`,
`EntityCanonicalized`, `EntityAliasAdded`, `EntityMergeConfirmed`,
`EntitySplit`), and `scope_notifications` already emits the **`project`**
surface for all `Project-*` stream events. So **persons**, **worklist**, and
persona **composition** (which persons contribute) become live purely by
wiring the gallery pages to `useLiveInvalidation` and mapping the `project`
surface to the gallery query keys.

## The one backend gap: persona-lens content

`LensApplied` / `LensExtractionGenerated` / `LensExtractionOverridden` are
**Interview-stream** events, and the **Interview aggregate does not carry
`project_id`** (it never reads it from `InterviewCreated`), so lens events
have no project linkage the SSE watcher can use without a per-event DB lookup.
Owner decision (2026-07-25): include persona-lens-content liveness via a small,
additive stamp rather than a watcher-side lookup.

### Backend change (additive; wire format stays frozen)

1. **Interview aggregate carries `project_id`.** `_apply_interview_created`
   sets `self.project_id = event.project_id` (fall back to
   `event.data.get("project_id")`). The lens-emitting aggregate methods pass
   `project_id=self.project_id` to `_add_event`, so newly-emitted lens events
   carry `project_id` in their envelope/metadata. This is additive: `project_id`
   is an existing optional envelope/metadata field (already present on all
   Project-stream events) — **not** one of the FROZEN identifiers (event type
   names, `aggregate_type` "Sentence", `Sentence-{uuid}` streams, `sentence_id`,
   handler file names are all unchanged). Only newly-emitted lens events get it;
   historical ones don't, which is all liveness needs.
2. **Watcher decodes metadata `project_id`.** `EsdbWatcher._handle_event`
   already decodes `event.data`; it additionally decodes `event.metadata`
   (best-effort — malformed metadata is ignored, never raises) to extract
   `project_id`, and passes it to `scope_notifications`.
3. **`scope_notifications(stream_name, payload, *, event_project_id=None)`**:
   for `Interview-*` stream events, when a `project_id` is resolvable (from the
   payload or `event_project_id`), ALSO emit a `Notification("project",
   project_id=…)` — in addition to the existing `transcript`(+`interviews`)
   notifications. Coarse by design: any Interview-stream event with a project
   nudges the gallery; the debounce + trailing re-invalidate absorb enrichment
   bursts (per-event-type precision is a deliberate non-goal). The `project`
   surface stays the single "project changed" signal; consumers decide which
   keys it maps to.

## Frontend change

4. **`keysForSurface("project", scopes)`** currently returns
   `[transcript(interviewId?), persons(projectId)]`. Extend it to also return
   `personas(projectId)`, `worklist(projectId)`, and — when a `personId` scope
   is present — `persona(projectId, personId)` and `person(projectId, personId)`.
   (Queries not mounted on the current page invalidate as no-ops.)
5. **`useLiveInvalidation` gains an optional `personId` scope** (mirrors the
   existing optional `interviewId`), threaded into the SSE URL is NOT needed —
   the SSE subscription is still project/interview-scoped server-side; `personId`
   only refines which client keys the `project` surface invalidates. So
   `personId` is used solely by `keysForSurface`, not `buildStreamUrl`.
6. **Wire `useLiveInvalidation` into the 5 gallery pages** (each already knows
   its route params): personas grid `{projectId}`; persona detail
   `{projectId, personId}`; persons grid `{projectId}`; person detail
   `{projectId, personId}`; worklist `{projectId}`. Render the existing
   `LiveIndicator` on each (consistent with the workbench).

## Testing

- **Backend unit:** the Interview aggregate stamps `project_id` on lens events
  (and stores it on `InterviewCreated`); `_handle_event` decodes metadata
  `project_id` and passes it through (fake event with metadata); malformed
  metadata is ignored; `scope_notifications` emits `project` for an
  Interview-stream event with a resolvable `project_id`, and still returns the
  transcript/interviews notifications; a `Project-*` event is unchanged.
- **Frontend (Vitest):** `keysForSurface("project", …)` includes the gallery
  keys (and the detail keys only when `personId` is scoped); `useLiveInvalidation`
  with a `personId` scope invalidates the detail key on a `project` notification;
  the gallery pages mount the hook (no direct EventSource).
- **Live smoke:** extend the SSE smoke family (env-gated) — with a gallery SSE
  stream open, (a) a resolution event (link a speaker→person via the API) makes
  a `project` notification arrive; (b) re-running the persona lens
  (`python -m src.lens <iid> persona`) makes a `project` notification arrive —
  this leg VERIFIES the aggregate-stamp mechanism end-to-end (proves the
  newly-emitted lens event carried `project_id` to the watcher).

## Non-goals

- WebSockets; finer per-surface tags (coarse `project`-scoped invalidation is
  deliberate — the debounce is the safety valve).
- Back-filling `project_id` onto historical lens events (only new events need
  it for liveness).
- Gallery real-time beyond invalidation (no live-streaming individual rows;
  invalidate → refetch through the existing `/ui` reads).
- Edit observability (M5.2).
