# M5.0 — UI Scaffolding: Two-Surface App Shell (design)

**Status:** draft 2026-07-17 (awaiting owner review)
**Parent:** ROADMAP "Upcoming — the UI arc" (mapped 2026-07-17, owner decisions:
Next.js; dev identity switcher; scaffolding first, real-time committed in M5.1).

## Goal

The first user interface: a Next.js app with two distinct surfaces mirroring
the backend's CQRS split — a **workbench** (write side: navigate projects →
interviews → line-by-line transcript; manipulate metadata and line items) and
a **gallery** (read side + review actions: dynamic views that consume
projections). Dev identity switcher stands in for auth. Freshness is
polling/refetch; the real-time feed is M5.1 (committed, not optional).

## Domain model the UI encodes (owner-defined, 2026-07-17)

- **Speaker ≠ Person** (clarified 2026-07-17): a Speaker is a per-interview
  artifact from ingestion (each interview mints its own speakers from labels
  or inference — the same human in three interviews is three Speaker nodes).
  A **Person** is the project-scoped identity construct that unifies them
  (`Speaker —IDENTIFIED_AS→ Person`, 1:N across interviews). Automation
  links them ONLY in the reliable case (front-matter participants matching
  speaker display names — the ResolutionEngine's auto band); name heuristics
  produce worklist *suggestions* (human accepts); messy transcripts (generic
  labels, shared mics) have NO automation by design — the UI must provide the
  manual path (below).
- **Persona** = interpretive construct: a general domain representation (a
  role/archetype) built from persona-lens data (traits, goals, pain points,
  notable quotes).
- **Person ↔ Persona is many-to-many and loosely coupled.** A person can wear
  multiple hats (fulfill several personas); a persona can be fulfilled by
  anyone who takes on that domain. Person data *contributes to* persona data;
  neither owns the other.
- **Honest v1 gap:** the graph has no first-class Persona entity today — the
  persona lens attributes items to speakers/persons; archetype synthesis does
  not exist yet. Therefore: v1 persona views are **persona profiles seeded
  from per-person contributions** (the data that exists), but the UI's types,
  routes, and navigation treat Persona as its OWN entity type from day one —
  never a property or sub-page of Person — so the future **persona-synthesis
  milestone** (minting archetype Personas aggregating many persons' data, m:n)
  extends the gallery instead of rewriting it. That milestone goes on the
  ROADMAP backlog now.
- **Cards are a list/discovery pattern, never the core display.** Person cards
  and persona cards appear in grids/lists carrying core info plus a navigation
  affordance into the detailed core view. The exact interaction mechanism
  (context menu, button, click-through) is deliberately NOT pinned — an
  implementation/iteration concern, per owner.

## Architecture

**Approach (chosen):** Next.js (App Router, TypeScript) in `frontend/`, using
Next.js rewrites to proxy `/api/*` → FastAPI (no CORS changes; backend origin
not exposed). One new backend module pair supplies UI-shaped reads:
`src/ui/reader.py` (all Cypher; plain async session functions, fake-session
testable — the export/ask reader idiom) + `src/api/routers/ui.py` mounted
under `/ui/*`. Corrections reuse the EXISTING endpoints unchanged.

Rejected: composing screens from many fine-grained existing endpoints (the
transcript view would need N calls per line); a BFF layer inside Next.js
route handlers (an extra hop nothing needs before M5.1).

### Loose coupling UI ↔ core (owner requirement)

- **Derived contract:** TypeScript API types are GENERATED from FastAPI's
  OpenAPI schema (`openapi-typescript` against `/openapi.json`, checked into
  `frontend/src/api/schema.d.ts` with a regen script + a CI-friendly drift
  check). The contract is derived, never hand-drifted.
- **Single client layer:** one API-client module + TanStack Query hooks own
  ALL fetching/caching/invalidation; components never fetch directly.
- **Commands are intents:** correction calls surface the backend's real
  semantics — 202 accepted → optimistic/pending UI state → refetch confirms;
  409 conflicts render as actionable messages, never silent retries. The UI
  never pretends eventual consistency is synchronous.
- **No core concepts leak:** the frontend knows the `/ui/*` + correction
  contracts only — nothing of Neo4j, ESDB, streams, or projections.
- Frontend layering: `api/` (client + generated types) → `hooks/` (queries/
  mutations) → `components/` → `app/` (routes). Components depend on hooks,
  hooks on the client, never sideways.

## New backend reads (`/ui/*` — the only backend work)

| Endpoint | Serves |
|---|---|
| `GET /ui/projects` | Project list + interview counts (workbench + gallery nav roots) |
| `GET /ui/projects/{project_id}/interviews` | Interviews with title, created/status, fragment count |
| `GET /ui/interviews/{interview_id}/transcript` | The workbench aggregate: ordered lines — text, speaker (+person), utterance grouping, entities, segment membership, lens-item attachments, edited flag |
| `GET /ui/projects/{project_id}/personas` | Persona-profile card data (per contributing person: headline dimension counts, a representative quote) |
| `GET /ui/personas/{project_id}/{person_id}` | Persona core view: dimension-grouped items (traits/goals/pain points/quotes) with per-interview provenance |
| `GET /ui/projects/{project_id}/persons` | Person card data (display name, linked-speaker/interview counts) |
| `GET /ui/persons/{project_id}/{person_id}` | Person core view: identity facts — linked speakers per interview, aliases, contributes-to persona profile link |

> **Deviation note (final review, 2026-07-24):** "aliases" above was a spec
> defect — the backend has no person-alias concept, so there was nothing to
> build. Person aliases are deferred until the backend grows one (owner
> confirmed at M5.0 merge).

All project-scoped queries pin `(:Project {project_id})-[:CONTAINS_INTERVIEW]->`
(the M4.5b leak class). Reads only — the router contains zero write paths.

## Workbench (write surface)

- **Shell:** identity bar with the dev user switcher — a picker/free-text
  user id stored client-side and sent as `X-User-ID` on every mutating call
  (the existing actor-provenance contract). Real auth is a future milestone.
- **Navigation:** projects → interviews → transcript screen.
- **Transcript screen:** line-by-line display (ordered fragments, speaker
  labels, utterance grouping, segment headings); interview metadata panel;
  per-line detail panel (entities, lens items, analysis flags, edit history
  via the existing history endpoint).
- **Core corrections wired in v1** (existing endpoints, nothing new):
  transcript text edit (`edit_sentence`), speaker rename + fragment
  reattribute, segment remove, lens-item override, AND **manual
  speaker→person linking** (owner decision 2026-07-17): a per-speaker
  "identify as person…" affordance with a person picker (existing persons in
  the project, or create-new via the deterministic person id) calling the
  existing link endpoint, plus unlink. This is the escape hatch for speakers
  automation can't reach (generic labels, shared mics). Entity-level
  resolution corrections (merge/split/alias) stay off the workbench — they
  live on the gallery worklist (below).
- Every correction: optimistic/pending state → poll/refetch until the
  projection reflects it (bounded, with a "still processing" state) → settled.

## Gallery (read surface + review actions)

- **Persona area:** project-scoped persona-profile card grid AND per-interview
  persona lists (both discovery surfaces) → **persona core view** (the
  detailed projection display). Cards navigate; they never try to BE the view.
- **Person area:** person card grid → **person core view** (identity facts,
  loosely linked to the persona profiles they contribute to).
- **Worklist (actionable, owner decision):** the review worklist renders
  low-confidence items and merge/link suggestions WITH accept affordances
  calling the existing resolution endpoints (merge, link, alias). Degradation
  flags (`embedding_unavailable`) render as a visible "suggestions degraded"
  notice, never a broken page.
- Ask, segments-as-view, observability: later milestones.

## Error handling

API failure → inline, non-blocking notices (the backend's degradation-flag
doctrine made visible). 4xx validation/conflict → actionable message with the
server's detail. Never a white-screen; every screen has loading/empty/error
states as first-class renders.

## Testing

- Backend: `src/ui/reader.py` query-text pins with fake sessions (incl.
  project-scoping clauses); router tests for shapes/404s — inside the existing
  `./scripts/test.sh` gate.
- Frontend: Vitest component tests (transcript line rendering incl. edited/
  pending states; card→core-view navigation; identity switcher header
  injection); `tsc --noEmit` + ESLint as gates; one Playwright smoke
  (nav → transcript renders → text edit round-trips) against the live test
  stack, env-gated like the deployed smoke (own make target, not in default
  suites).
- A `make ui-dev` / `make ui-build` pair; frontend gates documented alongside
  the Python ones.

## Non-goals (M5.0)

- Real authentication (dev switcher only).
- Ingestion/upload UI (owner: transcript input is out of scope).
- Real-time updates (M5.1, committed); edit observability views (M5.2).
- **Persona synthesis** — first-class archetype Personas aggregating many
  persons' contributions (m:n realized in data, not just UI typing). Its own
  future milestone; goes on the ROADMAP backlog with this spec as context.
- Entity/canonical management screens beyond the worklist's accept actions.
- Visual polish beyond a clean neutral baseline (Tailwind + a lightweight
  component kit, light/dark) — restylable later without rework.
