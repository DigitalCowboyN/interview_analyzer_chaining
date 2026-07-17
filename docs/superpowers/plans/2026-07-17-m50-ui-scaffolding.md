# M5.0 — UI Scaffolding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The first UI: a Next.js two-surface app (workbench + gallery) in `frontend/`, backed by a new read-only `/ui/*` API, with a dev identity switcher and the core correction flows wired to existing endpoints.

**Architecture:** Backend adds ONE reader module + ONE router (reads only). Frontend is Next.js App Router + TypeScript in `frontend/`, proxying `/api/*` → FastAPI via rewrites (no CORS), with OpenAPI-generated types, a single API-client layer, and TanStack Query. Corrections are 202-accepted intents with pending→settled UI states. See the spec's "Loose coupling" section — it is binding.

**Tech Stack (pinned):** Next.js 15.x (App Router, TypeScript), Tailwind CSS v4, TanStack Query v5, openapi-typescript (type generation), Vitest + React Testing Library, Playwright (env-gated smoke), npm, Node ≥ 20. Backend: FastAPI + neo4j async driver (existing).

**Spec:** `docs/superpowers/specs/2026-07-17-m50-ui-scaffolding-design.md` (domain-model and loose-coupling sections are requirements, not prose).

## Global Constraints

- Backend additions are READS ONLY (`src/ui/reader.py` + `src/api/routers/ui.py`); zero write paths in the new router; every project-scoped query pins `(:Project {project_id: $project_id})-[:CONTAINS_INTERVIEW]->`. All mutations reuse existing endpoints unchanged.
- Wire format frozen; projection handlers remain the sole Neo4j writers.
- The frontend knows ONLY the HTTP contracts. No Neo4j/ESDB/projection concepts in frontend code or copy (UI copy says "processing", never "projection lag").
- Components never fetch directly: `frontend/src/api/` (client + generated types) → `frontend/src/hooks/` (TanStack Query hooks) → components → routes. Dependencies point one way.
- Cards are discovery surfaces that navigate to core views — never the core display. Interaction mechanism is implementer's choice (owner deliberately unpinned); navigation itself is required.
- UI treats Persona as its own entity type (own routes/types) even though v1 profiles are seeded per-person — the m:n future must extend, not rewrite.
- Python tests: `./scripts/test.sh <files> -q --no-cov` (targeted; FULL suite is controller-only). Python lint: `source .env 2>/dev/null; ~/.pyenv/shims/python -m flake8 <files>`. Frontend gates: `npm run lint`, `npm run typecheck`, `npm test` (Vitest) — run inside `frontend/`. FOREGROUND, sequential. `/usr/bin/grep`, not `grep`.
- Every commit ends with:
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` and
  `Claude-Session: https://claude.ai/code/session_0121ASUPnimKR64sB2odFiDv`

---

### Task 1: Backend — `/ui/*` reader + router

**Files:**
- Create: `src/ui/__init__.py`, `src/ui/reader.py`, `src/api/routers/ui.py`
- Modify: `src/main.py` (mount router)
- Test: `tests/ui/test_reader.py`, `tests/api/test_ui_router.py`

**Interfaces (Produces — the frontend contract):**
- `GET /ui/projects` → `{"projects": [{"project_id", "interview_count"}]}`
- `GET /ui/projects/{project_id}/interviews` → `{"interviews": [{"interview_id", "title", "created_at", "fragment_count"}]}` (404 unknown project)
- `GET /ui/interviews/{interview_id}/transcript` → `{"interview_id", "title", "metadata": {...front matter…}, "lines": [{"fragment_id", "sequence_order", "text", "speaker": {"speaker_id","display_name"}|null, "person": {"person_id","display_name"}|null, "utterance_id"|null, "segment": {"segment_id","topic"}|null, "entities": [{"surface","entity_type"}], "lens_items": [{"item_id","lens","node_type","text","confidence","human_locked"}], "edited": bool}]}` (404 unknown interview)
- `GET /ui/projects/{project_id}/personas` → persona-profile cards: `{"personas": [{"person_id", "display_name", "trait_count", "goal_count", "pain_point_count", "quote_count", "representative_quote"|null, "interview_ids": [...]}]}`
- `GET /ui/personas/{project_id}/{person_id}` → persona core view: `{"person_id", "display_name", "dimensions": {"traits": [...], "goals": [...], "pain_points": [...], "notable_quotes": [...]}}` where each item = `{"item_id","text","confidence","interview_id","interview_title"}` (per-interview provenance; 404 unknown)
- `GET /ui/projects/{project_id}/persons` → person cards: `{"persons": [{"person_id","display_name","speaker_count","interview_count"}]}`
- `GET /ui/persons/{project_id}/{person_id}` → person core view: `{"person_id","display_name","links": [{"interview_id","interview_title","speaker_id","speaker_display_name"}], "contributes_to_persona": bool}` (404 unknown)
- `GET /ui/projects/{project_id}/person-id?display_name=…` → `{"person_id"}` — compute-only derivation for the create-new-person flow, using THE SAME derivation the engine/suggestions use (read `src/resolution/candidates.py::person_groups` + `src/resolution/suggestions.py` line ~88 and reuse `person_id_for(project_id, <person_key derivation>)` exactly; keeping this server-side is a loose-coupling requirement — the frontend must never derive ids).

**Cypher notes (binding):** all project-scoped queries pin the `(:Project {project_id: $project_id})-[:CONTAINS_INTERVIEW]->` clause. Transcript lines: anchor `(:Interview {interview_id})-[:HAS_SENTENCE]->(f:Fragment)`, `ORDER BY f.sequence_order`; speaker via `[:SPOKEN_BY]`, person via `[:IDENTIFIED_AS]`, utterance via `[:PART_OF_UTTERANCE]`, segment via `(:Segment)-[:CONTAINS]->(f)`, entities via `[:MENTIONS]`, lens items via `(n:LensItem)-[:SUPPORTED_BY]->(f)`; null-strip collections with the export-reader idiom `[x IN collect(…) WHERE x.… IS NOT NULL]`; `edited` = `f.is_edited = true` (verify the actual edit-flag property by reading `SentenceEditedHandler` first — pin what the handler writes). Persona reads filter `n.lens = 'persona'` and `sp.merged_into IS NULL`; person reads filter `sp.merged_into IS NULL`.

- [ ] **Step 1: Write failing reader tests** — `tests/ui/test_reader.py` with the fake-session pattern from `tests/ask/test_reader.py`: one test per function pinning query text (project-scope clause, ORDER BY, null-strip, lens filter) + row-shape mapping. **Step 2:** run → fail. **Step 3:** implement `src/ui/reader.py` (plain async functions, session first arg — export/ask reader idiom). **Step 4:** reader tests green.
- [ ] **Step 5: Write failing router tests** — `tests/api/test_ui_router.py` mirroring `tests/api/test_ask_router.py`'s mocking idiom: 200 shapes for all eight endpoints, 404 legs, person-id derivation equality with `person_id_for` (import and compare — pins the contract). **Step 6:** implement `src/api/routers/ui.py` (thin: session → reader → shape; zero writes; no auth) + mount in `src/main.py` next to the other routers. **Step 7:** green.
- [ ] **Step 8:** flake8; commit `feat(api): /ui read layer — nav, transcript aggregate, persona/person views`.

---

### Task 2: Frontend scaffold — Next.js app, identity, API client, type generation

**Files:**
- Create: `frontend/` (Next.js app: `package.json`, `next.config.ts`, `tsconfig.json`, Tailwind config, `src/app/layout.tsx`, `src/app/page.tsx`, `src/api/client.ts`, `src/api/schema.d.ts` (generated), `src/identity/…`, `src/hooks/…` base, Vitest config + setup)
- Modify: `Makefile` (`ui-dev`, `ui-build`, `ui-test`, `ui-typegen` targets + help), root `README.md` gets a one-line pointer (full docs in Task 9), `.gitignore` (frontend build artifacts: `frontend/.next/`, `frontend/node_modules/`, `frontend/test-results/`)

**Interfaces:**
- Produces: `apiFetch` client (base `/api`, injects `X-User-ID` from the identity store on every request); `useIdentity()` (get/set user id, persisted in `localStorage`, default `"dev"`); generated `schema.d.ts` + `npm run typegen` (openapi-typescript against `http://localhost:8000/openapi.json`) + `npm run typecheck` failing on drift; `QueryProvider` wrapping the app; app shell with top-level nav: **Workbench | Gallery** + identity switcher in the header.

- [ ] **Step 1:** Scaffold with `create-next-app` (TypeScript, App Router, Tailwind, ESLint, `src/` dir, no import alias surprises — pin `@/*`). npm. Node ≥ 20 (document; do not add a version manager).
- [ ] **Step 2:** `next.config.ts` rewrites: `{ source: "/api/:path*", destination: "http://localhost:8000/:path*" }` (FastAPI routes are unprefixed at :8000). Backend URL overridable via `BACKEND_URL` env for the Playwright stack.
- [ ] **Step 3:** Identity: `src/identity/IdentityProvider.tsx` (context + localStorage persistence, default `"dev"`, a small set of preset dev users plus free-text) + header switcher component. `apiFetch` reads the current id and sets `X-User-ID` on every request; GETs include it too (harmless, consistent).
- [ ] **Step 4:** Type generation: `npm run typegen` script; run it against the live backend once and COMMIT `schema.d.ts`; `src/api/client.ts` exposes typed helpers keyed by path (openapi-typescript's `paths` type). Add a `typegen:check` script (regenerate to a temp file and diff — nonzero exit on drift) and document that backend contract changes require a regen commit.
- [ ] **Step 5:** TanStack Query provider (`QueryProvider` in `layout.tsx`), sensible defaults (staleTime ~5s, retry 1). Vitest + RTL configured (`npm test` runs headless, zero network — mock `apiFetch`).
- [ ] **Step 6:** App shell: header (product name, Workbench/Gallery nav links, identity switcher), empty landing page routing to `/workbench`. Loading/empty/error primitives (`<StateGate>` or equivalent — one shared component for the three states, used by every screen; this is the error-handling doctrine from the spec).
- [ ] **Step 7:** Tests: identity switcher persists + `apiFetch` header injection (Vitest); shell renders nav. Gates: `npm run lint && npm run typecheck && npm test` all green. Makefile targets added.
- [ ] **Step 8:** Commit `feat(ui): Next.js scaffold — shell, identity switcher, typed API client`.

---

### Task 3: Workbench navigation — projects and interviews

**Files:** `frontend/src/app/workbench/page.tsx` (projects), `frontend/src/app/workbench/[projectId]/page.tsx` (interviews), hooks `useProjects`, `useInterviews`, components + tests.

Behavior spec (binding): projects screen lists projects (id + interview count) → click-through to interviews screen (title, created, fragment count) → click-through to the transcript route (Task 4). All three UI states via the shared state component; breadcrumbs (`Workbench / {project} / {interview}`). Vitest: renders from mocked hooks, empty state, navigation targets correct.

- [ ] Implement + tests green + lint/typecheck; commit `feat(ui): workbench navigation — projects, interviews`.

---

### Task 4: Transcript screen — display

**Files:** `frontend/src/app/workbench/[projectId]/[interviewId]/page.tsx`, `useTranscript`, components (`TranscriptLine`, `SegmentHeading`, `MetadataPanel`, `LineDetailPanel`), tests.

Behavior spec (binding):
- Ordered lines; speaker label per line (person display name beside it when linked: "Jane (Jane Doe)" pattern — mirror the ask context convention); utterance grouping visually indicated (lines of one utterance visibly related); segment topic headings between lines where segments begin; `edited` badge on edited lines.
- Metadata panel: title + front-matter fields.
- Line detail panel (opens from a line): full text, entities, lens items (lens, type, text, confidence, lock state), edit history fetched lazily from the EXISTING history endpoint (`GET /sentences/{interview_id}/{sentence_index}/history` — read the endpoint for its index semantics first).
- Read-only in this task (corrections are Task 5). Vitest: line rendering incl. person suffix/edited badge/segment headings order; detail panel content from mocked data.

- [ ] Implement + tests green + lint/typecheck; commit `feat(ui): transcript screen — lines, segments, metadata, line detail`.

---

### Task 5: Workbench corrections — the intent pattern + four flows

**Files:** `frontend/src/hooks/mutations.ts` (shared intent-mutation wrapper), edit affordances in `TranscriptLine`/`LineDetailPanel`, tests.

Behavior spec (binding):
- **Shared intent pattern first** (one wrapper, used by every correction): fire mutation → optimistic "pending" state on the affected element → on 202, poll the relevant `/ui` read (bounded: every 2s, max 10 tries) until the change is reflected, then invalidate + settle; if not reflected in bounds, show "still processing — refresh later" (NOT an error); 409 → revert optimistic state + show the server's `detail` actionable message; network error → revert + notice.
- Flows: (1) transcript text edit → `POST /sentences/{interview_id}/{sentence_index}/edit` body `{text, editor_type: "human", note?}` (verify path/index semantics by reading `src/api/routers/edits.py` first); (2) speaker rename → existing speakers route; fragment reattribute → existing route (read `src/api/routers/speakers.py` for exact paths/bodies); (3) segment remove → `DELETE /segments/{interview_id}/{segment_id}?reason=…`; (4) lens-item override → existing lenses route (read it).
- Vitest: the intent wrapper's four outcomes (settle, bounded-timeout, 409 revert+message, network revert) with mocked client + fake timers; one flow test per correction asserting endpoint/body.

- [ ] Implement + tests green + lint/typecheck; commit `feat(ui): correction intents — text edit, speaker, segment, lens override`.

---

### Task 6: Manual speaker→person linking

**Files:** person-picker component + link/unlink affordances (speaker context in transcript + line detail), `usePersons`/`usePersonId` hooks, tests.

Behavior spec (binding): per-speaker "identify as person…" affordance → picker listing the project's persons (`GET /ui/projects/{id}/persons`) + create-new (name input → `GET /ui/projects/{id}/person-id?display_name=…` → link call carries `display_name` so the backend mints the person — spec: the FRONTEND NEVER derives ids); link → `POST /resolution/{project_id}/persons/{person_id}/link` body `{interview_id, speaker_id, display_name?}`; unlink affordance on linked speakers → the unlink endpoint (read it for body shape). Uses the Task 5 intent pattern (person suffix appears/disappears on settle). Vitest: picker lists + create-new flow calls derivation then link with display_name; unlink flow.

- [ ] Implement + tests green + lint/typecheck; commit `feat(ui): manual speaker→person linking — picker, create-new, unlink`.

---

### Task 7: Gallery — personas and persons

**Files:** `frontend/src/app/gallery/page.tsx` (gallery home), `gallery/personas/…` (project persona card grid, per-interview persona list, persona core view route), `gallery/persons/…` (person card grid, person core view route), hooks, tests.

Behavior spec (binding):
- Gallery home: project selector → two areas (Personas, Persons) + Worklist entry (Task 8).
- **Persona cards** (grid, project-scoped): display name, dimension counts, representative quote; card carries a navigation affordance to the persona core view (mechanism = implementer's choice; navigation required). **Per-interview persona list**: interview selector → persona profiles present in that interview → same core view.
- **Persona core view** (a CORE display, not a card): dimension-grouped items with per-interview provenance chips; distinct route `gallery/personas/[projectId]/[personId]`.
- **Person cards** → **person core view**: identity facts (linked speakers per interview), with a loose link to their persona profile (navigates to persona core view — the m:n future means this is a *link between views*, never an embedding of one in the other).
- Persona and Person are separate route trees and types — enforced by the spec's domain model.
- Vitest: card content + navigation targets; core-view rendering incl. provenance; empty states (project with no persona lens run yet shows a helpful empty state naming the CLI, copy pinned in test).

- [ ] Implement + tests green + lint/typecheck; commit `feat(ui): gallery — persona/person cards and core views`.

---

### Task 8: Gallery worklist with actions

**Files:** `frontend/src/app/gallery/worklist/page.tsx`, hooks, tests.

Behavior spec (binding): renders the existing `GET /review/worklist?project_id=…` response — low-confidence lens items (link into the workbench transcript at that interview), entity-merge suggestions (surfaces, score, band) with an accept affordance → merge endpoint, person-link suggestions with accept → link endpoint; `flags` containing `embedding_unavailable` renders a visible "suggestions degraded — embedding provider unavailable" banner while deterministic rows still render. Accepts use the Task 5 intent pattern (row settles/disappears on confirm-refetch). Vitest: row rendering per type, degraded banner, accept calls correct endpoints with correct bodies.

- [ ] Implement + tests green + lint/typecheck; commit `feat(ui): worklist — review rows, accept actions, degradation banner`.

---

### Task 9: Playwright smoke, docs, ROADMAP, gates

**Files:** `frontend/e2e/smoke.spec.ts`, `frontend/playwright.config.ts`, Makefile (`ui-smoke` target), `README.md`, `docs/ROADMAP.md`.

- [ ] **Step 1: Playwright smoke** (env-gated like the deployed smoke — its own make target, NOT in default suites): starts/expects the live test stack (backend on :8000 with test infra up, `npm run dev` or a built server), seeds one interview through the API command path (reuse the deployed-smoke seeding idiom), then: workbench nav → transcript renders seeded lines → performs a text edit → asserts the edited badge settles. Document required services in the file header.
- [ ] **Step 2: Docs** — README: `frontend/` section (dev quickstart: `make ui-dev`, typegen workflow, identity switcher note, gates); ROADMAP: M5.0 → ✅ with milestone section (one line per task), Current Phase → `M5.1 (Live workbench — real-time feed)`, stats line refreshed by controller after full gates.
- [ ] **Step 3: Gates (FOREGROUND)** — backend targeted: `./scripts/test.sh tests/ui/ tests/api/test_ui_router.py -q --no-cov`; frontend: `npm run lint && npm run typecheck && npm test` (in `frontend/`); `npm run build` (production build must succeed); Playwright smoke via `make ui-smoke`. Full Python suite is controller-only (post-task).
- [ ] **Step 4:** Commit `docs+test: M5.0 complete — Playwright smoke, README, ROADMAP`.

---

## Self-review notes (writing-plans checklist)

- **Spec coverage:** new reads→T1 (8 endpoints incl. the person-id derivation the spec's loose-coupling section demands); scaffold/identity/typegen/client→T2; workbench nav→T3; transcript display→T4; core corrections + intent pattern→T5; manual linking→T6; gallery persona/person (cards vs core views, m:n separation)→T7; actionable worklist + degradation banner→T8; smoke/docs/gates→T9. Non-goals respected (no auth, no upload, no real-time, no synthesis).
- **Contract consistency:** T1's response shapes are the single source; T3–T8 consume them via generated types (drift caught by `typegen:check`). Mutation contracts pinned to existing endpoints with read-the-router-first instructions where body shapes weren't pinned here.
- **Judgment points (deliberate):** exact component composition/styling, card interaction mechanism (owner-unpinned), and Playwright config details are implementer-authored against binding behavior specs.
