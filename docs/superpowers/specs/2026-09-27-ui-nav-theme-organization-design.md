# UI redesign: navigation, theme, and project organization — design

**Status:** approved in brainstorm (2026-09-27) — awaiting spec review, then plan
**Supersedes (in part):** the nav model of `2026-07-17-m50-ui-scaffolding-design.md`
(separate Workbench / Gallery roots, each with its own project selection).
The M5.0 domain model (Speaker ≠ Person, CQRS read/write split *in the backend*),
M5.1 liveness, and M5.2 edit observability are unchanged.
**Locks:** a new ADR ("UI is project-scoped; projects carry a derived kind").

## Problem

The owner's first look at the UI with real data (2026-09-27) found it unusable:

1. **Unreadable.** `frontend/src/app/globals.css` still carries the create-next-app
   default: under `prefers-color-scheme: dark` the body background becomes `#0a0a0a`,
   while ~27 components hard-code light-theme text (`text-neutral-900` ×103 uses of
   `text-neutral-*`, plus `border-/bg-neutral-*`, `text-blue/red/amber-*`). On a
   dark-mode OS that is near-black on near-black. There are no semantic color tokens.
2. **Test debris drowns real data.** Dev Neo4j holds ~146 projects; one holds real
   (sample) content. Integration tests mint a unique project per run (correct — it
   isolates assertions and teardown) but many never tear down: 91 `smoke-*`,
   17 `smoke-persona-*`, 16 `projection-smoke-*` (all with zero interviews), plus
   `ui-smoke-*`, `live-feed-smoke-*`, `live-gallery-smoke-*`, `deployed-smoke-*`,
   `test-project`, `replay-test-project`, `partial-replay-test`. Projects carry no
   name or kind, so the UI cannot tell test from real. (No eval data lives in the
   DB — the durable eval suite exercises the knowledge graph, not the pipeline.)
3. **Navigation leaks the CQRS split.** "Workbench" and "Gallery" are separate
   top-level surfaces, each re-asking for the project. The gallery holds its
   selected project in `useState`, so Back returns to an empty gallery home.
   Breadcrumbs show raw UUID project ids.
4. **"Just lists."** The product's output — lens items (decisions, action items,
   objectives, goals, pain points, quotes) — has no interview-level view; it is
   only reachable one line at a time via `LineDetailPanel`. The project screen is
   a bare list of titles with an ISO timestamp.

## Goals / success criteria

- Readable in both light and dark OS themes (body text ≥ 4.5:1 contrast, WCAG AA).
- From the landing page: **one click** into a project shows its interview list;
  **one click** on an interview shows that interview.
- The project is always switchable from a header dropdown on every page.
- All navigation state lives in the URL: Back/Forward and reload restore exactly
  where the user was.
- Real projects are listed individually; all test runs appear as a single
  **Test runs** entry; empty interviews/projects never clutter the default view.
- Opening an interview shows its insights alongside the transcript without
  further clicks.

## Non-goals

- Auth, project creation/renaming from the UI, uploads.
- Changing how tests isolate (they keep one project per run).
- New lenses or any change to extraction.
- A visual redesign beyond tokens + layout described here (no component library).

## Design

### A. Navigation model (project-scoped, URL-driven)

```
Header:  Interview Analyzer   [ Project ▾ ]                      [identity]

/                                   project landing (cards)
/projects/:projectId                project home, tabs: Interviews* | Personas | People | Review
/projects/:projectId/personas       (tab)            ← was /gallery/personas/:projectId
/projects/:projectId/personas/:pid                   ← was /gallery/personas/:projectId/:pid
/projects/:projectId/people         (tab)            ← was /gallery/persons/:projectId
/projects/:projectId/people/:pid                     ← was /gallery/persons/:projectId/:pid
/projects/:projectId/review         (tab)            ← was /gallery/worklist?project=
/projects/:projectId/interviews/:interviewId         ← was /workbench/:projectId/:interviewId
/projects/test-runs                 the Test runs bucket (see D)
```

- **Header `ProjectSwitcher`** replaces the Workbench/Gallery nav links. It reads
  the current project from the route params (never component state); choosing a
  project navigates to `/projects/:id` (same tab if the current tab exists).
  Real projects first, then a separator, then "Test runs".
- **Tabs** are links (`/projects/:id/<tab>`), so each is a history entry.
- **Breadcrumbs** use display names (`humanizeProjectId`: `samples` → "Samples";
  test ids → "Test runs / smoke · 3f75…") and interview titles, never raw ids
  when a name exists.
- **Redirects:** `/workbench`, `/gallery` → `/`; every old deep route above →
  its new route (Next.js `redirects()` in `next.config.ts`), so existing links and
  the M5.x smoke keep working until updated.

### B. Interview page

Two columns (stacked below `lg`):

- **Left — transcript**, as today (segments, utterance grouping, line click opens
  `LineDetailPanel`), under a compact header: title, participants (from metadata),
  date, line count, live indicator.
- **Right — Insights panel.** Fetches `GET /interviews/:id/lenses/{persona,meeting_minutes}/items`
  (existing endpoint; `limit=500`) and groups by `node_type` in a fixed order:
  Decisions, Action items, Objectives, Follow-ups, Goals, Pain points, Quotes.
  Each group shows a count and its items (text, confidence as a subtle badge,
  locked marker). Groups with zero items are omitted; if no lens has run, a quiet
  "No insights yet" note. Clicking an item highlights its
  `supporting_fragment_ids` in the transcript and scrolls to the first one.
- When a line is selected, `LineDetailPanel` replaces the Insights panel; closing
  it restores Insights. The selected line and highlighted insight are in the URL
  (`?line=<fragment_id>&insight=<item_id>`) so reload/Back preserve them.

### C. Project home — Interviews tab

Rows (not a bare list) with: title, formatted date (`Sep 27, 2026`),
participants, line count, and per-type insight counts as small chips
(e.g. "22 decisions · 18 actions"). Interviews with zero lines are hidden behind a
"Show N empty" toggle. Backend: extend `interview_rows` (`src/ui/reader.py`) to
return `participants` and `insight_counts: {node_type: n}` (one extra
`OPTIONAL MATCH` over `LensItem` by `interview_id`); the response gains fields,
nothing is removed.

### D. Project organization

- **Derived kind.** One classifier, `src/ui/project_kind.py`:
  `classify(project_id) -> ("test", suite) | ("real", None)`, driven by a single
  ordered `TEST_PREFIXES` table (`deployed-smoke-`, `projection-smoke-`,
  `live-gallery-smoke-`, `live-feed-smoke-`, `ui-smoke-`, `smoke-persona-`,
  `smoke-`, plus exact ids `test-project`, `replay-test-project`,
  `partial-replay-test`). Longest prefix wins; `suite` is the prefix sans dash.
  Kind is derived at read time — no event, no stored property — so it needs no
  migration and is fixed by editing one table.
- **API.** `GET /projects` rows gain `kind` and `suite`; projects with zero
  interviews are excluded unless `?include_empty=true`. New
  `GET /test-runs/interviews` returns all interviews in test-kind projects, each
  with `project_id` and `suite` (same row shape as C).
- **UI.** Landing and switcher list `real` projects individually and one
  **Test runs** entry. `/projects/test-runs` shows only the Interviews tab, grouped
  by suite with collapsible headers; each interview links to
  `/projects/<its own test project id>/interviews/:id`. `test-runs` is a reserved
  project id (the classifier treats it as the bucket, never a real project).
- **Stop the leak.** Every integration test that creates a project tears it
  down (the implementation plan audits each file named in Problem §2).
- **Clear the backlog.** `make dev-purge-test-data` DETACH-DELETEs the Neo4j
  subgraph of every test-kind project (mirroring the existing smoke teardown
  queries). Caveat, documented in the target's help: it cleans the read model
  only; an ESDB replay would restore them — acceptable for dev.
- **Real data.** Re-ingest the four labeled samples into project `samples`
  (Layer 1 + both lenses) and remove `ledgerline-demo`; ingest
  `data/input/GMT20231026-210203_Recording.txt` into project `real-interviews`
  (speaker inference via Anthropic). Enrichment remains blocked on the OpenAI
  embedding reachability issue and is out of scope.

### E. Theme

- Semantic tokens in `globals.css` via Tailwind v4 `@theme`: `bg`, `surface`,
  `surface-raised`, `fg`, `fg-muted`, `border`, `accent`, `accent-fg`, and status
  `danger`, `warning`, `success` (each with `-fg`/`-subtle` where used).
  Light values on `:root`; dark values under `@media (prefers-color-scheme: dark)`.
- Every hard-coded `neutral/blue/red/amber/emerald/white` class in the ~27
  components is replaced by the token utilities (`text-fg`, `bg-surface`, …).
  A lint guard (ESLint `no-restricted-syntax` on className literals, or a vitest
  grep test) fails on raw palette classes so the drift cannot return.
- Body font: Geist (already loaded) instead of the Arial fallback in `globals.css`.

## Components touched

| Unit | Change |
|---|---|
| `src/ui/project_kind.py` | new — classifier + prefix table |
| `src/ui/reader.py` | `project_rows` adds kind/suite + empty filter; `interview_rows` adds participants + insight counts; new `test_run_interview_rows` |
| `src/api/routers/ui.py` | `include_empty` param; `GET /test-runs/interviews` |
| `Makefile` | `dev-purge-test-data` |
| `tests/integration/*smoke*.py`, others per audit | add teardown where missing |
| `frontend/src/app/globals.css` | tokens, light/dark |
| `frontend/src/app/{page,projects/...}` | new route tree; old `workbench/`, `gallery/` pages removed |
| `frontend/next.config.ts` | redirects |
| `frontend/src/components/AppHeader`, new `ProjectSwitcher`, `ProjectTabs`, `InsightsPanel`, `InterviewRow` | new/changed |
| all components with palette classes | token swap |
| `frontend/e2e/smoke.spec.ts` | new routes |
| `frontend/src/api/schema.d.ts` | regenerate (`npm run typegen`) |

## Error handling

- Switcher/landing: loading and error states via existing `StateGate`.
- Unknown project id in URL → "Project not found" with a link to `/`.
- Insights fetch failure degrades to an inline error in the panel only; the
  transcript still renders.
- Unknown `?line=`/`?insight=` ids are ignored (panel stays on Insights).

## Testing

- **Python unit:** `project_kind.classify` table-driven (every prefix, longest-match
  ordering, real ids, exact-id tests).
- **Python integration:** `/projects` kind/suite + empty filter; `/test-runs/interviews`;
  `interview_rows` insight counts.
- **Vitest:** `ProjectSwitcher` (reads route, navigates), `InsightsPanel` grouping/order
  and highlight callback, `InterviewRow` chips, empty-toggle, the palette-class guard.
- **Playwright smoke:** landing → project (1 click) → interview (1 click); Back
  restores the project page; switcher changes project; insights visible.
- **Visual check:** Playwright screenshots of landing, project, interview in
  `colorScheme: 'light'` and `'dark'`, reviewed before claiming done.

## Knowledge-graph check

This spec changes surfaces the knowledge graph covers: the API (`/projects` shape,
new `/test-runs/interviews`), a new `src/ui/project_kind.py` module, a new make
target, new tests, and UI routes. On implementation: regenerate `docs/api/`,
`docs/cli/index.md` (new target), `docs/code/index.md` + `docs/tests/index.md`;
the new `test_run_interview_rows` query carries a `graphq:` docstring tag like its
siblings (run the graph-queries check). The locked decision is recorded as a new
ADR (project-scoped UI + derived project kind); run `make adr-index` and
`make knowledge-check`.
