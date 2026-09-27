# UI redesign (navigation, theme, project organization) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the UI readable in light and dark themes, project-scoped with URL-driven navigation (project → interview in one click each), insight-first on the interview page, and free of test-run clutter.

**Architecture:** Backend gains a read-time project classifier (`src/ui/project_kind.py`) and richer `/ui` rows (kind/suite, participants, insight counts, a test-runs listing). The Next.js app replaces the `/workbench` + `/gallery` trees with one `/projects/:id/...` tree, a header project switcher, semantic color tokens, and an Insights panel fed by the existing lens-items endpoint. A dev purge tool plus a shared integration fixture stop and clear test debris.

**Tech Stack:** FastAPI + Neo4j (Python 3.10, pytest, FakeSession reader tests), Next.js 15 App Router + React 19 + TanStack Query 5 + Tailwind v4, Vitest + Testing Library, Playwright.

**Spec:** `docs/superpowers/specs/2026-09-27-ui-nav-theme-organization-design.md` (ADR-0030)

## Global Constraints

- Python via `~/.pyenv/versions/3.10.7/bin/python` (or `make`, which is pyenv-aware); source `.env` with `set -a; source .env; set +a` before running tests or CLIs.
- Host-run CLIs need `export ESDB_CONNECTION_STRING='esdb://localhost:2113?tls=false'` (`.env` points at the docker-internal hostname).
- Frontend commands run from `frontend/`: `npx vitest run <path>`, `npm run typecheck`, `npm run lint`, `npm run typegen`.
- Frontend layering rule (from `src/api/client.ts`): `api → hooks → components → routes`, one direction; components never call `fetch`.
- No raw Tailwind palette classes (`neutral|gray|zinc|slate|red|green|amber|yellow|blue|emerald|sky-NNN`, `text-white`, `bg-white`, `text-black`, `bg-black`) in `frontend/src/**/*.tsx` after Task 4.
- Body text contrast ≥ 4.5:1 (WCAG AA) in both themes.
- Old routes (`/workbench…`, `/gallery…`) must redirect, never 404.
- `test-runs` is a reserved project id.
- Every new Cypher reader function carries a `graphq:` docstring tag like its siblings.
- Commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

- **Project ids with URL-meaningful characters** (`/`, spaces, `%`) — every route builder must `encodeURIComponent`; pinned in Task 5's `routes.test.ts`.
- **A project URL for an id the switcher doesn't list** (empty project, typo, stale bookmark) — header shows the raw id, page shows "Project not found" with a link home, no crash; pinned in Task 6 (switcher) and Task 5 (project page 404 state).
- **Unparseable or microsecond `created_at` strings** (`toString()` of Neo4j datetimes has 6 fractional digits) — `formatDate` returns a readable date or the raw string, never `Invalid Date`; pinned in Task 7.
- **Stale `?line=` / `?insight=` query params** after an edit deletes a line or a lens re-run replaces items — ignored, panel falls back to Insights; pinned in Task 8.
- **Lens node types outside the fixed order** (e.g. persona `Trait`) — shown in a trailing group labeled by raw type, never dropped; pinned in Task 8 (`groupInsights`).

---

## File Structure

**Backend**
- Create `src/ui/project_kind.py` — `classify(project_id) -> ProjectKind`; the one prefix table.
- Modify `src/ui/reader.py` — `project_rows` (kind/suite, empty filter), `interview_rows` (participants, insight counts), new `test_run_interview_rows`.
- Modify `src/api/routers/ui.py` — `include_empty` param on `/ui/projects`; new `GET /ui/test-runs/interviews`.
- Create `tools/dev/__init__.py`, `tools/dev/purge.py` — `purge_project(session, project_id)`, CLI for `--test-projects` / `--project`.
- Modify `Makefile` — `dev-purge-test-data`.
- Modify `tests/integration/conftest.py` — `isolated_project_id` factory fixture.
- Modify leaking tests: `tests/integration/test_{ask_smoke,end_to_end_smoke,layer1_projection_smoke,layer2_enrichment_smoke,layer3_lens_smoke,layer4_resolution_smoke,layer5_export_smoke}.py`.
- Tests: `tests/ui/test_project_kind.py`, `tests/ui/test_reader.py`, `tests/api/test_ui_router.py`, `tests/tools/test_dev_purge.py`.

**Frontend** (`frontend/src/`)
- `app/globals.css` — tokens.
- `lib/routes.ts` — every app URL builder (single source; replaces scattered template strings).
- `lib/projectName.ts` — `displayProjectName`.
- `lib/formatDate.ts` — `formatDate`.
- `lib/insights.ts` — `INSIGHT_TYPES`, `groupInsights`.
- `hooks/useProjects.ts` (extend types), `hooks/useInterviews.ts` (extend types), new `hooks/useTestRunInterviews.ts`, new `hooks/useInsights.ts`, `hooks/queryKeys.ts`.
- Components: new `ProjectSwitcher.tsx`, `ProjectTabs.tsx`, `ProjectCard.tsx`, `InterviewRow.tsx`, `InterviewHeader.tsx`, `InsightsPanel.tsx`; modified `AppHeader.tsx`, `TranscriptLine.tsx`, `WorklistRows.tsx`, `PersonaCardGrid.tsx`, `PersonCardGrid.tsx`, `PersonCoreView.tsx`; removed `ProjectList.tsx`, `InterviewList.tsx`, `MetadataPanel.tsx` (+ their tests).
- Routes: `app/page.tsx` (landing), `app/projects/[projectId]/{layout,page}.tsx`, `.../interviews/[interviewId]/page.tsx`, `.../personas/page.tsx`, `.../personas/[personId]/page.tsx`, `.../people/page.tsx`, `.../people/[personId]/page.tsx`, `.../review/page.tsx`, `app/projects/test-runs/page.tsx`; delete `app/workbench/**`, `app/gallery/**`.
- `next.config.ts` — redirects.
- `e2e/smoke.spec.ts` (routes), new `e2e/screenshots.spec.ts`, `playwright.config.ts` (testMatch).
- Guard test: `src/__tests__/palette-guard.test.ts`.

---

### Task 1: Project kind classifier

**Files:**
- Create: `src/ui/project_kind.py`
- Test: `tests/ui/test_project_kind.py`

**Interfaces:**
- Produces: `ProjectKind` (frozen dataclass: `kind: Literal["real","test"]`, `suite: Optional[str]`), `classify(project_id: str) -> ProjectKind`, `TEST_RUNS_ID = "test-runs"`, `TEST_PREFIXES: tuple[str, ...]`, `TEST_EXACT_IDS: frozenset[str]`.

- [ ] **Step 1: Write the failing test**

```python
"""project_kind (ADR-0030): read-time real/test classification of project ids."""

import pytest

from src.ui.project_kind import TEST_RUNS_ID, ProjectKind, classify


@pytest.mark.parametrize(
    "project_id, suite",
    [
        ("deployed-smoke-f4ee7682", "deployed-smoke"),
        ("projection-smoke-c779", "projection-smoke"),
        ("live-gallery-smoke-8f42", "live-gallery-smoke"),
        ("live-feed-smoke-54eb", "live-feed-smoke"),
        ("ui-smoke-1505", "ui-smoke"),
        ("smoke-persona-3885", "smoke-persona"),
        ("smoke-27bd6007", "smoke"),
        ("test-project", "test-project"),
        ("replay-test-project", "replay-test-project"),
        ("partial-replay-test", "partial-replay-test"),
    ],
)
def test_test_projects_classify_with_suite(project_id, suite):
    assert classify(project_id) == ProjectKind(kind="test", suite=suite)


def test_longest_prefix_wins_over_bare_smoke():
    # "smoke-persona-" must not be swallowed by "smoke-".
    assert classify("smoke-persona-x").suite == "smoke-persona"


@pytest.mark.parametrize(
    "project_id",
    ["samples", "real-interviews", "ledgerline-demo", "smokehouse-research", "my-test-project"],
)
def test_real_projects(project_id):
    assert classify(project_id) == ProjectKind(kind="real", suite=None)


def test_reserved_bucket_id_is_test_kind():
    assert classify(TEST_RUNS_ID).kind == "test"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `set -a; source .env; set +a; ~/.pyenv/versions/3.10.7/bin/python -m pytest tests/ui/test_project_kind.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.ui.project_kind'`

- [ ] **Step 3: Write minimal implementation**

```python
"""Read-time project classification (ADR-0030).

Projects carry no stored name or kind. Integration tests mint one project per
run under a known id prefix, so kind is derived here from the id alone — one
table, no event, no migration. A new test suite with a new prefix must be
added to TEST_PREFIXES or its runs will show up as real projects.
"""

from dataclasses import dataclass
from typing import Literal, Optional

# governed-by: ADR-0030

TEST_RUNS_ID = "test-runs"  # reserved: the UI's single bucket for all test runs

# Matched longest-first, so order here is documentation only.
TEST_PREFIXES: tuple = (
    "deployed-smoke-",
    "projection-smoke-",
    "live-gallery-smoke-",
    "live-feed-smoke-",
    "ui-smoke-",
    "smoke-persona-",
    "smoke-",
)
TEST_EXACT_IDS = frozenset(
    {"test-project", "replay-test-project", "partial-replay-test", TEST_RUNS_ID}
)
_PREFIXES_LONGEST_FIRST = tuple(sorted(TEST_PREFIXES, key=len, reverse=True))


@dataclass(frozen=True)
class ProjectKind:
    kind: Literal["real", "test"]
    suite: Optional[str]


def classify(project_id: str) -> ProjectKind:
    """Classify a project id as a real project or a test run (with its suite)."""
    if project_id in TEST_EXACT_IDS:
        return ProjectKind(kind="test", suite=project_id)
    for prefix in _PREFIXES_LONGEST_FIRST:
        if project_id.startswith(prefix):
            return ProjectKind(kind="test", suite=prefix[:-1])
    return ProjectKind(kind="real", suite=None)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `~/.pyenv/versions/3.10.7/bin/python -m pytest tests/ui/test_project_kind.py -q`
Expected: PASS (all parametrized cases)

- [ ] **Step 5: Commit**

```bash
git add src/ui/project_kind.py tests/ui/test_project_kind.py
git commit -m "feat(ui): read-time project kind classifier (ADR-0030)"
```

---

### Task 2: Richer `/ui` project + interview rows, test-runs listing

**Files:**
- Modify: `src/ui/reader.py` (`project_rows` ~L89, `interview_rows` ~L104; add `test_run_interview_rows` after `interview_rows`)
- Modify: `src/api/routers/ui.py` (`list_projects` ~L85; add route after `list_interviews`)
- Test: `tests/ui/test_reader.py`, `tests/api/test_ui_router.py`
- Regenerate: `frontend/openapi.json`, `frontend/src/api/schema.d.ts`

**Interfaces:**
- Consumes: `classify`, `TEST_PREFIXES`, `TEST_EXACT_IDS` from Task 1.
- Produces (HTTP, consumed by Tasks 5–7):
  - `GET /ui/projects?include_empty=false` → `{"projects": [{project_id, interview_count, kind: "real"|"test", suite: str|null}]}`; zero-interview projects excluded unless `include_empty=true`. Order: real first, then test, each by `project_id`.
  - `GET /ui/projects/{project_id}/interviews` → rows gain `participants: string[]` (sorted) and `insight_counts: {node_type: int}`.
  - `GET /ui/test-runs/interviews` → `{"interviews": [ {…interview row…, project_id, suite} ]}` for all test-kind projects, ordered by suite then `created_at`.
- Produces (Python): `reader.project_rows(session) -> list[dict]` (unchanged signature, rows now include `kind`, `suite`); `reader.test_run_interview_rows(session) -> list[dict]`.

- [ ] **Step 1: Write the failing reader tests** (append to `tests/ui/test_reader.py`; replace `test_project_rows_counts_interviews_per_project` and `test_interview_rows_scopes_to_project_and_counts_fragments`)

```python
@pytest.mark.asyncio
async def test_project_rows_adds_kind_and_suite():
    session = FakeSession(rows=[
        {"project_id": "smoke-abc", "interview_count": 1},
        {"project_id": "samples", "interview_count": 4},
    ])
    rows = await reader.project_rows(session)
    assert rows == [
        {"project_id": "samples", "interview_count": 4, "kind": "real", "suite": None},
        {"project_id": "smoke-abc", "interview_count": 1, "kind": "test", "suite": "smoke"},
    ]
    assert "CONTAINS_INTERVIEW" in session.last_query


@pytest.mark.asyncio
async def test_interview_rows_returns_participants_and_insight_counts():
    session = FakeSession(rows=[{
        "interview_id": IID, "title": "T", "created_at": "2026-01-01T00:00:00",
        "fragment_count": 5, "participants": ["Bob", "Alice"],
        "insight_pairs": [{"node_type": "Decision", "n": 2}, {"node_type": None, "n": 0}],
    }])
    rows = await reader.interview_rows(session, PID)
    assert rows[0]["participants"] == ["Alice", "Bob"]
    assert rows[0]["insight_counts"] == {"Decision": 2}
    assert "insight_pairs" not in rows[0]
    q = session.last_query
    assert "(:Project {project_id: $project_id})-[:CONTAINS_INTERVIEW]->" in q
    assert "HAS_PARTICIPANT" in q and "merged_into IS NULL" in q
    assert "LensItem" in q
    assert "ORDER BY created_at" in q


@pytest.mark.asyncio
async def test_test_run_interview_rows_filters_to_test_projects_and_tags_suite():
    session = FakeSession(rows=[
        {"project_id": "samples", "interview_id": "a", "title": "Real", "created_at": "2026-01-02",
         "fragment_count": 3, "participants": [], "insight_pairs": []},
        {"project_id": "ui-smoke-1", "interview_id": "b", "title": "S", "created_at": "2026-01-01",
         "fragment_count": 2, "participants": ["Alice"], "insight_pairs": []},
    ])
    rows = await reader.test_run_interview_rows(session)
    assert [r["interview_id"] for r in rows] == ["b"]
    assert rows[0]["suite"] == "ui-smoke"
    assert rows[0]["project_id"] == "ui-smoke-1"
```

- [ ] **Step 2: Run to verify they fail**

Run: `~/.pyenv/versions/3.10.7/bin/python -m pytest tests/ui/test_reader.py -q -k "project_rows or interview_rows or test_run"`
Expected: FAIL — missing `kind` key / `AttributeError: ... test_run_interview_rows`

- [ ] **Step 3: Implement in `src/ui/reader.py`**

Add `from src.ui.project_kind import classify` to the imports. Replace `project_rows` and `interview_rows`, and add the shared row shaping + `test_run_interview_rows`:

```python
async def project_rows(session) -> List[Dict[str, Any]]:
    """Every project with its interview count and derived kind (ADR-0030).
    Real projects first, then test runs; each group ordered by project_id.

    graphq: purpose=ui scope=domain-broad audience=[api]
    """
    query = """
    MATCH (p:Project)
    OPTIONAL MATCH (p)-[:CONTAINS_INTERVIEW]->(i:Interview)
    RETURN p.project_id AS project_id, count(DISTINCT i) AS interview_count
    ORDER BY p.project_id
    """
    result = await session.run(query)
    rows = []
    async for r in result:
        kind = classify(r["project_id"])
        rows.append({**dict(r), "kind": kind.kind, "suite": kind.suite})
    rows.sort(key=lambda row: (row["kind"] != "real", row["project_id"]))
    return rows


_INTERVIEW_ROW_TAIL = """
    OPTIONAL MATCH (i)-[:HAS_SENTENCE]->(f:Fragment)
    WITH p, i, count(DISTINCT f) AS fragment_count
    OPTIONAL MATCH (i)-[:HAS_PARTICIPANT]->(sp:Speaker)
    WHERE sp.merged_into IS NULL
    WITH p, i, fragment_count, collect(DISTINCT sp.display_name) AS participants
    OPTIONAL MATCH (n:LensItem {interview_id: i.interview_id})
    WITH p, i, fragment_count, participants, n.node_type AS node_type, count(n) AS n
    WITH p, i, fragment_count, participants,
         collect({node_type: node_type, n: n}) AS insight_pairs
    RETURN p.project_id AS project_id, i.interview_id AS interview_id, i.title AS title,
           toString(i.created_at) AS created_at, fragment_count, participants, insight_pairs
"""


def _shape_interview_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Fold insight_pairs into {node_type: count}; sort participants."""
    shaped = {k: v for k, v in row.items() if k != "insight_pairs"}
    shaped["participants"] = sorted(p for p in (row.get("participants") or []) if p)
    shaped["insight_counts"] = {
        pair["node_type"]: pair["n"]
        for pair in (row.get("insight_pairs") or [])
        if pair["node_type"] is not None and pair["n"]
    }
    return shaped


async def interview_rows(session, project_id: str) -> List[Dict[str, Any]]:
    """Project's interviews with fragment counts, participants, insight counts.

    graphq: purpose=ui scope=domain-broad audience=[api]
    """
    query = (
        "MATCH (p:Project {project_id: $project_id})-[:CONTAINS_INTERVIEW]->(i:Interview)"
        + _INTERVIEW_ROW_TAIL
        + "ORDER BY created_at"
    )
    result = await session.run(query, project_id=project_id)
    return [_shape_interview_row(dict(r)) async for r in result]


async def test_run_interview_rows(session) -> List[Dict[str, Any]]:
    """Interviews of every test-kind project (ADR-0030 'Test runs' bucket),
    tagged with suite; ordered by suite, then created_at.

    graphq: purpose=ui scope=domain-broad audience=[api]
    """
    query = (
        "MATCH (p:Project)-[:CONTAINS_INTERVIEW]->(i:Interview)"
        + _INTERVIEW_ROW_TAIL
        + "ORDER BY created_at"
    )
    result = await session.run(query)
    rows = []
    async for r in result:
        kind = classify(r["project_id"])
        if kind.kind == "test":
            rows.append({**_shape_interview_row(dict(r)), "suite": kind.suite})
    rows.sort(key=lambda row: (row["suite"], row["created_at"] or ""))
    return rows
```

Note: `interview_rows` now also returns `project_id` — harmless extra field.

- [ ] **Step 4: Run reader tests**

Run: `~/.pyenv/versions/3.10.7/bin/python -m pytest tests/ui/test_reader.py -q`
Expected: PASS

- [ ] **Step 5: Write the failing router tests** (in `tests/api/test_ui_router.py`, replace `test_list_projects`; add after `test_list_interviews_404_unknown_project`)

```python
def test_list_projects_excludes_empty_by_default(client):
    rows = [
        {"project_id": "samples", "interview_count": 2, "kind": "real", "suite": None},
        {"project_id": "projection-smoke-1", "interview_count": 0, "kind": "test",
         "suite": "projection-smoke"},
    ]
    with patch_session(), _MultiPatch(patch_reader(project_rows=rows)):
        resp = client.get("/ui/projects")
    assert resp.status_code == 200
    assert resp.json() == {"projects": [rows[0]]}


def test_list_projects_include_empty(client):
    rows = [{"project_id": "p0", "interview_count": 0, "kind": "real", "suite": None}]
    with patch_session(), _MultiPatch(patch_reader(project_rows=rows)):
        resp = client.get("/ui/projects", params={"include_empty": "true"})
    assert resp.json() == {"projects": rows}


def test_list_test_run_interviews(client):
    rows = [{"project_id": "smoke-1", "interview_id": IID, "title": "T",
             "created_at": "2026-01-01", "fragment_count": 2, "participants": [],
             "insight_counts": {}, "suite": "smoke"}]
    with patch_session(), _MultiPatch(patch_reader(test_run_interview_rows=rows)):
        resp = client.get("/ui/test-runs/interviews")
    assert resp.status_code == 200
    assert resp.json() == {"interviews": rows}
```

- [ ] **Step 6: Run to verify they fail**

Run: `~/.pyenv/versions/3.10.7/bin/python -m pytest tests/api/test_ui_router.py -q -k "list_projects or test_run"`
Expected: FAIL — empty project still returned; 404 on `/ui/test-runs/interviews`

- [ ] **Step 7: Implement in `src/api/routers/ui.py`**

```python
@router.get("/projects")
async def list_projects(include_empty: bool = False):
    async with await Neo4jConnectionManager.get_session() as session:
        rows = await reader.project_rows(session)
    if not include_empty:
        rows = [row for row in rows if row["interview_count"] > 0]
    return {"projects": rows}
```

Add below `list_interviews`:

```python
@router.get("/test-runs/interviews")
async def list_test_run_interviews():
    """All interviews from test-kind projects (ADR-0030 'Test runs' bucket)."""
    async with await Neo4jConnectionManager.get_session() as session:
        rows = await reader.test_run_interview_rows(session)
    return {"interviews": rows}
```

- [ ] **Step 8: Run all UI backend tests**

Run: `~/.pyenv/versions/3.10.7/bin/python -m pytest tests/ui tests/api/test_ui_router.py -q`
Expected: PASS

- [ ] **Step 9: Live-check the query against dev Neo4j** (FakeSession can't validate Cypher)

With the dev stack up (`docker compose up -d neo4j eventstore projection-service`) and the API running (`make run-api` in another shell, `.env` sourced, ESDB override exported):

```bash
curl -s localhost:8000/ui/projects | python3 -m json.tool | head -20
curl -s localhost:8000/ui/projects/ledgerline-demo/interviews | python3 -m json.tool | head -30
curl -s localhost:8000/ui/test-runs/interviews | python3 -c "import json,sys; d=json.load(sys.stdin)['interviews']; print(len(d), sorted({r['suite'] for r in d}))"
```
Expected: real projects first with `kind`; interview rows show non-empty `participants` and `insight_counts` like `{"Goal": 31, ...}`; test-runs lists several suites.

- [ ] **Step 10: Regenerate frontend types**

Run: `cd frontend && npm run typegen && npm run typecheck`
Expected: `schema.d.ts` gains `/ui/test-runs/interviews` and `include_empty`; typecheck passes.

- [ ] **Step 11: Commit**

```bash
git add src/ui/reader.py src/api/routers/ui.py tests/ui/test_reader.py tests/api/test_ui_router.py frontend/openapi.json frontend/src/api/schema.d.ts
git commit -m "feat(ui-api): project kind, interview participants + insight counts, test-runs listing"
```

---

### Task 3: Stop the test-data leak and purge the backlog

**Files:**
- Create: `tools/dev/__init__.py` (empty), `tools/dev/purge.py`
- Modify: `Makefile` (add target near `ui-smoke`), `tests/integration/conftest.py`
- Modify: `tests/integration/test_ask_smoke.py`, `test_end_to_end_smoke.py`, `test_layer1_projection_smoke.py`, `test_layer2_enrichment_smoke.py`, `test_layer3_lens_smoke.py`, `test_layer4_resolution_smoke.py`, `test_layer5_export_smoke.py`
- Test: `tests/tools/test_dev_purge.py` (create `tests/tools/__init__.py` if the dir is new)

**Interfaces:**
- Consumes: `classify` (Task 1).
- Produces: `async purge_project(session, project_id: str) -> None`; `async purge_projects(session, project_ids: Iterable[str]) -> int`; CLI `python -m tools.dev.purge (--test-projects | --project ID) [--dry-run]`; pytest fixture `isolated_project_id(prefix: str) -> str`.

ADR-0003 (projection service is the sole Neo4j writer) is not violated in spirit: like the existing smoke teardowns, this is dev/test-only read-model cleanup, never application code. Say so in the module docstring.

- [ ] **Step 1: Write the failing test**

```python
"""tools.dev.purge: read-model cleanup of test-kind projects (ADR-0030)."""

import pytest

from tools.dev import purge


class FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def __aiter__(self):
        return self._aiter()

    async def _aiter(self):
        for row in self._rows:
            yield row


class FakeSession:
    def __init__(self, project_ids=()):
        self.queries, self.params = [], []
        self._project_ids = project_ids

    async def run(self, query, **params):
        self.queries.append(query)
        self.params.append(params)
        return FakeResult([{"project_id": p} for p in self._project_ids])


@pytest.mark.asyncio
async def test_purge_project_deletes_project_subgraph_and_interview_scoped_nodes():
    session = FakeSession()
    await purge.purge_project(session, "smoke-1")
    q = session.queries[-1]
    assert "MATCH (p:Project {project_id: $project_id})" in q
    assert "x.interview_id IN iids" in q  # LensItem/Segment/Claim etc. carry interview_id
    assert "DETACH DELETE" in q
    assert session.params[-1] == {"project_id": "smoke-1"}


@pytest.mark.asyncio
async def test_test_project_ids_selects_only_test_kind():
    session = FakeSession(project_ids=["samples", "smoke-1", "ui-smoke-2", "real-interviews"])
    assert await purge.test_project_ids(session) == ["smoke-1", "ui-smoke-2"]


@pytest.mark.asyncio
async def test_purge_projects_returns_count():
    session = FakeSession()
    assert await purge.purge_projects(session, ["a", "b"]) == 2
    assert len(session.queries) == 2
```

- [ ] **Step 2: Run to verify it fails**

Run: `~/.pyenv/versions/3.10.7/bin/python -m pytest tests/tools/test_dev_purge.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'tools.dev'`

- [ ] **Step 3: Implement `tools/dev/purge.py`**

```python
"""Dev-only read-model cleanup of test-run projects (ADR-0030).

Deletes a project's Neo4j subgraph: the Project, its Interviews, their
Fragments/Speakers/Utterances, and every node carrying one of those
interview_ids (LensItem, Segment, Claim, ...). Like the integration tests'
own teardowns this is test/dev tooling, not application code, so ADR-0003's
"projection service is the sole writer" still holds for the running system.

Caveat: ESDB events are untouched — a projection replay would restore
purged projects. Acceptable for dev.

    python -m tools.dev.purge --test-projects [--dry-run]
    python -m tools.dev.purge --project ledgerline-demo
"""

import argparse
import asyncio
from typing import Iterable, List

from src.ui.project_kind import classify
from src.utils.neo4j_driver import Neo4jConnectionManager

_PURGE_QUERY = """
MATCH (p:Project {project_id: $project_id})
OPTIONAL MATCH (p)-[:CONTAINS_INTERVIEW]->(i:Interview)
WITH p, collect(DISTINCT i) AS interviews,
     [x IN collect(DISTINCT i.interview_id) WHERE x IS NOT NULL] AS iids
OPTIONAL MATCH (x) WHERE x.interview_id IN iids
WITH p, interviews, iids, collect(DISTINCT x) AS scoped
UNWIND (interviews + [null]) AS i
OPTIONAL MATCH (i)-[:HAS_SENTENCE]->(f:Fragment)
OPTIONAL MATCH (i)-[:HAS_PARTICIPANT]->(sp:Speaker)
OPTIONAL MATCH (f)-[:PART_OF_UTTERANCE]->(u:Utterance)
WITH p, interviews, scoped,
     collect(DISTINCT f) + collect(DISTINCT sp) + collect(DISTINCT u) AS owned
FOREACH (n IN scoped + owned + interviews | DETACH DELETE n)
DETACH DELETE p
"""


async def purge_project(session, project_id: str) -> None:
    """Delete one project's read-model subgraph."""
    await session.run(_PURGE_QUERY, project_id=project_id)


async def purge_projects(session, project_ids: Iterable[str]) -> int:
    count = 0
    for project_id in project_ids:
        await purge_project(session, project_id)
        count += 1
    return count


async def test_project_ids(session) -> List[str]:
    result = await session.run("MATCH (p:Project) RETURN p.project_id AS project_id")
    ids = [r["project_id"] async for r in result]
    return sorted(pid for pid in ids if classify(pid).kind == "test")


async def _main(args) -> None:
    async with await Neo4jConnectionManager.get_session() as session:
        ids = await test_project_ids(session) if args.test_projects else [args.project]
        if args.dry_run:
            print(f"would purge {len(ids)} project(s)")
            return
        print(f"purged {await purge_projects(session, ids)} project(s)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--test-projects", action="store_true")
    group.add_argument("--project")
    parser.add_argument("--dry-run", action="store_true")
    asyncio.run(_main(parser.parse_args()))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run unit tests**

Run: `~/.pyenv/versions/3.10.7/bin/python -m pytest tests/tools/test_dev_purge.py -q`
Expected: PASS

- [ ] **Step 5: Validate the Cypher against dev Neo4j on one project**

```bash
set -a; source .env; set +a
P=$NEO4J_PASSWORD
docker compose exec -T neo4j cypher-shell -u neo4j -p "$P" "MATCH (p:Project) WHERE p.project_id STARTS WITH 'ui-smoke-' RETURN p.project_id LIMIT 1"
# take that id as $ID
~/.pyenv/versions/3.10.7/bin/python -m tools.dev.purge --project "$ID"
docker compose exec -T neo4j cypher-shell -u neo4j -p "$P" "MATCH (x) WHERE x.project_id = '$ID' OR (x:Project AND x.project_id='$ID') RETURN count(x)"
```
Expected: `purged 1 project(s)`; count `0`. If Cypher errors (e.g. `UNWIND` of an empty list swallowing `p`), fix the query — the `+ [null]` guard exists so a project with zero interviews still reaches `DETACH DELETE p`; confirm on a `projection-smoke-*` project (zero interviews) too.

- [ ] **Step 6: Add the make target** (after `ui-smoke` in `Makefile`)

```make
.PHONY: dev-purge-test-data
dev-purge-test-data: ## Delete test-run projects from dev Neo4j (read model only; ESDB replay restores them)
	$(PYTHON) -m tools.dev.purge --test-projects
```

- [ ] **Step 7: Add the shared fixture** (append to `tests/integration/conftest.py`)

```python
@pytest.fixture
async def isolated_project_id():
    """Factory: mint a unique test project id under `prefix` and purge its
    read-model subgraph after the test (ADR-0030 — tests must not leak).
    `prefix` must be one of src.ui.project_kind.TEST_PREFIXES sans the dash.

    Async so teardown runs on the test's own loop (pytest.ini:
    asyncio_mode=auto, function loop scope) — the Neo4j async driver is
    loop-bound, so a sync fixture calling asyncio.run() would break."""
    import uuid

    from src.utils.neo4j_driver import Neo4jConnectionManager
    from tools.dev.purge import purge_projects

    minted = []

    def _mint(prefix: str = "smoke") -> str:
        project_id = f"{prefix}-{uuid.uuid4()}"
        minted.append(project_id)
        return project_id

    yield _mint

    if minted:
        async with await Neo4jConnectionManager.get_session() as session:
            await purge_projects(session, minted)
```

- [ ] **Step 8: Convert each leaking test**

In each file listed under **Files**, find every `f"smoke-{uuid_mod.uuid4()}"` / `f"smoke-persona-{...}"` / `f"smoke-{uuid.uuid4()}"` (grep: `grep -n 'smoke-' tests/integration/test_*smoke*.py`), add `isolated_project_id` to the test function's parameters, and replace the literal with `isolated_project_id("smoke")` or `isolated_project_id("smoke-persona")`. Example (`test_layer1_projection_smoke.py`):

```python
async def test_ingested_interview_projects_speaker_utterance_subgraph(tmp_path, isolated_project_id):
    ...
    orchestrator = IngestionOrchestrator(
        project_id=isolated_project_id("smoke"), map_dir=tmp_path / "maps"
    )
```

Leave `test_deployed_projection_smoke.py`, `test_live_feed_smoke.py`, `test_projection_ordering_smoke.py` alone — they already tear down.

- [ ] **Step 9: Run the converted integration tests**

Run: `make test-infra-up` (if not up), then `set -a; source .env; set +a; ~/.pyenv/versions/3.10.7/bin/python -m pytest tests/integration/test_layer1_projection_smoke.py tests/integration/test_layer3_lens_smoke.py -q -m integration`
Expected: same pass/fail as before this change (live-LLM tests may still fail on quota — note, don't fix). Then confirm no new `smoke-*` project remains:
`docker compose exec -T neo4j-test cypher-shell -u neo4j -p testpassword "MATCH (p:Project) WHERE p.project_id STARTS WITH 'smoke-' RETURN count(p)"` — count unchanged from before the run.

- [ ] **Step 10: Purge dev backlog**

Run: `make dev-purge-test-data`
Expected: `purged ~140 project(s)`; `curl -s localhost:8000/ui/projects` then lists only real projects.

- [ ] **Step 11: Commit**

```bash
git add tools/dev tests/tools Makefile tests/integration
git commit -m "feat(dev): purge test-run projects; integration tests tear down their projects"
```

---

### Task 4: Semantic color tokens + palette guard

**Files:**
- Modify: `frontend/src/app/globals.css`
- Create: `frontend/src/__tests__/palette-guard.test.ts`
- Modify: every `frontend/src/**/*.tsx` with raw palette classes (27 files; list with the grep in Step 3)

**Interfaces:**
- Produces Tailwind utilities used by every later task: `bg-bg`, `bg-surface`, `bg-surface-raised`, `text-fg`, `text-fg-muted`, `border-border`, `divide-border`, `text-accent`, `bg-accent`, `text-accent-fg`, `bg-accent-subtle`, `text-danger`, `bg-danger-subtle`, `text-warning`, `bg-warning-subtle`, `border-warning`, `text-success`, `bg-success-subtle`, `bg-highlight`.

- [ ] **Step 1: Write the failing guard test**

```ts
import { describe, it, expect } from "vitest";
import { readdirSync, readFileSync, statSync } from "node:fs";
import path from "node:path";

// ADR-0030 / spec §E: colors come from semantic tokens only, so both themes
// stay readable. Raw palette classes are how dark mode broke before.
const RAW_PALETTE =
  /\b(?:text|bg|border|divide|ring|from|to|via)-(?:neutral|gray|zinc|slate|stone|red|green|amber|yellow|blue|emerald|sky|indigo)-\d{2,3}\b|\b(?:text|bg)-(?:white|black)\b/g;

function tsxFiles(dir: string): string[] {
  return readdirSync(dir).flatMap((name) => {
    const full = path.join(dir, name);
    if (statSync(full).isDirectory()) return name === "__tests__" ? [] : tsxFiles(full);
    return full.endsWith(".tsx") ? [full] : [];
  });
}

describe("palette guard", () => {
  it("no component uses raw Tailwind palette classes", () => {
    const srcDir = path.resolve(__dirname, "..");
    const offenders = tsxFiles(srcDir).flatMap((file) => {
      const hits = readFileSync(file, "utf8").match(RAW_PALETTE) ?? [];
      return hits.map((hit) => `${path.relative(srcDir, file)}: ${hit}`);
    });
    expect(offenders).toEqual([]);
  });
});
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd frontend && npx vitest run src/__tests__/palette-guard.test.ts`
Expected: FAIL listing ~190 offenders.

- [ ] **Step 3: Replace `globals.css`**

```css
@import "tailwindcss";

/* Semantic color tokens (spec §E). Components use ONLY these utilities —
   src/__tests__/palette-guard.test.ts fails on raw palette classes. */
:root {
  --bg: #f6f7f9;
  --surface: #ffffff;
  --surface-raised: #eef0f3;
  --fg: #16181d;
  --fg-muted: #525866;
  --border: #dde1e7;
  --accent: #2458d6;
  --accent-fg: #ffffff;
  --accent-subtle: #e3ebfd;
  --danger: #b42318;
  --danger-subtle: #fdecea;
  --warning: #8a4b00;
  --warning-subtle: #fdf1dc;
  --success: #067647;
  --success-subtle: #e3f6ec;
  --highlight: #fff4c2;
}

@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0f1115;
    --surface: #171a20;
    --surface-raised: #20242c;
    --fg: #e8eaee;
    --fg-muted: #a3aab6;
    --border: #2d323c;
    --accent: #7aa7ff;
    --accent-fg: #0b1220;
    --accent-subtle: #1b2a47;
    --danger: #ff8a80;
    --danger-subtle: #3a1614;
    --warning: #f5c26b;
    --warning-subtle: #3a2a0c;
    --success: #5fd4a0;
    --success-subtle: #0f2e22;
    --highlight: #3d3514;
  }
}

@theme inline {
  --color-bg: var(--bg);
  --color-surface: var(--surface);
  --color-surface-raised: var(--surface-raised);
  --color-fg: var(--fg);
  --color-fg-muted: var(--fg-muted);
  --color-border: var(--border);
  --color-accent: var(--accent);
  --color-accent-fg: var(--accent-fg);
  --color-accent-subtle: var(--accent-subtle);
  --color-danger: var(--danger);
  --color-danger-subtle: var(--danger-subtle);
  --color-warning: var(--warning);
  --color-warning-subtle: var(--warning-subtle);
  --color-success: var(--success);
  --color-success-subtle: var(--success-subtle);
  --color-highlight: var(--highlight);
  --font-sans: var(--font-geist-sans);
  --font-mono: var(--font-geist-mono);
}

body {
  background: var(--bg);
  color: var(--fg);
  font-family: var(--font-geist-sans), system-ui, sans-serif;
}
```

Contrast (fg on surface / fg-muted on surface / accent on surface): light ≈ 17:1 / 7.3:1 / 6.0:1; dark ≈ 15:1 / 7.6:1 / 7.4:1 — all ≥ 4.5:1.

- [ ] **Step 4: Swap classes file by file**

List targets: `grep -rlE '(neutral|gray|red|green|amber|blue|emerald)-[0-9]|(text|bg)-(white|black)\b' frontend/src --include='*.tsx' | grep -v __tests__`

Apply this mapping (keep any `hover:`/`focus:` prefix):

| Raw | Token |
|---|---|
| `text-neutral-{700,800,900}` | `text-fg` |
| `text-neutral-{300,400,500,600}` | `text-fg-muted` |
| `border-neutral-*`, `divide-neutral-*` | `border-border`, `divide-border` |
| `bg-neutral-{50,100,200}` | `bg-surface-raised` |
| `bg-neutral-{700,800,900}` (primary button) | `bg-accent` |
| `bg-white` | `bg-surface` |
| `text-white` on an accent/primary bg | `text-accent-fg` |
| `text-blue-*` | `text-accent` |
| `bg-blue-{500,600,700}` | `bg-accent` |
| `bg-blue-{50,100}` | `bg-accent-subtle` |
| `text-red-*` | `text-danger` |
| `bg-red-{50,100}` | `bg-danger-subtle` |
| `text-amber-*` | `text-warning` |
| `bg-amber-{50,100}` | `bg-warning-subtle` |
| `border-amber-*` | `border-warning` |
| `bg-emerald-*` | `bg-success` |

For each `text-white` hit, confirm it sits on `bg-accent` (or `bg-danger`/`bg-success`); if on `bg-danger` use `text-surface`. Also give card/panel containers a `bg-surface` where they currently rely on the page background (e.g. `LineDetailPanel` root, `StateGate` error box) so they read as raised surfaces.

- [ ] **Step 5: Run guard + full vitest**

Run: `cd frontend && npx vitest run`
Expected: guard PASS. If a component test asserted a raw class (e.g. `toHaveClass("border-neutral-300")`), update the assertion to the mapped token class — do not weaken it to a presence check.

- [ ] **Step 6: Typecheck + lint**

Run: `cd frontend && npm run typecheck && npm run lint`
Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add frontend/src
git commit -m "feat(ui): semantic color tokens for light/dark; guard against raw palette classes"
```

---

### Task 5: Project-scoped route tree, route builders, redirects

**Files:**
- Create: `frontend/src/lib/routes.ts`, `frontend/src/lib/__tests__/routes.test.ts`
- Create: `frontend/src/lib/projectName.ts`, `frontend/src/lib/__tests__/projectName.test.ts`
- Create: `frontend/src/components/ProjectTabs.tsx`, `frontend/src/components/ProjectCard.tsx` (+ `__tests__`)
- Create routes: `app/projects/[projectId]/layout.tsx`, `app/projects/[projectId]/page.tsx`, `app/projects/[projectId]/interviews/[interviewId]/page.tsx`, `app/projects/[projectId]/personas/page.tsx`, `app/projects/[projectId]/personas/[personId]/page.tsx`, `app/projects/[projectId]/people/page.tsx`, `app/projects/[projectId]/people/[personId]/page.tsx`, `app/projects/[projectId]/review/page.tsx`
- Modify: `app/page.tsx` (landing), `next.config.ts`, `hooks/useProjects.ts`, `components/{WorklistRows,PersonaCardGrid,PersonCardGrid,PersonCoreView,Breadcrumbs}.tsx`
- Delete: `app/workbench/**`, `app/gallery/**`, `components/ProjectList.tsx` + its test (page tests move with their pages)

**Interfaces:**
- Consumes: `GET /ui/projects` rows `{project_id, interview_count, kind, suite}` (Task 2); token utilities (Task 4).
- Produces:
  - `routes.home()`, `routes.project(pid)`, `routes.interview(pid, iid)`, `routes.personas(pid)`, `routes.persona(pid, personId)`, `routes.people(pid)`, `routes.person(pid, personId)`, `routes.review(pid)`, `routes.testRuns()`; `TEST_RUNS_ID = "test-runs"`.
  - `displayProjectName(project: {project_id: string; kind?: "real"|"test"; suite?: string|null}) -> string`.
  - `ProjectSummary` gains `kind: "real" | "test"; suite: string | null`.
  - `useProject(projectId) -> {project: ProjectSummary | undefined, isLoading: boolean}`.

- [ ] **Step 1: Write failing lib tests**

`frontend/src/lib/__tests__/routes.test.ts`:
```ts
import { describe, it, expect } from "vitest";
import { routes, TEST_RUNS_ID } from "@/lib/routes";

describe("routes", () => {
  it("builds project-scoped paths", () => {
    expect(routes.home()).toBe("/");
    expect(routes.project("samples")).toBe("/projects/samples");
    expect(routes.interview("samples", "i1")).toBe("/projects/samples/interviews/i1");
    expect(routes.personas("p")).toBe("/projects/p/personas");
    expect(routes.persona("p", "x")).toBe("/projects/p/personas/x");
    expect(routes.people("p")).toBe("/projects/p/people");
    expect(routes.person("p", "x")).toBe("/projects/p/people/x");
    expect(routes.review("p")).toBe("/projects/p/review");
    expect(routes.testRuns()).toBe(`/projects/${TEST_RUNS_ID}`);
  });

  it("encodes URL-meaningful characters in ids", () => {
    expect(routes.interview("a/b c", "i%1")).toBe("/projects/a%2Fb%20c/interviews/i%251");
  });
});
```

`frontend/src/lib/__tests__/projectName.test.ts`:
```ts
import { describe, it, expect } from "vitest";
import { displayProjectName } from "@/lib/projectName";

describe("displayProjectName", () => {
  it("title-cases real project ids", () => {
    expect(displayProjectName({ project_id: "samples", kind: "real" })).toBe("Samples");
    expect(displayProjectName({ project_id: "real-interviews", kind: "real" })).toBe("Real Interviews");
    expect(displayProjectName({ project_id: "q3_research", kind: "real" })).toBe("Q3 Research");
  });

  it("labels test runs by suite and short id", () => {
    expect(
      displayProjectName({ project_id: "ui-smoke-150548b2-230b", kind: "test", suite: "ui-smoke" }),
    ).toBe("ui-smoke · 150548b2");
  });

  it("names the bucket", () => {
    expect(displayProjectName({ project_id: "test-runs" })).toBe("Test runs");
  });

  it("falls back to the raw id when kind is unknown", () => {
    expect(displayProjectName({ project_id: "whatever-1" })).toBe("whatever-1");
  });
});
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd frontend && npx vitest run src/lib`
Expected: FAIL — cannot resolve `@/lib/routes`

- [ ] **Step 3: Implement libs**

`frontend/src/lib/routes.ts`:
```ts
/** Every in-app URL (ADR-0030: the project is the top-level nav scope).
 * Components link via these builders only — never hand-built strings. */
const e = encodeURIComponent;

export const TEST_RUNS_ID = "test-runs";

export const routes = {
  home: () => "/",
  project: (projectId: string) => `/projects/${e(projectId)}`,
  interview: (projectId: string, interviewId: string) =>
    `/projects/${e(projectId)}/interviews/${e(interviewId)}`,
  personas: (projectId: string) => `/projects/${e(projectId)}/personas`,
  persona: (projectId: string, personId: string) =>
    `/projects/${e(projectId)}/personas/${e(personId)}`,
  people: (projectId: string) => `/projects/${e(projectId)}/people`,
  person: (projectId: string, personId: string) =>
    `/projects/${e(projectId)}/people/${e(personId)}`,
  review: (projectId: string) => `/projects/${e(projectId)}/review`,
  testRuns: () => `/projects/${TEST_RUNS_ID}`,
};
```

`frontend/src/lib/projectName.ts`:
```ts
import { TEST_RUNS_ID } from "@/lib/routes";

interface Nameable {
  project_id: string;
  kind?: "real" | "test";
  suite?: string | null;
}

/** Human label for a project: "Samples", "ui-smoke · 150548b2", "Test runs". */
export function displayProjectName(project: Nameable): string {
  if (project.project_id === TEST_RUNS_ID) return "Test runs";
  if (project.kind === "test" && project.suite) {
    const rest = project.project_id.slice(project.suite.length + 1);
    return `${project.suite} · ${rest.split("-")[0] || project.project_id}`;
  }
  if (project.kind === "real") {
    return project.project_id
      .split(/[-_]+/)
      .filter(Boolean)
      .map((word) => word[0].toUpperCase() + word.slice(1))
      .join(" ");
  }
  return project.project_id;
}
```

- [ ] **Step 4: Run lib tests**

Run: `cd frontend && npx vitest run src/lib`
Expected: PASS

- [ ] **Step 5: Extend `hooks/useProjects.ts`**

```ts
export interface ProjectSummary {
  project_id: string;
  interview_count: number;
  kind: "real" | "test";
  suite: string | null;
}

/** One project's summary from the cached projects list (undefined if unlisted). */
export function useProject(projectId: string) {
  const { data, isLoading } = useProjects();
  return { project: data?.find((p) => p.project_id === projectId), isLoading };
}
```
Update `hooks/__tests__/useProjects.test.tsx` fixtures to include `kind`/`suite`, and add a test: `useProject("b")` returns the `b` row; `useProject("zzz")` returns `undefined`.

- [ ] **Step 6: Write failing component tests**

`components/__tests__/ProjectTabs.test.tsx`:
```tsx
import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { usePathname } from "next/navigation";
import { ProjectTabs } from "@/components/ProjectTabs";

vi.mock("next/navigation", () => ({ usePathname: vi.fn() }));

describe("ProjectTabs", () => {
  it("links every tab and marks the active one", () => {
    vi.mocked(usePathname).mockReturnValue("/projects/samples/personas");
    render(<ProjectTabs projectId="samples" />);
    expect(screen.getByRole("link", { name: "Interviews" })).toHaveAttribute("href", "/projects/samples");
    expect(screen.getByRole("link", { name: "People" })).toHaveAttribute("href", "/projects/samples/people");
    expect(screen.getByRole("link", { name: "Personas" })).toHaveAttribute("aria-current", "page");
    expect(screen.getByRole("link", { name: "Interviews" })).not.toHaveAttribute("aria-current");
  });

  it("treats interview pages as the Interviews tab", () => {
    vi.mocked(usePathname).mockReturnValue("/projects/samples/interviews/i1");
    render(<ProjectTabs projectId="samples" />);
    expect(screen.getByRole("link", { name: "Interviews" })).toHaveAttribute("aria-current", "page");
  });
});
```

`components/__tests__/ProjectCard.test.tsx`:
```tsx
import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { ProjectCard } from "@/components/ProjectCard";

describe("ProjectCard", () => {
  it("shows the display name and count, linking to the project", () => {
    render(<ProjectCard href="/projects/samples" name="Samples" interviewCount={4} />);
    const link = screen.getByRole("link", { name: /Samples/ });
    expect(link).toHaveAttribute("href", "/projects/samples");
    expect(link).toHaveTextContent("4 interviews");
  });

  it("singularizes one interview", () => {
    render(<ProjectCard href="/x" name="X" interviewCount={1} />);
    expect(screen.getByRole("link")).toHaveTextContent("1 interview");
  });
});
```

- [ ] **Step 7: Run to verify they fail**

Run: `cd frontend && npx vitest run src/components/__tests__/ProjectTabs.test.tsx src/components/__tests__/ProjectCard.test.tsx`
Expected: FAIL — modules not found

- [ ] **Step 8: Implement components**

`components/ProjectTabs.tsx`:
```tsx
"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { routes } from "@/lib/routes";

/** Per-project tabs; each tab is a URL so Back/Forward walk tab history. */
export function ProjectTabs({ projectId }: { projectId: string }) {
  const pathname = usePathname() ?? "";
  const tabs = [
    { label: "Interviews", href: routes.project(projectId), match: (p: string) =>
        p === routes.project(projectId) || p.startsWith(`${routes.project(projectId)}/interviews`) },
    { label: "Personas", href: routes.personas(projectId), match: (p: string) => p.startsWith(routes.personas(projectId)) },
    { label: "People", href: routes.people(projectId), match: (p: string) => p.startsWith(routes.people(projectId)) },
    { label: "Review", href: routes.review(projectId), match: (p: string) => p.startsWith(routes.review(projectId)) },
  ];

  return (
    <nav aria-label="Project sections" className="flex gap-1 border-b border-border px-6">
      {tabs.map((tab) => {
        const active = tab.match(pathname);
        return (
          <Link
            key={tab.label}
            href={tab.href}
            aria-current={active ? "page" : undefined}
            className={`-mb-px border-b-2 px-3 py-2 text-sm ${
              active
                ? "border-accent font-medium text-fg"
                : "border-transparent text-fg-muted hover:text-fg"
            }`}
          >
            {tab.label}
          </Link>
        );
      })}
    </nav>
  );
}
```

`components/ProjectCard.tsx`:
```tsx
import Link from "next/link";

/** Landing card for one project (or the Test runs bucket). */
export function ProjectCard({
  href,
  name,
  interviewCount,
  subtle = false,
}: {
  href: string;
  name: string;
  interviewCount: number;
  subtle?: boolean;
}) {
  return (
    <Link
      href={href}
      className={`block rounded-lg border border-border p-4 hover:border-accent ${
        subtle ? "bg-surface-raised" : "bg-surface"
      }`}
    >
      <span className="block font-medium text-fg">{name}</span>
      <span className="mt-1 block text-sm text-fg-muted">
        {interviewCount} {interviewCount === 1 ? "interview" : "interviews"}
      </span>
    </Link>
  );
}
```

- [ ] **Step 9: Run component tests**

Run: `cd frontend && npx vitest run src/components/__tests__/ProjectTabs.test.tsx src/components/__tests__/ProjectCard.test.tsx`
Expected: PASS

- [ ] **Step 10: Landing page `app/page.tsx`** (replaces the redirect)

```tsx
"use client";

import { useProjects } from "@/hooks/useProjects";
import { StateGate } from "@/components/StateGate";
import { ProjectCard } from "@/components/ProjectCard";
import { displayProjectName } from "@/lib/projectName";
import { routes } from "@/lib/routes";

/** Landing: real projects as cards, all test runs as one bucket card. */
export default function HomePage() {
  const { data: projects, isLoading, isError, error } = useProjects();
  const real = projects?.filter((p) => p.kind === "real") ?? [];
  const testInterviews = (projects ?? [])
    .filter((p) => p.kind === "test")
    .reduce((sum, p) => sum + p.interview_count, 0);

  return (
    <div className="mx-auto max-w-5xl p-6">
      <h1 className="text-xl font-semibold text-fg">Projects</h1>
      <div className="mt-4">
        <StateGate
          isLoading={isLoading}
          isError={isError}
          error={error}
          isEmpty={projects?.length === 0}
          emptyFallback={<p className="p-4 text-sm text-fg-muted">No projects yet.</p>}
        >
          <ul className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {real.map((p) => (
              <li key={p.project_id}>
                <ProjectCard
                  href={routes.project(p.project_id)}
                  name={displayProjectName(p)}
                  interviewCount={p.interview_count}
                />
              </li>
            ))}
            {testInterviews > 0 && (
              <li>
                <ProjectCard href={routes.testRuns()} name="Test runs" interviewCount={testInterviews} subtle />
              </li>
            )}
          </ul>
        </StateGate>
      </div>
    </div>
  );
}
```

Test `app/__tests__/page.test.tsx` (mock `@/hooks/useProjects`): given rows `[{samples, 4, real}, {smoke-1, 2, test}, {ui-smoke-1, 1, test}]` → a "Samples" link to `/projects/samples` with "4 interviews", and a "Test runs" link to `/projects/test-runs` with "3 interviews"; no link text contains `smoke-1`. Also: no test projects → no "Test runs" card.

- [ ] **Step 11: Project layout `app/projects/[projectId]/layout.tsx`**

```tsx
"use client";

import { useParams } from "next/navigation";
import type { ReactNode } from "react";
import { ProjectTabs } from "@/components/ProjectTabs";

export default function ProjectLayout({ children }: { children: ReactNode }) {
  const { projectId } = useParams<{ projectId: string }>();
  return (
    <>
      <ProjectTabs projectId={projectId} />
      {children}
    </>
  );
}
```

(`app/projects/test-runs/page.tsx` in Task 7 is a static segment, so it takes precedence over `[projectId]` and gets no tabs — intended.)

- [ ] **Step 12: Move pages** (`git mv`, then edit)

| From | To |
|---|---|
| `app/workbench/[projectId]/page.tsx` | `app/projects/[projectId]/page.tsx` |
| `app/workbench/[projectId]/[interviewId]/page.tsx` | `app/projects/[projectId]/interviews/[interviewId]/page.tsx` |
| `app/gallery/personas/[projectId]/page.tsx` | `app/projects/[projectId]/personas/page.tsx` |
| `app/gallery/personas/[projectId]/[personId]/page.tsx` | `app/projects/[projectId]/personas/[personId]/page.tsx` |
| `app/gallery/persons/[projectId]/page.tsx` | `app/projects/[projectId]/people/page.tsx` |
| `app/gallery/persons/[projectId]/[personId]/page.tsx` | `app/projects/[projectId]/people/[personId]/page.tsx` |
| `app/gallery/worklist/page.tsx` | `app/projects/[projectId]/review/page.tsx` |

Move each `__tests__/page.test.tsx` alongside and fix its import path. Then in every moved page:
- Remove the `<Breadcrumbs>` element and its import (the header switcher + tabs now carry location; the interview page gets its own header in Task 8). Keep `LiveIndicator`, right-aligned in the page's top row.
- Review page: replace `useSearchParams().get("project")` with `useParams<{projectId: string}>()`; remove the `Suspense` wrapper and the "Select a project" branch; rename heading to "Review".
- People pages: heading "People" (was "Persons").
- Replace every hand-built href with `routes.*`: `WorklistRows.workbenchHref` → `routes.interview(projectId, interviewId)` (delete the helper); `PersonaCardGrid` → `routes.persona(...)`; `PersonCardGrid` → `routes.person(...)`; `PersonCoreView` → `routes.persona(...)`.
- Page tests: delete assertions on removed breadcrumbs; update href assertions to the new paths; the review page test mocks `useParams` instead of `useSearchParams`.

Then: `git rm -r frontend/src/app/workbench frontend/src/app/gallery frontend/src/components/ProjectList.tsx frontend/src/components/__tests__/ProjectList.test.tsx`. `Breadcrumbs` stays (Task 8 uses it); `InterviewList` is replaced in Task 7.

Project page 404: in `app/projects/[projectId]/page.tsx`, when `useInterviews` errors with `ApiError` status 404, render `<p>Project not found. <Link href={routes.home()}>All projects</Link></p>` instead of the generic error. Add a page test for it (mock `useInterviews` returning `{isError: true, error: new ApiError(404, "Project not found")}`).

- [ ] **Step 13: Redirects in `next.config.ts`** (add alongside `rewrites`)

```ts
  async redirects() {
    return [
      { source: "/workbench", destination: "/", permanent: false },
      { source: "/gallery", destination: "/", permanent: false },
      { source: "/workbench/:projectId", destination: "/projects/:projectId", permanent: false },
      {
        source: "/workbench/:projectId/:interviewId",
        destination: "/projects/:projectId/interviews/:interviewId",
        permanent: false,
      },
      { source: "/gallery/personas/:projectId", destination: "/projects/:projectId/personas", permanent: false },
      {
        source: "/gallery/personas/:projectId/:personId",
        destination: "/projects/:projectId/personas/:personId",
        permanent: false,
      },
      { source: "/gallery/persons/:projectId", destination: "/projects/:projectId/people", permanent: false },
      {
        source: "/gallery/persons/:projectId/:personId",
        destination: "/projects/:projectId/people/:personId",
        permanent: false,
      },
      {
        source: "/gallery/worklist",
        has: [{ type: "query", key: "project", value: "(?<project>.+)" }],
        destination: "/projects/:project/review",
        permanent: false,
      },
      { source: "/gallery/worklist", destination: "/", permanent: false },
    ];
  },
```

- [ ] **Step 14: Run the frontend suite**

Run: `cd frontend && npx vitest run && npm run typecheck && npm run lint`
Expected: PASS / clean.

- [ ] **Step 15: Verify redirects against the dev server** (`npm run dev` running)

```bash
for u in /workbench /gallery /workbench/samples /gallery/persons/samples "/gallery/worklist?project=samples"; do
  printf '%s -> ' "$u"; curl -s -o /dev/null -w '%{http_code} %{redirect_url}\n' "localhost:3000$u"; done
```
Expected: each `307` to the new path.

- [ ] **Step 16: Commit**

```bash
git add -A frontend
git commit -m "feat(ui): project-scoped route tree with tabs, landing cards, route builders, legacy redirects"
```

---

### Task 6: Header project switcher

**Files:**
- Create: `frontend/src/components/ProjectSwitcher.tsx`, `frontend/src/components/__tests__/ProjectSwitcher.test.tsx`
- Modify: `frontend/src/components/AppHeader.tsx`, `frontend/src/components/__tests__/AppHeader.test.tsx`

**Interfaces:**
- Consumes: `useProjects`, `displayProjectName`, `routes`, `TEST_RUNS_ID` (Task 5).
- Produces: `<ProjectSwitcher />` (no props; reads `useParams().projectId` or `usePathname()` for the test-runs bucket).

- [ ] **Step 1: Write the failing test**

```tsx
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useParams, usePathname, useRouter } from "next/navigation";
import { ProjectSwitcher } from "@/components/ProjectSwitcher";
import { useProjects } from "@/hooks/useProjects";

vi.mock("next/navigation", () => ({ useParams: vi.fn(), usePathname: vi.fn(), useRouter: vi.fn() }));
vi.mock("@/hooks/useProjects", () => ({ useProjects: vi.fn() }));

const push = vi.fn();
const projects = [
  { project_id: "samples", interview_count: 4, kind: "real", suite: null },
  { project_id: "real-interviews", interview_count: 1, kind: "real", suite: null },
  { project_id: "smoke-1", interview_count: 1, kind: "test", suite: "smoke" },
];

beforeEach(() => {
  push.mockReset();
  vi.mocked(useRouter).mockReturnValue({ push } as never);
  vi.mocked(usePathname).mockReturnValue("/projects/samples");
  vi.mocked(useParams).mockReturnValue({ projectId: "samples" });
  vi.mocked(useProjects).mockReturnValue({ data: projects, isLoading: false } as never);
});

describe("ProjectSwitcher", () => {
  it("selects the project from the URL and lists real projects plus Test runs", () => {
    render(<ProjectSwitcher />);
    const select = screen.getByRole("combobox", { name: "Project" });
    expect(select).toHaveValue("samples");
    const labels = screen.getAllByRole("option").map((o) => o.textContent);
    expect(labels).toEqual(["All projects", "Samples", "Real Interviews", "Test runs"]);
  });

  it("navigates to the chosen project", async () => {
    render(<ProjectSwitcher />);
    await userEvent.selectOptions(screen.getByRole("combobox", { name: "Project" }), "real-interviews");
    expect(push).toHaveBeenCalledWith("/projects/real-interviews");
  });

  it("navigates home for All projects and to the bucket for Test runs", async () => {
    render(<ProjectSwitcher />);
    const select = screen.getByRole("combobox", { name: "Project" });
    await userEvent.selectOptions(select, "test-runs");
    expect(push).toHaveBeenLastCalledWith("/projects/test-runs");
    await userEvent.selectOptions(select, "");
    expect(push).toHaveBeenLastCalledWith("/");
  });

  it("maps a test project's own URL to the Test runs entry", () => {
    vi.mocked(useParams).mockReturnValue({ projectId: "smoke-1" });
    render(<ProjectSwitcher />);
    expect(screen.getByRole("combobox", { name: "Project" })).toHaveValue("test-runs");
  });

  it("shows an unlisted project id as its own option instead of crashing", () => {
    vi.mocked(useParams).mockReturnValue({ projectId: "ghost" });
    render(<ProjectSwitcher />);
    expect(screen.getByRole("combobox", { name: "Project" })).toHaveValue("ghost");
    expect(screen.getByRole("option", { name: "ghost" })).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd frontend && npx vitest run src/components/__tests__/ProjectSwitcher.test.tsx`
Expected: FAIL — module not found

- [ ] **Step 3: Implement**

```tsx
"use client";

import { useParams, usePathname, useRouter } from "next/navigation";
import { useProjects } from "@/hooks/useProjects";
import { displayProjectName } from "@/lib/projectName";
import { routes, TEST_RUNS_ID } from "@/lib/routes";

/** Header project dropdown. The URL is the only source of truth for the
 * current project (ADR-0030) — no component state, so Back always agrees. */
export function ProjectSwitcher() {
  const router = useRouter();
  const pathname = usePathname() ?? "";
  const params = useParams<{ projectId?: string }>();
  const { data: projects } = useProjects();

  const real = projects?.filter((p) => p.kind === "real") ?? [];
  const hasTests = projects?.some((p) => p.kind === "test") ?? false;
  const urlProject = params?.projectId ?? (pathname === routes.testRuns() ? TEST_RUNS_ID : "");
  const urlKind = projects?.find((p) => p.project_id === urlProject)?.kind;
  const current = urlKind === "test" ? TEST_RUNS_ID : urlProject;
  const unlisted = current !== "" && current !== TEST_RUNS_ID && !real.some((p) => p.project_id === current);

  function onChange(value: string) {
    if (value === "") router.push(routes.home());
    else if (value === TEST_RUNS_ID) router.push(routes.testRuns());
    else router.push(routes.project(value));
  }

  return (
    <label className="flex items-center gap-2 text-sm text-fg-muted">
      <span className="sr-only">Project</span>
      <select
        aria-label="Project"
        value={current}
        onChange={(e) => onChange(e.target.value)}
        className="rounded-md border border-border bg-surface px-2 py-1 text-sm text-fg"
      >
        <option value="">All projects</option>
        {real.map((p) => (
          <option key={p.project_id} value={p.project_id}>
            {displayProjectName(p)}
          </option>
        ))}
        {unlisted && <option value={current}>{current}</option>}
        {(hasTests || current === TEST_RUNS_ID) && <option value={TEST_RUNS_ID}>Test runs</option>}
      </select>
    </label>
  );
}
```

- [ ] **Step 4: Run test**

Run: `cd frontend && npx vitest run src/components/__tests__/ProjectSwitcher.test.tsx`
Expected: PASS

- [ ] **Step 5: Rewrite `AppHeader.tsx`**

```tsx
"use client";

import Link from "next/link";
import { IdentitySwitcher } from "@/identity/IdentitySwitcher";
import { ProjectSwitcher } from "@/components/ProjectSwitcher";
import { routes } from "@/lib/routes";

export function AppHeader() {
  return (
    <header className="sticky top-0 z-10 flex items-center justify-between border-b border-border bg-surface px-6 py-3">
      <div className="flex items-center gap-6">
        <Link href={routes.home()} className="font-semibold text-fg">
          Interview Analyzer
        </Link>
        <ProjectSwitcher />
      </div>
      <IdentitySwitcher />
    </header>
  );
}
```

Update `AppHeader.test.tsx`: drop Workbench/Gallery link assertions; assert the "Interview Analyzer" link goes to `/` and a `combobox` named "Project" renders (mock `@/hooks/useProjects` and `next/navigation` as in Step 1).

- [ ] **Step 6: Run suite**

Run: `cd frontend && npx vitest run && npm run typecheck`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add frontend/src/components
git commit -m "feat(ui): URL-driven project switcher in the header"
```

---

### Task 7: Interview rows with insight chips; Test runs bucket page

**Files:**
- Create: `frontend/src/lib/formatDate.ts`, `frontend/src/lib/insights.ts` (types + order only; `groupInsights` lands in Task 8), `frontend/src/components/InterviewRow.tsx`, `frontend/src/hooks/useTestRunInterviews.ts`, `frontend/src/app/projects/test-runs/page.tsx` (+ tests for each)
- Modify: `frontend/src/hooks/useInterviews.ts`, `frontend/src/hooks/queryKeys.ts`, `frontend/src/app/projects/[projectId]/page.tsx`
- Delete: `frontend/src/components/InterviewList.tsx` + test

**Interfaces:**
- Consumes: `/ui/projects/{id}/interviews` rows with `participants`, `insight_counts`; `/ui/test-runs/interviews` (Task 2); `routes` (Task 5).
- Produces:
  - `formatDate(iso: string | null | undefined) -> string`.
  - `INSIGHT_TYPES: readonly {nodeType: string; label: string; chip: string}[]` in display order: Decision/"Decisions"/"decisions", ActionItem/"Action items"/"actions", Objective/"Objectives"/"objectives", FollowUp/"Follow-ups"/"follow-ups", Goal/"Goals"/"goals", PainPoint/"Pain points"/"pain points", NotableQuote/"Quotes"/"quotes".
  - `InterviewSummary` gains `participants: string[]`, `insight_counts: Record<string, number>`; `TestRunInterview = InterviewSummary & {project_id: string; suite: string}`.
  - `useTestRunInterviews()`; `queryKeys.testRunInterviews()`.
  - `<InterviewRow href interview />`.

- [ ] **Step 1: Write failing tests**

`lib/__tests__/formatDate.test.ts`:
```ts
import { describe, it, expect } from "vitest";
import { formatDate } from "@/lib/formatDate";

describe("formatDate", () => {
  it("formats ISO with microseconds (Neo4j toString) as a short date", () => {
    expect(formatDate("2026-09-27T10:12:13.623624Z")).toBe("Sep 27, 2026");
  });
  it("formats plain dates", () => {
    expect(formatDate("2026-01-05")).toBe("Jan 5, 2026");
  });
  it("returns the raw string when unparseable and empty for nullish", () => {
    expect(formatDate("not a date")).toBe("not a date");
    expect(formatDate(null)).toBe("");
    expect(formatDate(undefined)).toBe("");
  });
});
```

`components/__tests__/InterviewRow.test.tsx`:
```tsx
import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { InterviewRow } from "@/components/InterviewRow";

const base = {
  interview_id: "i1",
  title: "Weekly Planning Sync",
  created_at: "2026-09-27T10:12:13.623624Z",
  fragment_count: 81,
  participants: ["Maya Chen", "Priya Nandan"],
  insight_counts: { ActionItem: 18, Decision: 22, Trait: 3 },
};

describe("InterviewRow", () => {
  it("links to the interview and shows title, date, participants, lines", () => {
    render(<InterviewRow href="/projects/s/interviews/i1" interview={base} />);
    const link = screen.getByRole("link", { name: /Weekly Planning Sync/ });
    expect(link).toHaveAttribute("href", "/projects/s/interviews/i1");
    expect(link).toHaveTextContent("Sep 27, 2026");
    expect(link).toHaveTextContent("Maya Chen, Priya Nandan");
    expect(link).toHaveTextContent("81 lines");
  });

  it("renders insight chips in canonical order, unknown types last", () => {
    render(<InterviewRow href="/x" interview={base} />);
    const chips = screen.getAllByTestId("insight-chip").map((c) => c.textContent);
    expect(chips).toEqual(["22 decisions", "18 actions", "3 Trait"]);
  });

  it("omits chips when there are no insights", () => {
    render(<InterviewRow href="/x" interview={{ ...base, insight_counts: {} }} />);
    expect(screen.queryAllByTestId("insight-chip")).toHaveLength(0);
  });
});
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd frontend && npx vitest run src/lib/__tests__/formatDate.test.ts src/components/__tests__/InterviewRow.test.tsx`
Expected: FAIL — modules not found

- [ ] **Step 3: Implement**

`lib/formatDate.ts`:
```ts
const FORMAT = new Intl.DateTimeFormat("en-US", {
  month: "short",
  day: "numeric",
  year: "numeric",
  timeZone: "UTC",
});

/** "Sep 27, 2026". Trims >3 fractional-second digits (Neo4j emits 6), which
 * some engines refuse to parse; unparseable input comes back unchanged. */
export function formatDate(iso: string | null | undefined): string {
  if (!iso) return "";
  const normalized = iso.replace(/(\.\d{3})\d+/, "$1");
  const date = new Date(normalized);
  return Number.isNaN(date.getTime()) ? iso : FORMAT.format(date);
}
```

`lib/insights.ts`:
```ts
/** Lens node types in display order (spec §B). Types not listed here still
 * render — after these, labeled by their raw node_type. */
export const INSIGHT_TYPES = [
  { nodeType: "Decision", label: "Decisions", chip: "decisions" },
  { nodeType: "ActionItem", label: "Action items", chip: "actions" },
  { nodeType: "Objective", label: "Objectives", chip: "objectives" },
  { nodeType: "FollowUp", label: "Follow-ups", chip: "follow-ups" },
  { nodeType: "Goal", label: "Goals", chip: "goals" },
  { nodeType: "PainPoint", label: "Pain points", chip: "pain points" },
  { nodeType: "NotableQuote", label: "Quotes", chip: "quotes" },
] as const;

const KNOWN = new Map<string, { label: string; chip: string }>(
  INSIGHT_TYPES.map((t) => [t.nodeType, t]),
);

/** Node types present in `nodeTypes`, canonical ones first, then the rest alphabetically. */
export function orderNodeTypes(nodeTypes: Iterable<string>): string[] {
  const present = new Set(nodeTypes);
  const known = INSIGHT_TYPES.map((t) => t.nodeType as string).filter((t) => present.has(t));
  const unknown = [...present].filter((t) => !KNOWN.has(t)).sort();
  return [...known, ...unknown];
}

export function insightLabel(nodeType: string): string {
  return KNOWN.get(nodeType)?.label ?? nodeType;
}

export function insightChip(nodeType: string): string {
  return KNOWN.get(nodeType)?.chip ?? nodeType;
}
```

`hooks/useInterviews.ts` — extend `InterviewSummary`:
```ts
export interface InterviewSummary {
  interview_id: string;
  title: string;
  created_at: string;
  fragment_count: number;
  participants: string[];
  insight_counts: Record<string, number>;
}
```

`components/InterviewRow.tsx`:
```tsx
import Link from "next/link";
import type { InterviewSummary } from "@/hooks/useInterviews";
import { formatDate } from "@/lib/formatDate";
import { insightChip, orderNodeTypes } from "@/lib/insights";

/** One interview in a project list: identity, size, and what was extracted. */
export function InterviewRow({ href, interview }: { href: string; interview: InterviewSummary }) {
  const counts = interview.insight_counts ?? {};
  const types = orderNodeTypes(Object.keys(counts));

  return (
    <Link
      href={href}
      className="block rounded-lg border border-border bg-surface p-4 hover:border-accent"
    >
      <div className="flex items-baseline justify-between gap-4">
        <span className="font-medium text-fg">{interview.title}</span>
        <span className="shrink-0 text-sm text-fg-muted">{formatDate(interview.created_at)}</span>
      </div>
      <div className="mt-1 text-sm text-fg-muted">
        {interview.participants.length > 0 && <span>{interview.participants.join(", ")} · </span>}
        <span>
          {interview.fragment_count} {interview.fragment_count === 1 ? "line" : "lines"}
        </span>
      </div>
      {types.length > 0 && (
        <ul className="mt-3 flex flex-wrap gap-2">
          {types.map((t) => (
            <li
              key={t}
              data-testid="insight-chip"
              className="rounded-full bg-accent-subtle px-2 py-0.5 text-xs text-accent"
            >
              {counts[t]} {insightChip(t)}
            </li>
          ))}
        </ul>
      )}
    </Link>
  );
}
```

- [ ] **Step 4: Run tests**

Run: `cd frontend && npx vitest run src/lib/__tests__/formatDate.test.ts src/components/__tests__/InterviewRow.test.tsx`
Expected: PASS

- [ ] **Step 5: Project interviews page** — in `app/projects/[projectId]/page.tsx` replace `<InterviewList …/>` with:

```tsx
const [showEmpty, setShowEmpty] = useState(false);
const withLines = interviews?.filter((i) => i.fragment_count > 0) ?? [];
const empty = (interviews?.length ?? 0) - withLines.length;
const visible = showEmpty ? interviews ?? [] : withLines;
...
<ul className="space-y-3">
  {visible.map((i) => (
    <li key={i.interview_id}>
      <InterviewRow href={routes.interview(projectId, i.interview_id)} interview={i} />
    </li>
  ))}
</ul>
{empty > 0 && (
  <button type="button" onClick={() => setShowEmpty((v) => !v)} className="mt-3 text-sm text-accent">
    {showEmpty ? "Hide empty interviews" : `Show ${empty} empty`}
  </button>
)}
```
Wrap the page body in `mx-auto max-w-5xl p-6`, heading "Interviews". Page test: 2 rows with lines + 1 with `fragment_count: 0` → 2 links, a "Show 1 empty" button; clicking it → 3 links. Delete `components/InterviewList.tsx` and its test.

- [ ] **Step 6: Test-runs hook + page (failing test first)**

`hooks/queryKeys.ts`: add `testRunInterviews: () => ["test-runs", "interviews"] as const,`

`hooks/useTestRunInterviews.ts`:
```ts
import { useQuery } from "@tanstack/react-query";
import { apiGet } from "@/api/client";
import { queryKeys } from "@/hooks/queryKeys";
import type { InterviewSummary } from "@/hooks/useInterviews";

/** Row of `GET /ui/test-runs/interviews` — pinned to src/api/routers/ui.py::list_test_run_interviews. */
export type TestRunInterview = InterviewSummary & { project_id: string; suite: string };

export function useTestRunInterviews() {
  return useQuery({
    queryKey: queryKeys.testRunInterviews(),
    queryFn: async () => {
      const data = (await apiGet("/ui/test-runs/interviews")) as { interviews: TestRunInterview[] };
      return data.interviews;
    },
  });
}
```

`app/projects/test-runs/__tests__/page.test.tsx` (mock the hook): rows for suites `smoke` (2) and `ui-smoke` (1) → two `group` sections (use `<details>`; query `screen.getAllByRole("group")`) with summaries "smoke (2)" and "ui-smoke (1)"; a row link points to `/projects/smoke-1/interviews/a`.

`app/projects/test-runs/page.tsx`:
```tsx
"use client";

import { useTestRunInterviews } from "@/hooks/useTestRunInterviews";
import { StateGate } from "@/components/StateGate";
import { InterviewRow } from "@/components/InterviewRow";
import { routes } from "@/lib/routes";

/** ADR-0030 bucket: every test-run interview, grouped by suite. */
export default function TestRunsPage() {
  const { data, isLoading, isError, error } = useTestRunInterviews();
  const bySuite = new Map<string, NonNullable<typeof data>>();
  for (const row of data ?? []) bySuite.set(row.suite, [...(bySuite.get(row.suite) ?? []), row]);

  return (
    <div className="mx-auto max-w-5xl p-6">
      <h1 className="text-xl font-semibold text-fg">Test runs</h1>
      <p className="mt-1 text-sm text-fg-muted">
        Interviews created by integration and smoke tests, grouped by suite.
      </p>
      <div className="mt-4 space-y-3">
        <StateGate
          isLoading={isLoading}
          isError={isError}
          error={error}
          isEmpty={data?.length === 0}
          emptyFallback={<p className="p-4 text-sm text-fg-muted">No test runs.</p>}
        >
          {[...bySuite.entries()].map(([suite, rows]) => (
            <details key={suite} className="rounded-lg border border-border bg-surface">
              <summary className="cursor-pointer px-4 py-3 font-medium text-fg">
                {suite} ({rows.length})
              </summary>
              <ul className="space-y-3 p-4 pt-0">
                {rows.map((row) => (
                  <li key={row.interview_id}>
                    <InterviewRow href={routes.interview(row.project_id, row.interview_id)} interview={row} />
                  </li>
                ))}
              </ul>
            </details>
          ))}
        </StateGate>
      </div>
    </div>
  );
}
```

- [ ] **Step 7: Run suite**

Run: `cd frontend && npx vitest run && npm run typecheck && npm run lint`
Expected: PASS / clean.

- [ ] **Step 8: Commit**

```bash
git add -A frontend/src
git commit -m "feat(ui): interview rows with participants and insight chips; Test runs bucket"
```

---

### Task 8: Interview page — header, Insights panel, highlight, URL state

**Files:**
- Create: `frontend/src/hooks/useInsights.ts`, `frontend/src/components/InsightsPanel.tsx`, `frontend/src/components/InterviewHeader.tsx` (+ tests)
- Modify: `frontend/src/lib/insights.ts` (add `groupInsights`), `frontend/src/hooks/queryKeys.ts`, `frontend/src/components/TranscriptLine.tsx`, `frontend/src/app/projects/[projectId]/interviews/[interviewId]/page.tsx` (+ test)
- Delete: `frontend/src/components/MetadataPanel.tsx` + test

**Interfaces:**
- Consumes: `GET /interviews/{interview_id}/lenses/{lens}/items?limit=500` → `{items: [{item_id, node_type, confidence, locked, supporting_fragment_ids: string[], fields: Record<string, unknown>, …}]}` (existing, `src/api/routers/queries.py`); `useTranscript`, `useInterviews`, `useProject`, `displayProjectName`, `routes`, `formatDate`, `orderNodeTypes`, `insightLabel`.
- Produces:
  - `Insight = {item_id: string; node_type: string; lens: string; text: string; confidence: number; locked: boolean; supporting_fragment_ids: string[]}`.
  - `useInsights(interviewId) -> UseQueryResult<Insight[]>`; `queryKeys.insights(interviewId)`.
  - `groupInsights(items: Insight[]) -> {nodeType: string; label: string; items: Insight[]}[]`.
  - `<InsightsPanel insights selectedId onSelect(insight) />`, `<InterviewHeader title participants date lineCount metadata />`.
  - `TranscriptLine` gains `highlighted?: boolean` and renders `id={"line-" + fragment_id}`.

- [ ] **Step 1: Write failing tests**

Append to `lib/__tests__/insights.test.ts` (create):
```ts
import { describe, it, expect } from "vitest";
import { groupInsights, type Insight } from "@/lib/insights";

const mk = (item_id: string, node_type: string): Insight => ({
  item_id, node_type, lens: "persona", text: item_id, confidence: 0.9, locked: false,
  supporting_fragment_ids: [],
});

describe("groupInsights", () => {
  it("groups in canonical order and keeps unknown types in a trailing group", () => {
    const groups = groupInsights([mk("q", "NotableQuote"), mk("t", "Trait"), mk("d", "Decision"), mk("d2", "Decision")]);
    expect(groups.map((g) => [g.label, g.items.length])).toEqual([
      ["Decisions", 2], ["Quotes", 1], ["Trait", 1],
    ]);
  });

  it("returns no groups for no items", () => {
    expect(groupInsights([])).toEqual([]);
  });
});
```

`components/__tests__/InsightsPanel.test.tsx`:
```tsx
import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { InsightsPanel } from "@/components/InsightsPanel";
import type { Insight } from "@/lib/insights";

const items: Insight[] = [
  { item_id: "d1", node_type: "Decision", lens: "meeting_minutes", text: "Ship CSV export", confidence: 0.92, locked: true, supporting_fragment_ids: ["f1"] },
  { item_id: "a1", node_type: "ActionItem", lens: "meeting_minutes", text: "Ravi drafts spec", confidence: 0.8, locked: false, supporting_fragment_ids: ["f2"] },
];

describe("InsightsPanel", () => {
  it("renders groups with counts and items", () => {
    render(<InsightsPanel insights={items} selectedId={null} onSelect={() => {}} />);
    expect(screen.getByRole("heading", { name: "Decisions (1)" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Action items (1)" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Ship CSV export/ })).toHaveTextContent("locked");
  });

  it("calls onSelect and marks the selected item", async () => {
    const onSelect = vi.fn();
    const { rerender } = render(<InsightsPanel insights={items} selectedId={null} onSelect={onSelect} />);
    await userEvent.click(screen.getByRole("button", { name: /Ravi drafts spec/ }));
    expect(onSelect).toHaveBeenCalledWith(items[1]);
    rerender(<InsightsPanel insights={items} selectedId="a1" onSelect={onSelect} />);
    expect(screen.getByRole("button", { name: /Ravi drafts spec/ })).toHaveAttribute("aria-pressed", "true");
  });

  it("shows a quiet empty state", () => {
    render(<InsightsPanel insights={[]} selectedId={null} onSelect={() => {}} />);
    expect(screen.getByText("No insights yet.")).toBeInTheDocument();
  });
});
```

`components/__tests__/InterviewHeader.test.tsx`:
```tsx
import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { InterviewHeader } from "@/components/InterviewHeader";

describe("InterviewHeader", () => {
  it("shows title, participants, date, and line count; hides empty metadata", () => {
    render(<InterviewHeader title="Sync" participants={["Maya", "Ravi"]} date="Sep 27, 2026" lineCount={81} metadata={{}} />);
    expect(screen.getByRole("heading", { level: 1, name: "Sync" })).toBeInTheDocument();
    expect(screen.getByText("Maya, Ravi · Sep 27, 2026 · 81 lines")).toBeInTheDocument();
    expect(screen.queryByText(/No metadata/)).not.toBeInTheDocument();
  });

  it("lists metadata when present", () => {
    render(<InterviewHeader title="T" participants={[]} date="" lineCount={1} metadata={{ source: "zoom" }} />);
    expect(screen.getByText("source")).toBeInTheDocument();
    expect(screen.getByText("zoom")).toBeInTheDocument();
  });
});
```

Add to `components/__tests__/TranscriptLine.test.tsx`:
```tsx
it("anchors by fragment id and marks highlighted lines", () => {
  const { container } = render(
    <TranscriptLine line={line} continuesUtterance={false} onSelect={() => {}} highlighted />,
  );
  const button = container.querySelector(`#line-${line.fragment_id}`)!;
  expect(button).toHaveAttribute("data-highlighted", "true");
  expect(button).toHaveClass("bg-highlight");
});
```
(reuse the file's existing `line` fixture name; adjust if it differs.)

- [ ] **Step 2: Run to verify they fail**

Run: `cd frontend && npx vitest run src/lib/__tests__/insights.test.ts src/components/__tests__/InsightsPanel.test.tsx src/components/__tests__/InterviewHeader.test.tsx src/components/__tests__/TranscriptLine.test.tsx`
Expected: FAIL

- [ ] **Step 3: Implement**

Append to `lib/insights.ts`:
```ts
export interface Insight {
  item_id: string;
  node_type: string;
  lens: string;
  text: string;
  confidence: number;
  locked: boolean;
  supporting_fragment_ids: string[];
}

export interface InsightGroup {
  nodeType: string;
  label: string;
  items: Insight[];
}

export function groupInsights(items: Insight[]): InsightGroup[] {
  const byType = new Map<string, Insight[]>();
  for (const item of items) byType.set(item.node_type, [...(byType.get(item.node_type) ?? []), item]);
  return orderNodeTypes(byType.keys()).map((nodeType) => ({
    nodeType,
    label: insightLabel(nodeType),
    items: byType.get(nodeType)!,
  }));
}
```

`hooks/queryKeys.ts`: add `insights: (interviewId: string) => ["interviews", interviewId, "insights"] as const,` — note it shares the `["interviews", interviewId]` prefix, so the existing live invalidation of an interview (check `useLiveInvalidation` invalidates by that prefix; if it invalidates only `transcript`, add `queryKeys.insights(interviewId)` beside it) refreshes insights after a lens re-run.

`hooks/useInsights.ts`:
```ts
import { useQuery } from "@tanstack/react-query";
import { apiGet } from "@/api/client";
import { queryKeys } from "@/hooks/queryKeys";
import type { Insight } from "@/lib/insights";

const LENSES = ["meeting_minutes", "persona"] as const;

interface LensItemRow {
  item_id: string;
  node_type: string;
  confidence: number;
  locked: boolean;
  supporting_fragment_ids: string[];
  fields: Record<string, unknown>;
}

/** Every lens item for an interview across both lenses (existing
 * `GET /interviews/{id}/lenses/{lens}/items`; src/api/routers/queries.py). */
export function useInsights(interviewId: string) {
  return useQuery({
    queryKey: queryKeys.insights(interviewId),
    queryFn: async (): Promise<Insight[]> => {
      const perLens = await Promise.all(
        LENSES.map(async (lens) => {
          const data = (await apiGet("/interviews/{interview_id}/lenses/{lens}/items", {
            params: { interview_id: interviewId, lens },
            query: { limit: 500 },
          })) as { items: LensItemRow[] };
          return data.items.map((row) => ({
            item_id: row.item_id,
            node_type: row.node_type,
            lens,
            text: String(row.fields?.text ?? ""),
            confidence: row.confidence,
            locked: row.locked,
            supporting_fragment_ids: row.supporting_fragment_ids ?? [],
          }));
        }),
      );
      return perLens.flat();
    },
    enabled: Boolean(interviewId),
  });
}
```
Before relying on `fields.text`: `curl -s "localhost:8000/interviews/<iid>/lenses/persona/items?limit=1" | python3 -m json.tool` and confirm `text` sits under `fields`; if it's top-level, read `row.text` instead and adjust `LensItemRow`.

Hook test `hooks/__tests__/useInsights.test.tsx`: mock `@/api/client`'s `apiGet` to return one item per lens; assert the hook yields 2 insights with `lens` set and `text` read from `fields.text`, and that `apiGet` was called with `query: { limit: 500 }` for both lenses.

`components/InsightsPanel.tsx`:
```tsx
import { groupInsights, type Insight } from "@/lib/insights";

/** Interview-level lens output (spec §B), grouped by type in canonical order. */
export function InsightsPanel({
  insights,
  selectedId,
  onSelect,
}: {
  insights: Insight[];
  selectedId: string | null;
  onSelect: (insight: Insight) => void;
}) {
  const groups = groupInsights(insights);
  if (groups.length === 0) return <p className="p-4 text-sm text-fg-muted">No insights yet.</p>;

  return (
    <div className="space-y-5">
      {groups.map((group) => (
        <section key={group.nodeType}>
          <h2 className="text-xs font-semibold uppercase tracking-wide text-fg-muted">
            {group.label} ({group.items.length})
          </h2>
          <ul className="mt-2 space-y-1">
            {group.items.map((item) => {
              const selected = item.item_id === selectedId;
              return (
                <li key={item.item_id}>
                  <button
                    type="button"
                    aria-pressed={selected}
                    onClick={() => onSelect(item)}
                    className={`w-full rounded-md px-2 py-1.5 text-left text-sm ${
                      selected ? "bg-accent-subtle text-fg" : "text-fg hover:bg-surface-raised"
                    }`}
                  >
                    {item.text}
                    <span className="ml-2 text-xs text-fg-muted">{Math.round(item.confidence * 100)}%</span>
                    {item.locked && <span className="ml-2 text-xs text-warning">locked</span>}
                  </button>
                </li>
              );
            })}
          </ul>
        </section>
      ))}
    </div>
  );
}
```

`components/InterviewHeader.tsx`:
```tsx
import type { TranscriptMetadata } from "@/hooks/useTranscript";

/** Interview title strip. Front-matter metadata shows only when the graph has it. */
export function InterviewHeader({
  title,
  participants,
  date,
  lineCount,
  metadata,
}: {
  title: string;
  participants: string[];
  date: string;
  lineCount: number;
  metadata: TranscriptMetadata;
}) {
  const facts = [
    participants.join(", "),
    date,
    `${lineCount} ${lineCount === 1 ? "line" : "lines"}`,
  ].filter(Boolean);
  const entries = Object.entries(metadata);

  return (
    <div>
      <h1 className="text-xl font-semibold text-fg">{title}</h1>
      <p className="mt-1 text-sm text-fg-muted">{facts.join(" · ")}</p>
      {entries.length > 0 && (
        <dl className="mt-2 grid grid-cols-2 gap-x-4 gap-y-1 text-sm">
          {entries.map(([key, value]) => (
            <div key={key} className="contents">
              <dt className="text-fg-muted">{key}</dt>
              <dd className="text-fg">{String(value)}</dd>
            </div>
          ))}
        </dl>
      )}
    </div>
  );
}
```

`TranscriptLine.tsx`: add `highlighted?: boolean` to props; on the `<button>` add `id={\`line-${line.fragment_id}\`}`, `data-highlighted={highlighted ? "true" : undefined}`, and append `${highlighted ? " bg-highlight" : ""}` to its className.

- [ ] **Step 4: Run tests**

Run: `cd frontend && npx vitest run src/lib src/components src/hooks`
Expected: PASS

- [ ] **Step 5: Rewrite the interview page** `app/projects/[projectId]/interviews/[interviewId]/page.tsx`

Keep the existing transcript rendering loop (segments, utterance grouping) verbatim; change the frame:

```tsx
"use client";

import { useParams, usePathname, useRouter, useSearchParams } from "next/navigation";
import { Suspense, useMemo } from "react";
// ...existing imports minus MetadataPanel...
import { Breadcrumbs } from "@/components/Breadcrumbs";
import { InterviewHeader } from "@/components/InterviewHeader";
import { InsightsPanel } from "@/components/InsightsPanel";
import { useInsights } from "@/hooks/useInsights";
import { useInterviews } from "@/hooks/useInterviews";
import { useProject } from "@/hooks/useProjects";
import { displayProjectName } from "@/lib/projectName";
import { formatDate } from "@/lib/formatDate";
import { routes } from "@/lib/routes";
import type { Insight } from "@/lib/insights";

function TranscriptPageContent() {
  const { projectId, interviewId } = useParams<{ projectId: string; interviewId: string }>();
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const { data: transcript, isLoading, isError, error } = useTranscript(interviewId);
  const insightsQuery = useInsights(interviewId);
  const { data: interviews } = useInterviews(projectId);
  const { project } = useProject(projectId);
  const liveStatus = useLiveInvalidation({ interviewId, projectId });

  // URL is the state (reload/Back restore it); replace() so line clicks don't
  // pile up history entries — Back leaves the interview.
  function setParam(key: "line" | "insight", value: string | null) {
    const next = new URLSearchParams(searchParams.toString());
    if (value) next.set(key, value);
    else next.delete(key);
    const qs = next.toString();
    router.replace(qs ? `${pathname}?${qs}` : pathname, { scroll: false });
  }

  const selectedLine =
    transcript?.lines.find((l) => l.fragment_id === searchParams.get("line")) ?? null;
  const insights = insightsQuery.data ?? [];
  const selectedInsight = insights.find((i) => i.item_id === searchParams.get("insight")) ?? null;
  const highlighted = useMemo(
    () => new Set(selectedInsight?.supporting_fragment_ids ?? []),
    [selectedInsight],
  );
  const participants = useMemo(
    () => [...new Set(transcript?.lines.map((l) => l.speaker?.display_name).filter(Boolean) as string[])],
    [transcript],
  );
  const summary = interviews?.find((i) => i.interview_id === interviewId);

  function onSelectInsight(insight: Insight) {
    setParam("insight", insight.item_id);
    const first = insight.supporting_fragment_ids[0];
    if (first) document.getElementById(`line-${first}`)?.scrollIntoView?.({ block: "center", behavior: "smooth" });
  }

  return (
    <div className="mx-auto grid max-w-7xl gap-6 p-6 lg:grid-cols-[minmax(0,1fr)_24rem]">
      <div className="min-w-0">
        <div className="flex items-center justify-between">
          <Breadcrumbs
            items={[
              { label: project ? displayProjectName(project) : projectId, href: routes.project(projectId) },
              { label: transcript?.title ?? "Interview" },
            ]}
          />
          <LiveIndicator status={liveStatus} />
        </div>
        <StateGate /* existing props */>
          {transcript && (
            <>
              <InterviewHeader
                title={transcript.title}
                participants={participants}
                date={formatDate(summary?.created_at)}
                lineCount={transcript.lines.length}
                metadata={transcript.metadata}
              />
              <div className="mt-4">
                {/* existing lines.map(...) — pass these two props to TranscriptLine: */}
                {/*   highlighted={highlighted.has(line.fragment_id)}                  */}
                {/*   onSelect={(l) => setParam("line", l.fragment_id)}                */}
              </div>
            </>
          )}
        </StateGate>
      </div>
      <aside className="lg:sticky lg:top-20 lg:max-h-[calc(100vh-6rem)] lg:overflow-y-auto">
        {selectedLine ? (
          <LineDetailPanel
            projectId={projectId}
            interviewId={interviewId}
            line={selectedLine}
            onClose={() => setParam("line", null)}
          />
        ) : (
          <div className="rounded-lg border border-border bg-surface p-4">
            <h2 className="mb-3 font-semibold text-fg">Insights</h2>
            {insightsQuery.isError ? (
              <p className="text-sm text-danger">Couldn’t load insights.</p>
            ) : (
              <InsightsPanel insights={insights} selectedId={selectedInsight?.item_id ?? null} onSelect={onSelectInsight} />
            )}
          </div>
        )}
      </aside>
    </div>
  );
}

export default function TranscriptPage() {
  return (
    <Suspense fallback={<div className="p-6 text-sm text-fg-muted">Loading…</div>}>
      <TranscriptPageContent />
    </Suspense>
  );
}
```

If `LineDetailPanel`'s root uses fixed positioning/width for its old side-sibling placement, change it to fill its container (`w-full`), keeping `role="dialog"` and `aria-label="Line detail"` (the e2e smoke queries them). Delete `MetadataPanel.tsx` and its test.

- [ ] **Step 6: Update the page test** (`…/interviews/[interviewId]/__tests__/page.test.tsx`)

Mock `next/navigation` (`useParams`, `usePathname` → `/projects/p1/interviews/i1`, `useRouter` → `{replace}`, `useSearchParams` → `new URLSearchParams(<per test>)`), and `@/hooks/useInsights`, `@/hooks/useInterviews`, `@/hooks/useProjects`. Cases:
1. Renders the Insights panel with a group heading when no `line` param.
2. `?line=f1` → Line detail dialog replaces Insights.
3. `?insight=d1` where `d1.supporting_fragment_ids = ["f1"]` → the `#line-f1` button has `data-highlighted="true"`, `#line-f2` doesn't.
4. **Stale params:** `?line=gone&insight=gone` → Insights panel shows, no dialog, no highlighted lines, no throw.
5. Clicking a transcript line calls `replace` with `/projects/p1/interviews/i1?line=<fragment_id>`.
6. Insights fetch error → "Couldn’t load insights." while transcript lines still render.

- [ ] **Step 7: Run suite**

Run: `cd frontend && npx vitest run && npm run typecheck && npm run lint`
Expected: PASS / clean.

- [ ] **Step 8: Commit**

```bash
git add -A frontend/src
git commit -m "feat(ui): insight-first interview page with highlight and URL-held selection"
```

---

### Task 9: Real data, e2e + visual verification, knowledge docs

**Files:**
- Modify: `frontend/e2e/smoke.spec.ts`, `frontend/playwright.config.ts`
- Create: `frontend/e2e/screenshots.spec.ts`
- Modify: `Makefile` (`ui-screenshots` target)
- Regenerate: `docs/api/`, `docs/cli/index.md`, `docs/code/index.md`, `docs/tests/index.md`, `docs/graph/{index,graph}.md`, ADR index
- Modify: `docs/adr/0030-…md` only if `adr-check` asks for markers; add `# governed-by: ADR-0030` comment to `frontend/src/lib/routes.ts` header

**Interfaces:**
- Consumes: everything above; dev stack; `.env`.

- [ ] **Step 1: Load real data** (sequential lens runs — parallel runs on one interview conflict on ESDB stream position)

```bash
set -a; source .env; set +a; export ESDB_CONNECTION_STRING='esdb://localhost:2113?tls=false'
PY=~/.pyenv/versions/3.10.7/bin/python
for f in user_interview_mature team_meeting_mature stakeholder_review_mixed focus_group_mixed; do
  $PY -m src.ingestion data/samples/$f.txt --project-id samples | tail -1; done
$PY -m src.ingestion data/input/GMT20231026-210203_Recording.txt --project-id real-interviews | tail -1
sleep 5; curl -s localhost:8000/ui/projects/samples/interviews | python3 -c "import json,sys; [print(r['interview_id'], r['title']) for r in json.load(sys.stdin)['interviews']]"
```
Then per interview id: `meeting_minutes` for "Weekly Planning Sync" and "Q1 Stakeholder Review"; `persona` for the user interview, focus group, stakeholder review, and the real-interviews interview — **one command at a time**: `$PY -m src.lens <iid> <lens>`.
Finally remove the old demo project: `$PY -m tools.dev.purge --project ledgerline-demo`.
Expected: `/ui/projects` lists `real-interviews` and `samples` (plus the Test runs bucket only if tests have since run).

- [ ] **Step 2: Update the e2e smoke to the new routes**

In `frontend/e2e/smoke.spec.ts` replace the nav leg (≈L115–122):

```ts
  await page.goto(`/projects/${encodeURIComponent(data.project_id)}`);
  await page.getByRole("link", { name: new RegExp(data.title) }).click();
  await expect(page).toHaveURL(
    new RegExp(`/projects/${encodeURIComponent(data.project_id)}/interviews/${data.interview_id}$`),
  );
```
(The seeded `ui-smoke-*` project is a test project, so it's reached via its own URL or the Test runs bucket — not a landing card.) Replace the later `page.goto(\`/workbench/...\`)` (≈L163) with `page.goto(\`/projects/${encodeURIComponent(data.project_id)}/interviews/${data.interview_id}\`)`. Update the header comment's journey description. Line clicks now use `router.replace` with `?line=`, so the URL assertions that follow a line click must allow a query string (`/...${data.interview_id}(\\?.*)?$`).

Add a nav test after the existing one:

```ts
test("one click into a project, one into an interview, Back returns to the project", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("link", { name: /Samples/ }).click();
  await expect(page).toHaveURL(/\/projects\/samples$/);
  await page.getByRole("link", { name: /Weekly Planning Sync/ }).click();
  await expect(page.getByRole("heading", { name: /Decisions \(\d+\)/ })).toBeVisible();
  await page.goBack();
  await expect(page).toHaveURL(/\/projects\/samples$/);
  await expect(page.getByRole("combobox", { name: "Project" })).toHaveValue("samples");
});
```

- [ ] **Step 3: Screenshot spec + config**

`frontend/playwright.config.ts`: change `testMatch: "smoke.spec.ts"` to `testMatch: ["smoke.spec.ts", "screenshots.spec.ts"]`.

`frontend/e2e/screenshots.spec.ts`:
```ts
import { test, expect } from "@playwright/test";

/** Visual check (spec §Testing): landing, project, interview in both themes.
 * Needs the `samples` project loaded (plan Task 9 Step 1). Output in
 * test-results/screens/ — look at every image before calling the UI done. */
for (const colorScheme of ["light", "dark"] as const) {
  test.describe(`${colorScheme} theme`, () => {
    test.use({ colorScheme, viewport: { width: 1440, height: 900 } });

    test(`screens (${colorScheme})`, async ({ page }) => {
      await page.goto("/");
      await expect(page.getByRole("link", { name: /Samples/ })).toBeVisible();
      await page.screenshot({ path: `test-results/screens/landing-${colorScheme}.png`, fullPage: true });

      await page.getByRole("link", { name: /Samples/ }).click();
      await expect(page.getByRole("link", { name: /Weekly Planning Sync/ })).toBeVisible();
      await page.screenshot({ path: `test-results/screens/project-${colorScheme}.png`, fullPage: true });

      await page.getByRole("link", { name: /Weekly Planning Sync/ }).click();
      await expect(page.getByRole("heading", { name: /Decisions/ })).toBeVisible();
      await page.screenshot({ path: `test-results/screens/interview-${colorScheme}.png` });
    });
  });
}
```

`Makefile` (after `ui-smoke`):
```make
.PHONY: ui-screenshots
ui-screenshots: ## Playwright light/dark screenshots of landing, project, interview (needs `samples` loaded)
	cd frontend && UI_SMOKE=1 npx playwright test screenshots.spec.ts
```

- [ ] **Step 4: Run e2e + screenshots**

Run: `make ui-smoke` then `make ui-screenshots`
Expected: both pass. Then **Read each of the six PNGs** in `frontend/test-results/screens/` and check: text legible on every surface in both themes; header shows the project dropdown; interview page shows transcript left, grouped Insights right. Fix and re-run until true — do not claim done from test status alone.

- [ ] **Step 5: Full test pass**

```bash
make test-unit
cd frontend && npx vitest run && npm run typecheck && npm run lint && npm run typegen:check
```
Expected: all green (pre-existing live-LLM quota failures excepted — report them, don't hide them).

- [ ] **Step 6: Knowledge-graph upkeep**

Add `// governed-by: ADR-0030` to the top comment of `frontend/src/lib/routes.ts`. Then:

```bash
make adr-index && make adr-check
make api-check cli-check code-check tests-check graph-check knowledge-check 2>&1 | tail -30
```
Regenerate whatever each check reports out of sync (use the target each warning names, e.g. the domain's index/regenerate target listed in `make help`), until the only remaining warnings are ones that pre-date this branch (the "governed code changed after the ADR" notes for ADRs 0005/0011/0013/0020/0025–0027 and the three `code:utils*` units).

- [ ] **Step 7: Commit**

```bash
git add -A frontend Makefile docs
git commit -m "test(ui): e2e + light/dark screenshots for new nav; load samples + real-interviews; regenerate knowledge docs"
```

---

## Knowledge-graph check

Surfaces touched: `/ui` API shape (Task 2), new module `src/ui/project_kind.py` and `tools/dev/` (Tasks 1, 3), new make targets `dev-purge-test-data` and `ui-screenshots` (Tasks 3, 9), new tests, frontend routes. Task 9 Step 6 regenerates `docs/api/`, `docs/cli/index.md`, `docs/code/index.md`, `docs/tests/index.md`, `docs/graph/`, and runs `make adr-check` + `make knowledge-check`. New reader query `test_run_interview_rows` carries its `graphq:` tag. ADR-0030 records the decision (committed with the spec).
