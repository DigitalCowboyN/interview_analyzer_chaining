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
