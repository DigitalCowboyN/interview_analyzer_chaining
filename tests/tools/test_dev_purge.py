"""tools.dev.purge: read-model cleanup of test-kind projects (ADR-0030)."""
# verifies: code:tools.dev.purge

import types

import pytest

from tools.dev import purge


class FakeResult:
    def __init__(self, rows, nodes_deleted=0):
        self._rows = rows
        self._nodes_deleted = nodes_deleted

    def __aiter__(self):
        return self._aiter()

    async def _aiter(self):
        for row in self._rows:
            yield row

    async def consume(self):
        counters = types.SimpleNamespace(nodes_deleted=self._nodes_deleted)
        return types.SimpleNamespace(counters=counters)


class FakeSession:
    def __init__(self, project_ids=(), existing=()):
        self.queries, self.params = [], []
        self._project_ids = project_ids
        self._existing = existing

    async def run(self, query, **params):
        self.queries.append(query)
        self.params.append(params)
        if "project_id" in params:
            deleted = 5 if params["project_id"] in self._existing else 0
            return FakeResult([], nodes_deleted=deleted)
        return FakeResult([{"project_id": p} for p in self._project_ids])


@pytest.mark.asyncio
async def test_purge_project_deletes_project_subgraph_and_interview_scoped_nodes():
    session = FakeSession(existing={"smoke-1"})
    deleted = await purge.purge_project(session, "smoke-1")
    q = session.queries[-1]
    assert "MATCH (p:Project {project_id: $project_id})" in q
    assert "x.interview_id IN iids" in q  # LensItem/Segment/Claim etc. carry interview_id
    assert "x.project_id = $project_id" in q  # CanonicalEntity/Person are project-scoped
    assert "(f)-[:HAS_ANALYSIS]->(a:Analysis)" in q  # Analysis hangs off Fragment
    assert "collect(DISTINCT a)" in q
    assert "DETACH DELETE" in q
    assert session.params[-1] == {"project_id": "smoke-1"}
    assert deleted is True


@pytest.mark.asyncio
async def test_purge_project_missing_id_returns_false():
    session = FakeSession()
    assert await purge.purge_project(session, "does-not-exist") is False


@pytest.mark.asyncio
async def test_test_project_ids_selects_only_test_kind():
    session = FakeSession(project_ids=["samples", "smoke-1", "ui-smoke-2", "real-interviews"])
    assert await purge.test_project_ids(session) == ["smoke-1", "ui-smoke-2"]


@pytest.mark.asyncio
async def test_purge_projects_returns_count():
    session = FakeSession(existing={"a", "b"})
    assert await purge.purge_projects(session, ["a", "typo", "b"]) == 2
    assert len(session.queries) == 3


@pytest.mark.asyncio
async def test_purge_projects_missing_id_counts_zero():
    session = FakeSession()
    assert await purge.purge_projects(session, ["typo"]) == 0
