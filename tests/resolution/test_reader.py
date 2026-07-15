"""Resolution reader: project-scoped entity/speaker input rows (M4.5b)."""

import pytest

from src.resolution.reader import entity_surface_rows, speaker_rows


class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def __aiter__(self):
        self._iter = iter(self._rows)
        return self

    async def __anext__(self):
        try:
            return next(self._iter)
        except StopIteration:
            raise StopAsyncIteration


class _FakeSession:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    async def run(self, query, **params):
        self.calls.append((query, params))
        return _FakeResult(self.rows)


@pytest.mark.asyncio
async def test_entity_surface_rows_scopes_by_project():
    session = _FakeSession([{"surface": "acme corp", "entity_type": "ORG", "mentions": 3}])
    rows = await entity_surface_rows(session, "p1")
    assert rows == [{"surface": "acme corp", "entity_type": "ORG", "mentions": 3}]
    query, params = session.calls[0]
    assert params == {"project_id": "p1"}
    assert "CONTAINS_INTERVIEW" in query and "MENTIONS" in query and ":Fragment" in query


@pytest.mark.asyncio
async def test_speaker_rows_excludes_merged():
    session = _FakeSession([{
        "interview_id": "i1", "speaker_id": "s1", "display_name": "Jane Doe",
        "handle": "S1", "provisional": False,
    }])
    rows = await speaker_rows(session, "p1")
    assert rows[0]["display_name"] == "Jane Doe"
    query, _ = session.calls[0]
    assert "merged_into IS NULL" in query and "HAS_PARTICIPANT" in query
