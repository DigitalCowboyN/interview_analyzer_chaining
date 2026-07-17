"""UI reader (M5.0 Task 1): query-text pins for the /ui/* read layer.

Fake-session pattern mirrors tests/ask/test_reader.py: FakeSession records
every query + params in call order; rows list in, async iteration out.
"""

import pytest

from src.ui import reader

PID = "proj-1"
IID = "iv-1"
PERSON_ID = "person-1"


class FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def __aiter__(self):
        return self._aiter()

    async def _aiter(self):
        for row in self._rows:
            yield row

    async def single(self):
        return self._rows[0] if self._rows else None


class FakeSession:
    def __init__(self, rows):
        self._rows = rows
        self.queries = []
        self.params = []

    async def run(self, query, **params):
        self.queries.append(query)
        self.params.append(params)
        return FakeResult(self._rows)

    @property
    def last_query(self):
        return self.queries[-1]

    @property
    def last_params(self):
        return self.params[-1]


@pytest.mark.asyncio
async def test_project_exists_counts_project_node():
    session = FakeSession(rows=[{"found": 1}])
    assert await reader.project_exists(session, PID) is True
    assert "MATCH (p:Project {project_id: $project_id})" in session.last_query


@pytest.mark.asyncio
async def test_project_exists_false_when_missing():
    session = FakeSession(rows=[{"found": 0}])
    assert await reader.project_exists(session, PID) is False


@pytest.mark.asyncio
async def test_interview_exists_counts_interview_node():
    session = FakeSession(rows=[{"found": 1}])
    assert await reader.interview_exists(session, IID) is True
    assert "MATCH (i:Interview {interview_id: $interview_id})" in session.last_query


@pytest.mark.asyncio
async def test_project_rows_counts_interviews_per_project():
    session = FakeSession(
        rows=[{"project_id": PID, "interview_count": 3}]
    )
    rows = await reader.project_rows(session)
    assert rows[0]["interview_count"] == 3
    q = session.last_query
    assert "(:Project)-[:CONTAINS_INTERVIEW]->" in q or "MATCH (p:Project)" in q
    assert "CONTAINS_INTERVIEW" in q


@pytest.mark.asyncio
async def test_interview_rows_scopes_to_project_and_counts_fragments():
    session = FakeSession(
        rows=[{
            "interview_id": IID, "title": "T", "created_at": "2026-01-01T00:00:00",
            "fragment_count": 5,
        }]
    )
    rows = await reader.interview_rows(session, PID)
    assert rows[0]["fragment_count"] == 5
    q = session.last_query
    assert "(:Project {project_id: $project_id})-[:CONTAINS_INTERVIEW]->" in q
    assert "HAS_SENTENCE" in q
    assert session.last_params["project_id"] == PID


@pytest.mark.asyncio
async def test_interview_header_row_returns_title_and_metadata():
    session = FakeSession(
        rows=[{"interview_id": IID, "title": "T", "metadata": {}}]
    )
    row = await reader.interview_header_row(session, IID)
    assert row["interview_id"] == IID
    q = session.last_query
    assert "MATCH (i:Interview {interview_id: $interview_id})" in q


@pytest.mark.asyncio
async def test_interview_header_row_none_when_missing():
    session = FakeSession(rows=[])
    row = await reader.interview_header_row(session, IID)
    assert row is None


@pytest.mark.asyncio
async def test_transcript_line_rows_orders_and_null_strips():
    session = FakeSession(rows=[])
    await reader.transcript_line_rows(session, IID)
    q = session.last_query
    assert "MATCH (i:Interview {interview_id: $interview_id})-[:HAS_SENTENCE]->(f:Fragment)" in q
    assert "ORDER BY f.sequence_order" in q
    assert "SPOKEN_BY" in q and "IDENTIFIED_AS" in q
    assert "PART_OF_UTTERANCE" in q
    assert "Segment" in q and "MENTIONS" in q
    assert "SUPPORTED_BY" in q and "LensItem" in q
    assert "WHERE x IS NOT NULL" in q or "WHERE x.surface IS NOT NULL" in q
    assert "f.is_edited" in q
    assert "sp.merged_into IS NULL" in q


@pytest.mark.asyncio
async def test_persona_card_rows_filters_persona_lens():
    session = FakeSession(rows=[])
    await reader.persona_card_rows(session, PID)
    q = session.last_query
    assert "(:Project {project_id: $project_id})-[:CONTAINS_INTERVIEW]->" in q
    assert "n.lens = 'persona'" in q or "n.lens = $lens" in q
    assert "sp.merged_into IS NULL" in q
    assert "IDENTIFIED_AS" in q


@pytest.mark.asyncio
async def test_persona_exists_scopes_to_project_and_persona_lens():
    session = FakeSession(rows=[{"found": 1}])
    assert await reader.persona_exists(session, PID, PERSON_ID) is True
    q = session.last_query
    assert "n.lens = 'persona'" in q or "n.lens = $lens" in q
    assert "$person_id" in q


@pytest.mark.asyncio
async def test_persona_detail_rows_carries_per_interview_provenance():
    session = FakeSession(rows=[])
    await reader.persona_detail_rows(session, PID, PERSON_ID)
    q = session.last_query
    assert "n.lens = 'persona'" in q or "n.lens = $lens" in q
    assert "n.node_type" in q
    assert "i.interview_id" in q and "i.title" in q


@pytest.mark.asyncio
async def test_person_card_rows_scopes_to_project_and_filters_merged():
    session = FakeSession(rows=[])
    await reader.person_card_rows(session, PID)
    q = session.last_query
    assert "(:Project {project_id: $project_id})-[:CONTAINS_INTERVIEW]->" in q
    assert "sp.merged_into IS NULL" in q
    assert "IDENTIFIED_AS" in q


@pytest.mark.asyncio
async def test_person_exists_scopes_to_project():
    session = FakeSession(rows=[{"found": 1}])
    assert await reader.person_exists(session, PID, PERSON_ID) is True
    q = session.last_query
    assert "$person_id" in q
    assert "(:Project {project_id: $project_id})-[:CONTAINS_INTERVIEW]->" in q


@pytest.mark.asyncio
async def test_person_detail_rows_carries_speaker_links():
    session = FakeSession(rows=[])
    await reader.person_detail_rows(session, PID, PERSON_ID)
    q = session.last_query
    assert "sp.merged_into IS NULL" in q
    assert "i.interview_id" in q and "i.title" in q
    assert "sp.speaker_id" in q and "sp.display_name" in q


@pytest.mark.asyncio
async def test_person_contributes_to_persona_checks_persona_lens_items():
    session = FakeSession(rows=[{"found": 1}])
    assert await reader.person_contributes_to_persona(session, PID, PERSON_ID) is True
    q = session.last_query
    assert "n.lens = 'persona'" in q or "n.lens = $lens" in q
