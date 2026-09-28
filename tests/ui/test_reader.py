"""UI reader (M5.0 Task 1): query-text pins for the /ui/* read layer.

Fake-session pattern mirrors tests/ask/test_reader.py: FakeSession records
every query + params in call order; rows list in, async iteration out.
"""

import json

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
    assert "(p:Project {project_id: $project_id})-[:CONTAINS_INTERVIEW]->" in q
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


@pytest.mark.asyncio
async def test_interview_header_row_returns_title_and_metadata():
    session = FakeSession(
        rows=[{"interview_id": IID, "title": "T", "metadata_json": None}]
    )
    row = await reader.interview_header_row(session, IID)
    assert row["interview_id"] == IID
    q = session.last_query
    assert "MATCH (i:Interview {interview_id: $interview_id})" in q
    assert "metadata_json" in q


@pytest.mark.asyncio
async def test_interview_header_row_none_when_missing():
    session = FakeSession(rows=[])
    row = await reader.interview_header_row(session, IID)
    assert row is None


@pytest.mark.asyncio
async def test_interview_header_row_parses_metadata_json():
    session = FakeSession(
        rows=[{
            "interview_id": IID, "title": "T",
            "metadata_json": json.dumps({"interviewer": "Alice", "location": "Remote"}, sort_keys=True),
        }]
    )
    row = await reader.interview_header_row(session, IID)
    assert row["metadata"] == {"interviewer": "Alice", "location": "Remote"}


@pytest.mark.asyncio
async def test_interview_header_row_absent_metadata_json_returns_empty_dict():
    session = FakeSession(
        rows=[{"interview_id": IID, "title": "T", "metadata_json": None}]
    )
    row = await reader.interview_header_row(session, IID)
    assert row["metadata"] == {}


@pytest.mark.asyncio
async def test_interview_header_row_malformed_metadata_json_returns_empty_dict_no_raise():
    session = FakeSession(
        rows=[{"interview_id": IID, "title": "T", "metadata_json": "{not valid json"}]
    )
    row = await reader.interview_header_row(session, IID)
    assert row["metadata"] == {}


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
