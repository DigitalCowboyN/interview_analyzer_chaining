from unittest.mock import AsyncMock, MagicMock

import pytest

from src.export import reader

IID = "22222222-2222-2222-2222-222222222222"


def make_session(records):
    async def aiter(self):
        for r in records:
            yield r
    result = MagicMock()
    result.__aiter__ = aiter
    session = MagicMock()
    session.run = AsyncMock(return_value=result)
    return session


@pytest.mark.asyncio
async def test_transcript_rows_query_and_shape():
    session = make_session([
        {"sentence_id": "f1", "sequence_order": 0, "text": "Hi.",
         "speaker_id": "sp1", "speaker": "Alice", "utterance_id": "u1"},
    ])
    rows = await reader.transcript_rows(session, IID)
    assert rows[0]["speaker"] == "Alice"
    query = session.run.call_args[0][0]
    assert "HAS_SENTENCE" in query and "SPOKEN_BY" in query and "PART_OF_UTTERANCE" in query
    assert session.run.call_args.kwargs["interview_id"] == IID


@pytest.mark.asyncio
async def test_lens_item_rows_optional_filters():
    session = make_session([])
    await reader.lens_item_rows(session, IID, "meeting_minutes",
                                node_type="Decision", min_confidence=0.5, limit=10)
    query = session.run.call_args[0][0]
    kwargs = session.run.call_args.kwargs
    assert "LensItem" in query and "SUPPORTED_BY" in query
    assert kwargs["lens"] == "meeting_minutes"
    assert kwargs["node_type"] == "Decision" and kwargs["min_confidence"] == 0.5


@pytest.mark.asyncio
async def test_analysis_rows_takes_latest_analysis():
    session = make_session([])
    await reader.analysis_rows(session, IID)
    query = session.run.call_args[0][0]
    assert "HAS_ANALYSIS" in query and "created_at DESC" in query


@pytest.mark.asyncio
async def test_worklist_rows_filters_and_reasons():
    session = make_session([])
    result = await reader.worklist_rows(session, threshold=0.6)
    assert set(result) == {"lens_items", "claims"}
    lens_query = session.run.call_args_list[0][0][0]
    assert "_unresolved" in lens_query and "confidence < $threshold" in lens_query


@pytest.mark.asyncio
async def test_rollup_groups_by_display_name():
    session = make_session([])
    rows = await reader.speaker_rollup_rows(session, name="Alice Johnson")
    assert rows == []
    query = session.run.call_args_list[0][0][0]
    assert "display_name" in query and "merged_into IS NULL" in query


@pytest.mark.asyncio
async def test_rollup_queries_carry_scan_cap():
    session = make_session([])
    await reader.speaker_rollup_rows(session, scan_cap=123)
    for call in session.run.call_args_list:
        assert "LIMIT $scan_cap" in call[0][0]
        assert call.kwargs["scan_cap"] == 123


@pytest.mark.asyncio
async def test_rollup_substring_filter_on_grouped_data():
    rows = [
        {"display_name": "Alice Johnson", "node_type": "ActionItem", "relationship": "OWNED_BY",
         "text": "t", "interview_id": "i1", "item_id": "x1"},
        {"display_name": "Bob Reyes", "node_type": "Decision", "relationship": "DECIDED_BY",
         "text": "t", "interview_id": "i1", "item_id": "x2"},
    ]
    # first session.run call (items query) returns rows; second (claims) returns empty
    results = [rows, []]

    call_count = {"n": 0}

    def run_side_effect(query, **kw):
        res = MagicMock()
        data = results[min(call_count["n"], 1)]
        call_count["n"] += 1

        async def aiter(self):
            for r in data:
                yield r
        res.__aiter__ = aiter
        return res

    session = MagicMock()
    session.run = AsyncMock(side_effect=run_side_effect)
    grouped = await reader.speaker_rollup_rows(session, name="ali")
    assert [g["display_name"] for g in grouped] == ["Alice Johnson"]
