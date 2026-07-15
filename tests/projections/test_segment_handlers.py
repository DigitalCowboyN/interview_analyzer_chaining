"""Segment overlay projection handlers: Cypher params + ordering guards (M4.5c)."""

from unittest.mock import AsyncMock

import pytest

from src.events.envelope import AggregateType, EventEnvelope
from src.projections.handlers.segment_handlers import (
    SegmentIdentifiedHandler,
    SegmentRemovedHandler,
)

INTERVIEW_ID = "66666666-6666-6666-6666-666666666666"
SEGMENT_ID = "77777777-7777-7777-7777-777777777777"


def make_event(event_type, data):
    return EventEnvelope(
        event_type=event_type,
        aggregate_type=AggregateType.INTERVIEW,
        aggregate_id=INTERVIEW_ID,
        version=1,
        data=data,
    )


def make_tx(**record):
    """AsyncMock tx whose tx.run(...) returns an object with async single()."""
    tx = AsyncMock()
    tx.run.return_value.single = AsyncMock(return_value=record or None)
    return tx


# --- SegmentIdentifiedHandler ------------------------------------------------


@pytest.mark.asyncio
async def test_segment_identified_passes_params_and_raises_when_linked_short():
    handler = SegmentIdentifiedHandler()
    tx = make_tx(linked=2)
    data = {
        "segment_id": SEGMENT_ID,
        "topic": "onboarding",
        "start_index": 0,
        "end_index": 2,
        "confidence": 0.9,
    }
    with pytest.raises(ValueError, match="only 2/3"):
        await handler.apply(tx, make_event("SegmentIdentified", data))

    kwargs = tx.run.call_args.kwargs
    assert kwargs == {
        "segment_id": SEGMENT_ID,
        "topic": "onboarding",
        "interview_id": INTERVIEW_ID,
        "confidence": 0.9,
        "start_index": 0,
        "end_index": 2,
    }


@pytest.mark.asyncio
async def test_segment_identified_no_raise_when_linked_matches_range():
    handler = SegmentIdentifiedHandler()
    tx = make_tx(linked=3)
    data = {
        "segment_id": SEGMENT_ID,
        "topic": "onboarding",
        "start_index": 0,
        "end_index": 2,
        "confidence": 0.9,
    }
    await handler.apply(tx, make_event("SegmentIdentified", data))
    tx.run.assert_awaited_once()


@pytest.mark.asyncio
async def test_segment_identified_query_shape():
    handler = SegmentIdentifiedHandler()
    tx = make_tx(linked=3)
    data = {
        "segment_id": SEGMENT_ID,
        "topic": "onboarding",
        "start_index": 0,
        "end_index": 2,
        "confidence": 0.9,
    }
    await handler.apply(tx, make_event("SegmentIdentified", data))

    query = tx.run.call_args.args[0]
    assert "MERGE (seg:Segment {segment_id:" in query
    assert "DELETE old" in query
    assert "RETURN count(f) AS linked" in query


# --- SegmentRemovedHandler ----------------------------------------------------


@pytest.mark.asyncio
async def test_segment_removed_no_raise_when_found():
    handler = SegmentRemovedHandler()
    tx = make_tx(found=1)
    data = {"segment_id": SEGMENT_ID, "reason": "wrong split"}
    await handler.apply(tx, make_event("SegmentRemoved", data))

    tx.run.assert_awaited_once()
    query = tx.run.call_args.args[0]
    assert "DETACH DELETE" in query


@pytest.mark.asyncio
async def test_segment_removed_raises_when_not_found():
    handler = SegmentRemovedHandler()
    tx = make_tx(found=0)
    data = {"segment_id": SEGMENT_ID, "reason": None}
    with pytest.raises(ValueError, match="not projected yet"):
        await handler.apply(tx, make_event("SegmentRemoved", data))
