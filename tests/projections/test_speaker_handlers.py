from unittest.mock import AsyncMock

import pytest

from src.events.envelope import AggregateType, EventEnvelope
from src.projections.handlers.speaker_handlers import (
    SpeakerAttributedHandler,
    SpeakerCreatedHandler,
    SpeakerMergedHandler,
    SpeakerReattributedHandler,
    SpeakerRenamedHandler,
)

IID = "22222222-2222-2222-2222-222222222222"
SP1 = "33333333-3333-3333-3333-333333333333"
SP2 = "44444444-4444-4444-4444-444444444444"
SENT = "77777777-7777-7777-7777-777777777771"


def make_event(event_type, aggregate_type, aggregate_id, data, version=1):
    return EventEnvelope(
        event_type=event_type,
        aggregate_type=aggregate_type,
        aggregate_id=aggregate_id,
        version=version,
        data=data,
    )


@pytest.mark.asyncio
async def test_speaker_created_merges_speaker_and_participant_link():
    handler = SpeakerCreatedHandler()
    tx = AsyncMock()
    event = make_event(
        "SpeakerCreated", AggregateType.INTERVIEW, IID,
        {"speaker_id": SP1, "handle": "S1", "display_name": "S1",
         "provisional": True, "confidence": 0.8, "method": "inference"},
    )
    await handler.apply(tx, event)
    query = tx.run.call_args[0][0]
    params = tx.run.call_args[1]
    assert "MERGE (sp:Speaker {speaker_id: $speaker_id})" in query
    assert "HAS_PARTICIPANT" in query
    assert params["speaker_id"] == SP1
    assert params["interview_id"] == IID
    assert params["provisional"] is True


@pytest.mark.asyncio
async def test_speaker_renamed_updates_display_name():
    handler = SpeakerRenamedHandler()
    tx = AsyncMock()
    event = make_event(
        "SpeakerRenamed", AggregateType.INTERVIEW, IID,
        {"speaker_id": SP1, "old_display_name": "S1", "new_display_name": "Dana"},
    )
    await handler.apply(tx, event)
    query = tx.run.call_args[0][0]
    params = tx.run.call_args[1]
    assert "display_name" in query
    assert "provisional = false" in query
    assert params["new_display_name"] == "Dana"


@pytest.mark.asyncio
async def test_speaker_attributed_creates_spoken_by():
    handler = SpeakerAttributedHandler()
    tx = AsyncMock()
    event = make_event(
        "SpeakerAttributed", AggregateType.SENTENCE, SENT,
        {"speaker_id": SP1, "confidence": 0.72, "method": "inference"},
    )
    await handler.apply(tx, event)
    query = tx.run.call_args[0][0]
    params = tx.run.call_args[1]
    assert "SPOKEN_BY" in query
    assert params["confidence"] == 0.72


@pytest.mark.asyncio
async def test_speaker_reattributed_locks_edge():
    handler = SpeakerReattributedHandler()
    tx = AsyncMock()
    event = make_event(
        "SpeakerReattributed", AggregateType.SENTENCE, SENT,
        {"old_speaker_id": SP1, "new_speaker_id": SP2},
    )
    await handler.apply(tx, event)
    query = tx.run.call_args[0][0]
    params = tx.run.call_args[1]
    assert "SPOKEN_BY" in query
    assert "locked = true" in query
    assert params["new_speaker_id"] == SP2


@pytest.mark.asyncio
async def test_speaker_merged_moves_spoken_by_edges():
    handler = SpeakerMergedHandler()
    tx = AsyncMock()
    event = make_event(
        "SpeakerMerged", AggregateType.INTERVIEW, IID,
        {"surviving_speaker_id": SP1, "merged_speaker_id": SP2},
    )
    await handler.apply(tx, event)
    query = tx.run.call_args[0][0]
    assert "merged_into" in query
    assert "SPOKEN_BY" in query
    assert "SPOKE" in query


def test_handlers_registered_in_bootstrap():
    from unittest.mock import MagicMock

    from src.projections.bootstrap import create_handler_registry

    registry = create_handler_registry(parked_events_manager=MagicMock())
    for event_type in (
        "SpeakerCreated",
        "SpeakerRenamed",
        "SpeakerMerged",
        "SpeakerAttributed",
        "SpeakerReattributed",
    ):
        assert registry.has_handler(event_type), event_type
