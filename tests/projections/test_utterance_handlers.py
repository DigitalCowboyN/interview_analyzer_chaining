from unittest.mock import AsyncMock, MagicMock

import pytest

from src.events.envelope import AggregateType, EventEnvelope
from src.projections.handlers.utterance_handlers import (
    InterruptionRecordedHandler,
    StitchRemovedHandler,
    UtteranceIdentifiedHandler,
)

IID = "22222222-2222-2222-2222-222222222222"
SP1 = "33333333-3333-3333-3333-333333333333"
U1 = "55555555-5555-5555-5555-555555555555"
U2 = "66666666-6666-6666-6666-666666666666"
F1 = "77777777-7777-7777-7777-777777777771"
F2 = "77777777-7777-7777-7777-777777777772"


def make_event(event_type, data, version=3):
    return EventEnvelope(
        event_type=event_type,
        aggregate_type=AggregateType.INTERVIEW,
        aggregate_id=IID,
        version=version,
        data=data,
    )


def mock_matched(tx, matched):
    """Mock the RETURN count(s) AS matched record."""
    tx.run.return_value.single = AsyncMock(return_value={"matched": matched})


@pytest.mark.asyncio
async def test_utterance_identified_links_fragments_with_position():
    handler = UtteranceIdentifiedHandler()
    tx = AsyncMock()
    mock_matched(tx, 2)
    event = make_event(
        "UtteranceIdentified",
        {"utterance_id": U1, "speaker_id": SP1, "fragment_ids": [F1, F2], "confidence": 0.75},
    )
    await handler.apply(tx, event)
    query = tx.run.call_args[0][0]
    params = tx.run.call_args[1]
    assert "PART_OF_UTTERANCE" in query
    assert "SPOKE" in query
    assert params["fragments"] == [
        {"id": F1, "position": 0},
        {"id": F2, "position": 1},
    ]
    assert params["interview_id"] == IID


@pytest.mark.asyncio
async def test_interruption_recorded_creates_edge():
    handler = InterruptionRecordedHandler()
    tx = AsyncMock()
    event = make_event(
        "InterruptionRecorded",
        {"interrupting_utterance_id": U2, "interrupted_utterance_id": U1, "at_fragment_id": F2},
    )
    await handler.apply(tx, event)
    query = tx.run.call_args[0][0]
    params = tx.run.call_args[1]
    assert "INTERRUPTS" in query
    assert params["at_fragment_id"] == F2


@pytest.mark.asyncio
async def test_stitch_removed_detaches_utterance():
    handler = StitchRemovedHandler()
    tx = AsyncMock()
    event = make_event("StitchRemoved", {"utterance_id": U1, "reason": "wrong"})
    await handler.apply(tx, event)
    assert "DETACH DELETE" in tx.run.call_args[0][0]


def test_utterance_handlers_registered_in_bootstrap():
    from src.projections.bootstrap import create_handler_registry

    registry = create_handler_registry(parked_events_manager=MagicMock())
    for event_type in ("UtteranceIdentified", "InterruptionRecorded", "StitchRemoved"):
        assert registry.has_handler(event_type), event_type


@pytest.mark.asyncio
async def test_utterance_identified_raises_when_speaker_missing():
    # No result row means the Speaker MATCH found nothing (out-of-order
    # delivery); the handler must raise so retry/park logic engages.
    handler = UtteranceIdentifiedHandler()
    tx = AsyncMock()
    tx.run.return_value.single = AsyncMock(return_value=None)
    event = make_event(
        "UtteranceIdentified",
        {"utterance_id": U1, "speaker_id": SP1, "fragment_ids": [F1], "confidence": 0.75},
    )
    with pytest.raises(ValueError, match="no writes applied"):
        await handler.apply(tx, event)


@pytest.mark.asyncio
async def test_utterance_identified_raises_when_fragments_partially_missing():
    # On projection rebuild the interview stream races ahead of the sentence
    # stream; if some fragments are not yet projected the handler must raise
    # rather than seal a partial overlay behind the version guard.
    handler = UtteranceIdentifiedHandler()
    tx = AsyncMock()
    mock_matched(tx, 1)
    event = make_event(
        "UtteranceIdentified",
        {"utterance_id": U1, "speaker_id": SP1, "fragment_ids": [F1, F2], "confidence": 0.75},
    )
    with pytest.raises(ValueError, match="only 1/2 fragments matched"):
        await handler.apply(tx, event)
