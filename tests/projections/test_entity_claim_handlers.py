from unittest.mock import AsyncMock, MagicMock

import pytest

from src.events.envelope import AggregateType, EventEnvelope
from src.projections.handlers.claim_handlers import ClaimExtractedHandler
from src.projections.handlers.entity_handlers import EntitiesExtractedHandler

IID = "22222222-2222-2222-2222-222222222222"
F1 = "77777777-7777-7777-7777-777777777771"

ENTITY = {"text": "Neo4j", "entity_type": "product", "start": 4, "end": 9, "confidence": 0.9}


def make_event(event_type, aggregate_type, aggregate_id, data):
    return EventEnvelope(
        event_type=event_type,
        aggregate_type=aggregate_type,
        aggregate_id=aggregate_id,
        version=2,
        data=data,
    )


def mock_write_counters(tx, properties_set=1):
    counters = MagicMock(nodes_created=0, properties_set=properties_set, relationships_created=0)
    tx.run.return_value.consume = AsyncMock(return_value=MagicMock(counters=counters))


@pytest.mark.asyncio
async def test_entities_extracted_merges_entity_and_mentions():
    handler = EntitiesExtractedHandler()
    tx = AsyncMock()
    mock_write_counters(tx)
    event = make_event(
        "EntitiesExtracted", AggregateType.SENTENCE, F1,
        {"interview_id": IID, "model": "haiku", "provider": "anthropic", "entities": [ENTITY]},
    )
    await handler.apply(tx, event)
    queries = [c.args[0] for c in tx.run.call_args_list]
    assert any("entities_extracted_at" in q for q in queries)  # guard statement
    assert any("MERGE (e:Entity" in q and "MENTIONS" in q for q in queries)


@pytest.mark.asyncio
async def test_entities_extracted_raises_when_sentence_missing():
    handler = EntitiesExtractedHandler()
    tx = AsyncMock()
    mock_write_counters(tx, properties_set=0)
    event = make_event(
        "EntitiesExtracted", AggregateType.SENTENCE, F1,
        {"interview_id": IID, "model": "haiku", "provider": "anthropic", "entities": [ENTITY]},
    )
    with pytest.raises(ValueError, match="no writes applied"):
        await handler.apply(tx, event)


@pytest.mark.asyncio
async def test_entities_empty_extraction_still_guards_sentence_presence():
    handler = EntitiesExtractedHandler()
    tx = AsyncMock()
    mock_write_counters(tx)
    event = make_event(
        "EntitiesExtracted", AggregateType.SENTENCE, F1,
        {"interview_id": IID, "model": "haiku", "provider": "anthropic", "entities": []},
    )
    await handler.apply(tx, event)  # no raise; old MENTIONS cleared, none created
    queries = [c.args[0] for c in tx.run.call_args_list]
    assert any("entities_extracted_at" in q for q in queries)


@pytest.mark.asyncio
async def test_claim_extracted_creates_claim_with_support():
    handler = ClaimExtractedHandler()
    tx = AsyncMock()
    tx.run.return_value.single = AsyncMock(return_value={"supported": 2})
    event = make_event(
        "ClaimExtracted", AggregateType.INTERVIEW, IID,
        {"claim_id": "c-1", "utterance_id": "u-1", "speaker_id": "sp-1",
         "text": "We ship Friday", "kind": "commitment", "confidence": 0.8,
         "model": "haiku", "provider": "anthropic"},
    )
    await handler.apply(tx, event)
    query = tx.run.call_args[0][0]
    assert "MERGE (c:Claim {claim_id: $claim_id})" in query
    assert "MADE_BY" in query and "SUPPORTED_BY" in query
    params = tx.run.call_args[1]
    assert params["kind"] == "commitment"


@pytest.mark.asyncio
async def test_claim_extracted_raises_when_targets_missing():
    handler = ClaimExtractedHandler()
    tx = AsyncMock()
    tx.run.return_value.single = AsyncMock(return_value=None)
    event = make_event(
        "ClaimExtracted", AggregateType.INTERVIEW, IID,
        {"claim_id": "c", "utterance_id": "u", "speaker_id": "sp",
         "text": "x", "kind": "assertion", "confidence": 0.8, "model": "m", "provider": "p"},
    )
    with pytest.raises(ValueError, match="not yet projected"):
        await handler.apply(tx, event)


def test_new_event_types_in_bootstrap_and_allowlists():
    from src.projections.bootstrap import create_handler_registry
    from src.projections.config import get_all_allowed_event_types

    registry = create_handler_registry(parked_events_manager=MagicMock())
    allowed = set(get_all_allowed_event_types())
    for event_type in ("EntitiesExtracted", "ClaimExtracted"):
        assert registry.has_handler(event_type)
        assert event_type in allowed
