from unittest.mock import AsyncMock, MagicMock

import pytest

from src.enrichment.embedder import encode_vector
from src.events.envelope import AggregateType, EventEnvelope
from src.projections.handlers.embedding_handlers import EmbeddingGeneratedHandler

IID = "22222222-2222-2222-2222-222222222222"
F1 = "77777777-7777-7777-7777-777777777771"


@pytest.mark.asyncio
async def test_embedding_written_with_model_tag():
    handler = EmbeddingGeneratedHandler()
    handler._index_ensured = True  # skip index DDL in unit test
    tx = AsyncMock()
    counters = MagicMock(nodes_created=0, properties_set=3, relationships_created=0)
    tx.run.return_value.consume = AsyncMock(return_value=MagicMock(counters=counters))
    event = EventEnvelope(
        event_type="EmbeddingGenerated",
        aggregate_type=AggregateType.SENTENCE,
        aggregate_id=F1,
        version=3,
        data={
            "interview_id": IID,
            "model": "text-embedding-3-small",
            "dim": 3,
            "vector_b64": encode_vector([0.1, 0.2, 0.3]),
        },
    )
    await handler.apply(tx, event)
    params = tx.run.call_args[1]
    assert params["model"] == "text-embedding-3-small"
    assert len(params["vector"]) == 3


@pytest.mark.asyncio
async def test_embedding_raises_when_sentence_missing():
    handler = EmbeddingGeneratedHandler()
    handler._index_ensured = True
    tx = AsyncMock()
    counters = MagicMock(nodes_created=0, properties_set=0, relationships_created=0)
    tx.run.return_value.consume = AsyncMock(return_value=MagicMock(counters=counters))
    event = EventEnvelope(
        event_type="EmbeddingGenerated",
        aggregate_type=AggregateType.SENTENCE,
        aggregate_id=F1,
        version=3,
        data={"interview_id": IID, "model": "m", "dim": 3, "vector_b64": encode_vector([0.1, 0.2, 0.3])},
    )
    with pytest.raises(ValueError, match="no writes applied"):
        await handler.apply(tx, event)


def test_embedding_events_in_bootstrap_and_allowlists():
    from src.projections.bootstrap import create_handler_registry
    from src.projections.config import get_all_allowed_event_types

    registry = create_handler_registry(parked_events_manager=MagicMock())
    allowed = set(get_all_allowed_event_types())
    for event_type in ("EmbeddingGenerated", "UtteranceEmbeddingGenerated"):
        assert registry.has_handler(event_type)
        assert event_type in allowed
