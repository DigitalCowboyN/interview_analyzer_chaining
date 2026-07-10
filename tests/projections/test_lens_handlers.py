from unittest.mock import AsyncMock, MagicMock

import pytest

from src.events.envelope import AggregateType, EventEnvelope
from src.projections.handlers.lens_handlers import (
    LensAppliedHandler,
    LensExtractionGeneratedHandler,
    LensExtractionOverriddenHandler,
)

IID = "22222222-2222-2222-2222-222222222222"
SP1 = "33333333-3333-3333-3333-333333333333"
F1 = "77777777-7777-7777-7777-777777777771"
ITEM = "88888888-8888-8888-8888-888888888801"


def make_event(event_type, data):
    return EventEnvelope(
        event_type=event_type,
        aggregate_type=AggregateType.INTERVIEW,
        aggregate_id=IID,
        version=5,
        data=data,
    )


def item_data(**over):
    d = {
        "lens": "meeting_minutes",
        "lens_version": 1,
        "node_type": "Decision",
        "item_id": ITEM,
        "fields": {"text": "Go with X", "made_by": "S1"},
        "supporting_fragment_ids": [F1],
        "speaker_links": [{"relationship": "DECIDED_BY", "speaker_id": SP1}],
        "confidence": 0.9,
        "model": "haiku",
        "provider": "anthropic",
    }
    d.update(over)
    return d


@pytest.mark.asyncio
async def test_lens_applied_deletes_only_unlocked_older_items():
    handler = LensAppliedHandler()
    tx = AsyncMock()
    await handler.apply(tx, make_event("LensApplied", {"lens": "meeting_minutes", "lens_version": 2}))
    query = tx.run.call_args[0][0]
    assert "lens_version < $lens_version" in query
    assert "coalesce(n.locked, false) = false" in query
    assert "DETACH DELETE" in query


@pytest.mark.asyncio
async def test_extraction_merges_dual_label_node_with_links():
    handler = LensExtractionGeneratedHandler()
    tx = AsyncMock()
    tx.run.return_value.single = AsyncMock(return_value={"supported": 1, "linked": 1})
    await handler.apply(tx, make_event("LensExtractionGenerated", item_data()))
    query = tx.run.call_args[0][0]
    assert "MERGE (n:LensItem:Decision {item_id: $item_id})" in query
    assert "SUPPORTED_BY" in query
    assert "DECIDED_BY" in query


@pytest.mark.asyncio
async def test_extraction_rejects_invalid_label():
    handler = LensExtractionGeneratedHandler()
    tx = AsyncMock()
    with pytest.raises(ValueError, match="label"):
        await handler.apply(
            tx, make_event("LensExtractionGenerated", item_data(node_type="Bad Label; DROP"))
        )
    tx.run.assert_not_called()


@pytest.mark.asyncio
async def test_extraction_raises_when_fragments_missing():
    handler = LensExtractionGeneratedHandler()
    tx = AsyncMock()
    tx.run.return_value.single = AsyncMock(return_value={"supported": 0, "linked": 1})
    with pytest.raises(ValueError, match="not yet projected"):
        await handler.apply(tx, make_event("LensExtractionGenerated", item_data()))


@pytest.mark.asyncio
async def test_override_sets_fields_and_lock():
    handler = LensExtractionOverriddenHandler()
    tx = AsyncMock()
    counters = MagicMock(nodes_created=0, properties_set=2, relationships_created=0)
    tx.run.return_value.consume = AsyncMock(return_value=MagicMock(counters=counters))
    await handler.apply(
        tx,
        make_event(
            "LensExtractionOverridden",
            {"item_id": ITEM, "fields_overridden": {"text": "Go with Y"}, "note": "fixed"},
        ),
    )
    query = tx.run.call_args[0][0]
    assert "locked = true" in query


def test_lens_events_in_bootstrap_and_allowlists():
    from src.projections.bootstrap import create_handler_registry
    from src.projections.config import get_all_allowed_event_types

    registry = create_handler_registry(parked_events_manager=MagicMock())
    allowed = set(get_all_allowed_event_types())
    for et in ("LensApplied", "LensExtractionGenerated", "LensExtractionOverridden"):
        assert registry.has_handler(et)
        assert et in allowed
