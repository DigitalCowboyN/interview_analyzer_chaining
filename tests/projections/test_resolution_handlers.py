"""Resolution projection handlers: Cypher params + ordering guards (M4.5b)."""

from unittest.mock import AsyncMock

import pytest

from src.events.envelope import AggregateType, EventEnvelope
from src.projections.handlers.resolution_handlers import (
    EntityAliasAddedHandler,
    EntityCanonicalizedHandler,
    EntityMergeConfirmedHandler,
    EntitySplitHandler,
    PersonIdentifiedHandler,
    PersonLinkRemovedHandler,
    SpeakerLinkedToPersonHandler,
)

PID = "acme-2026"
CANONICAL_ID = "11111111-1111-1111-1111-111111111111"
MERGED_ID = "22222222-2222-2222-2222-222222222222"
NEW_CANONICAL_ID = "33333333-3333-3333-3333-333333333333"
PERSON_ID = "44444444-4444-4444-4444-444444444444"
SPEAKER_ID = "55555555-5555-5555-5555-555555555555"
AGGREGATE_ID = "66666666-6666-6666-6666-666666666666"


def make_event(event_type, data):
    return EventEnvelope(
        event_type=event_type,
        aggregate_type=AggregateType.PROJECT,
        aggregate_id=AGGREGATE_ID,
        version=1,
        data=data,
    )


def make_tx(**record):
    """AsyncMock tx whose tx.run(...) returns an object with async single()."""
    tx = AsyncMock()
    tx.run.return_value.single = AsyncMock(return_value=record or None)
    return tx


# --- EntityCanonicalizedHandler -------------------------------------------------


@pytest.mark.asyncio
async def test_entity_canonicalized_passes_params_and_raises_when_links_short():
    handler = EntityCanonicalizedHandler()
    tx = make_tx(links=1)
    data = {
        "canonical_id": CANONICAL_ID,
        "name": "Acme Corp",
        "entity_type": "ORG",
        "project_id": PID,
        "method": "deterministic",
        "confidence": 0.8,
        "surfaces": ["acme", "acme corp"],
    }
    with pytest.raises(ValueError, match="only 1/2"):
        await handler.apply(tx, make_event("EntityCanonicalized", data))

    kwargs = tx.run.call_args.kwargs
    assert kwargs == {
        "canonical_id": CANONICAL_ID,
        "name": "Acme Corp",
        "entity_type": "ORG",
        "project_id": PID,
        "method": "deterministic",
        "confidence": 0.8,
        "surfaces": ["acme", "acme corp"],
    }


@pytest.mark.asyncio
async def test_entity_canonicalized_no_raise_when_links_match_surfaces():
    handler = EntityCanonicalizedHandler()
    tx = make_tx(links=2)
    data = {
        "canonical_id": CANONICAL_ID,
        "name": "Acme Corp",
        "entity_type": "ORG",
        "project_id": PID,
        "method": "deterministic",
        "confidence": 0.8,
        "surfaces": ["acme", "acme corp"],
    }
    await handler.apply(tx, make_event("EntityCanonicalized", data))
    tx.run.assert_awaited_once()


# --- EntityAliasAddedHandler -----------------------------------------------------


@pytest.mark.asyncio
async def test_entity_alias_added_raises_when_links_zero():
    handler = EntityAliasAddedHandler()
    tx = make_tx(links=0)
    data = {
        "canonical_id": CANONICAL_ID,
        "surface": "acme inc",
        "project_id": PID,
        "method": "deterministic",
        "confidence": 0.75,
    }
    with pytest.raises(ValueError, match="not yet projected"):
        await handler.apply(tx, make_event("EntityAliasAdded", data))

    kwargs = tx.run.call_args.kwargs
    assert kwargs == {
        "canonical_id": CANONICAL_ID,
        "surface": "acme inc",
        "project_id": PID,
        "method": "deterministic",
        "confidence": 0.75,
    }


@pytest.mark.asyncio
async def test_entity_alias_added_no_raise_when_links_nonzero():
    handler = EntityAliasAddedHandler()
    tx = make_tx(links=1)
    data = {
        "canonical_id": CANONICAL_ID,
        "surface": "acme inc",
        "project_id": PID,
        "method": "human",
        "confidence": 1.0,
    }
    await handler.apply(tx, make_event("EntityAliasAdded", data))
    tx.run.assert_awaited_once()


# --- EntityMergeConfirmedHandler -------------------------------------------------


@pytest.mark.asyncio
async def test_entity_merge_confirmed_raises_when_query1_found_zero():
    handler = EntityMergeConfirmedHandler()
    tx = AsyncMock()
    tx.run.return_value.single = AsyncMock(return_value={"found": 0})
    data = {
        "canonical_id": CANONICAL_ID,
        "merged_canonical_id": MERGED_ID,
        "project_id": PID,
    }
    with pytest.raises(ValueError, match="canonicals not yet projected"):
        await handler.apply(tx, make_event("EntityMergeConfirmed", data))

    tx.run.assert_awaited_once()
    kwargs = tx.run.call_args.kwargs
    assert kwargs == {
        "canonical_id": CANONICAL_ID,
        "merged_canonical_id": MERGED_ID,
    }


@pytest.mark.asyncio
async def test_entity_merge_confirmed_runs_second_query_and_does_not_raise_on_zero_moves():
    handler = EntityMergeConfirmedHandler()
    tx = AsyncMock()
    tx.run.return_value.single = AsyncMock(return_value={"found": 2})
    data = {
        "canonical_id": CANONICAL_ID,
        "merged_canonical_id": MERGED_ID,
        "project_id": PID,
    }
    await handler.apply(tx, make_event("EntityMergeConfirmed", data))

    assert tx.run.await_count == 2
    lock_kwargs = tx.run.call_args_list[0].kwargs
    assert lock_kwargs == {
        "canonical_id": CANONICAL_ID,
        "merged_canonical_id": MERGED_ID,
    }
    move_query = tx.run.call_args_list[1].args[0]
    move_kwargs = tx.run.call_args_list[1].kwargs
    assert "DELETE a" in move_query
    assert move_kwargs == {
        "project_id": PID,
        "canonical_id": CANONICAL_ID,
        "merged_canonical_id": MERGED_ID,
    }


# --- EntitySplitHandler -----------------------------------------------------------


@pytest.mark.asyncio
async def test_entity_split_raises_when_query1_created_zero():
    handler = EntitySplitHandler()
    tx = AsyncMock()
    tx.run.return_value.single = AsyncMock(return_value={"created": 0})
    data = {
        "canonical_id": CANONICAL_ID,
        "new_canonical_id": NEW_CANONICAL_ID,
        "new_name": "Acme Subsidiary",
        "project_id": PID,
        "surfaces_removed": ["acme sub"],
    }
    with pytest.raises(ValueError, match="not yet projected"):
        await handler.apply(tx, make_event("EntitySplit", data))

    tx.run.assert_awaited_once()
    kwargs = tx.run.call_args.kwargs
    assert kwargs == {
        "canonical_id": CANONICAL_ID,
        "new_canonical_id": NEW_CANONICAL_ID,
        "new_name": "Acme Subsidiary",
        "project_id": PID,
    }


@pytest.mark.asyncio
async def test_entity_split_runs_second_query_after_first_succeeds():
    handler = EntitySplitHandler()
    tx = AsyncMock()
    tx.run.return_value.single = AsyncMock(return_value={"created": 1})
    data = {
        "canonical_id": CANONICAL_ID,
        "new_canonical_id": NEW_CANONICAL_ID,
        "new_name": "Acme Subsidiary",
        "project_id": PID,
        "surfaces_removed": ["acme sub", "acme subsidiary"],
    }
    await handler.apply(tx, make_event("EntitySplit", data))

    assert tx.run.await_count == 2
    move_query = tx.run.call_args_list[1].args[0]
    move_kwargs = tx.run.call_args_list[1].kwargs
    assert "DELETE a" in move_query
    assert move_kwargs == {
        "surfaces_removed": ["acme sub", "acme subsidiary"],
        "project_id": PID,
        "canonical_id": CANONICAL_ID,
        "new_canonical_id": NEW_CANONICAL_ID,
    }


# --- PersonIdentifiedHandler -------------------------------------------------------


@pytest.mark.asyncio
async def test_person_identified_merges_person_no_ordering_guard():
    handler = PersonIdentifiedHandler()
    tx = AsyncMock()
    data = {
        "person_id": PERSON_ID,
        "display_name": "Jane Doe",
        "project_id": PID,
    }
    await handler.apply(tx, make_event("PersonIdentified", data))

    tx.run.assert_awaited_once()
    query = tx.run.call_args.args[0]
    kwargs = tx.run.call_args.kwargs
    assert "MERGE (p:Person {person_id: $person_id})" in query
    assert kwargs == {
        "person_id": PERSON_ID,
        "display_name": "Jane Doe",
        "project_id": PID,
    }


# --- SpeakerLinkedToPersonHandler ---------------------------------------------------


@pytest.mark.asyncio
async def test_speaker_linked_to_person_raises_when_links_zero():
    handler = SpeakerLinkedToPersonHandler()
    tx = make_tx(links=0)
    data = {
        "speaker_id": SPEAKER_ID,
        "person_id": PERSON_ID,
        "interview_id": AGGREGATE_ID,
        "method": "exact_name",
        "confidence": 0.95,
        "project_id": PID,
    }
    with pytest.raises(ValueError, match="not yet projected"):
        await handler.apply(tx, make_event("SpeakerLinkedToPerson", data))

    kwargs = tx.run.call_args.kwargs
    assert kwargs == {
        "speaker_id": SPEAKER_ID,
        "person_id": PERSON_ID,
        "method": "exact_name",
        "confidence": 0.95,
    }


@pytest.mark.asyncio
async def test_speaker_linked_to_person_no_raise_when_links_nonzero():
    handler = SpeakerLinkedToPersonHandler()
    tx = make_tx(links=1)
    data = {
        "speaker_id": SPEAKER_ID,
        "person_id": PERSON_ID,
        "interview_id": AGGREGATE_ID,
        "method": "human",
        "confidence": 1.0,
        "project_id": PID,
    }
    await handler.apply(tx, make_event("SpeakerLinkedToPerson", data))
    tx.run.assert_awaited_once()


# --- PersonLinkRemovedHandler ------------------------------------------------------


@pytest.mark.asyncio
async def test_person_link_removed_raises_when_removed_zero():
    handler = PersonLinkRemovedHandler()
    tx = make_tx(removed=0)
    data = {
        "speaker_id": SPEAKER_ID,
        "person_id": PERSON_ID,
        "interview_id": AGGREGATE_ID,
        "project_id": PID,
        "note": "wrong match",
    }
    with pytest.raises(ValueError, match="not yet projected"):
        await handler.apply(tx, make_event("PersonLinkRemoved", data))

    kwargs = tx.run.call_args.kwargs
    assert kwargs == {"speaker_id": SPEAKER_ID, "person_id": PERSON_ID}


@pytest.mark.asyncio
async def test_person_link_removed_no_raise_when_removed_nonzero():
    handler = PersonLinkRemovedHandler()
    tx = make_tx(removed=1)
    data = {
        "speaker_id": SPEAKER_ID,
        "person_id": PERSON_ID,
        "interview_id": AGGREGATE_ID,
        "project_id": PID,
        "note": None,
    }
    await handler.apply(tx, make_event("PersonLinkRemoved", data))
    tx.run.assert_awaited_once()
