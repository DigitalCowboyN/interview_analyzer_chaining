"""Projection handlers for Speaker-related events."""

import logging

from src.events.envelope import EventEnvelope

from .base_handler import BaseProjectionHandler

logger = logging.getLogger(__name__)


def _raise_if_no_writes(summary, event_type: str, aggregate_id: str) -> None:
    """Raise when a MATCH-anchored write query touched nothing.

    Sentence and Speaker nodes are projected from independent subscriptions;
    zero writes means a MATCH found nothing (out-of-order delivery). Raising
    lets base-class retry/park engage instead of silently consuming the event.
    SET always fires on successful matches, so idempotent replays still count
    properties_set > 0 and do not raise.
    """
    counters = summary.counters
    if (
        counters.nodes_created == 0
        and counters.properties_set == 0
        and counters.relationships_created == 0
    ):
        raise ValueError(
            f"{event_type} for {aggregate_id}: no writes applied "
            f"(referenced nodes not yet projected?)"
        )


class SpeakerCreatedHandler(BaseProjectionHandler):
    """Creates Speaker node and links it to its Interview."""

    async def apply(self, tx, event: EventEnvelope):
        data = event.data
        query = """
        MATCH (i:Interview {interview_id: $interview_id})
        MERGE (sp:Speaker {speaker_id: $speaker_id})
        SET sp.handle = $handle,
            sp.display_name = $display_name,
            sp.provisional = $provisional,
            sp.confidence = $confidence,
            sp.method = $method,
            sp.interview_id = $interview_id,
            sp.merged_into = null
        MERGE (i)-[:HAS_PARTICIPANT]->(sp)
        """
        await tx.run(
            query,
            interview_id=event.aggregate_id,
            speaker_id=data["speaker_id"],
            handle=data["handle"],
            display_name=data["display_name"],
            provisional=data["provisional"],
            confidence=data.get("confidence"),
            method=data.get("method"),
        )
        logger.info(f"Applied SpeakerCreated for speaker {data['speaker_id']}")


class SpeakerRenamedHandler(BaseProjectionHandler):
    """Updates Speaker display name; clears provisional flag."""

    async def apply(self, tx, event: EventEnvelope):
        data = event.data
        query = """
        MATCH (sp:Speaker {speaker_id: $speaker_id})
        SET sp.display_name = $new_display_name,
            sp.provisional = false
        """
        await tx.run(
            query,
            speaker_id=data["speaker_id"],
            new_display_name=data["new_display_name"],
        )


class SpeakerMergedHandler(BaseProjectionHandler):
    """Moves SPOKEN_BY and SPOKE edges from merged speaker to survivor."""

    async def apply(self, tx, event: EventEnvelope):
        data = event.data
        query = """
        MATCH (merged:Speaker {speaker_id: $merged_speaker_id})
        MATCH (surviving:Speaker {speaker_id: $surviving_speaker_id})
        SET merged.merged_into = $surviving_speaker_id
        WITH merged, surviving
        CALL {
            WITH merged, surviving
            MATCH (s:Fragment)-[r:SPOKEN_BY]->(merged)
            MERGE (s)-[nr:SPOKEN_BY]->(surviving)
            SET nr.confidence = r.confidence, nr.method = r.method, nr.locked = r.locked
            DELETE r
        }
        WITH merged, surviving
        CALL {
            WITH merged, surviving
            MATCH (merged)-[sp:SPOKE]->(u:Utterance)
            MERGE (surviving)-[:SPOKE]->(u)
            DELETE sp
        }
        """
        await tx.run(
            query,
            merged_speaker_id=data["merged_speaker_id"],
            surviving_speaker_id=data["surviving_speaker_id"],
        )
        logger.info(
            f"Applied SpeakerMerged: {data['merged_speaker_id']} -> {data['surviving_speaker_id']}"
        )


class SpeakerAttributedHandler(BaseProjectionHandler):
    """Attributes a Sentence (fragment) to a Speaker via SPOKEN_BY."""

    async def apply(self, tx, event: EventEnvelope):
        data = event.data
        query = """
        MATCH (s:Fragment {aggregate_id: $aggregate_id})
        MATCH (sp:Speaker {speaker_id: $speaker_id})
        OPTIONAL MATCH (s)-[old:SPOKEN_BY]->(:Speaker)
        DELETE old
        MERGE (s)-[r:SPOKEN_BY]->(sp)
        SET r.confidence = $confidence, r.method = $method, r.locked = false
        """
        result = await tx.run(
            query,
            aggregate_id=event.aggregate_id,
            speaker_id=data["speaker_id"],
            confidence=data["confidence"],
            method=data["method"],
        )
        _raise_if_no_writes(await result.consume(), "SpeakerAttributed", event.aggregate_id)


class SpeakerReattributedHandler(BaseProjectionHandler):
    """Human correction: reattribute a fragment and lock it."""

    async def apply(self, tx, event: EventEnvelope):
        data = event.data
        query = """
        MATCH (s:Fragment {aggregate_id: $aggregate_id})
        MATCH (sp:Speaker {speaker_id: $new_speaker_id})
        OPTIONAL MATCH (s)-[old:SPOKEN_BY]->(:Speaker)
        DELETE old
        MERGE (s)-[r:SPOKEN_BY]->(sp)
        SET r.confidence = 1.0, r.method = 'human', r.locked = true
        """
        result = await tx.run(
            query,
            aggregate_id=event.aggregate_id,
            new_speaker_id=data["new_speaker_id"],
        )
        _raise_if_no_writes(await result.consume(), "SpeakerReattributed", event.aggregate_id)
