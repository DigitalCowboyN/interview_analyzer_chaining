"""Projection handlers for utterance stitching events (overlay on fragments)."""

import logging

from src.events.envelope import EventEnvelope

from .base_handler import BaseProjectionHandler

logger = logging.getLogger(__name__)


class UtteranceIdentifiedHandler(BaseProjectionHandler):
    """Creates an Utterance node and PART_OF_UTTERANCE overlay edges."""

    async def apply(self, tx, event: EventEnvelope):
        data = event.data
        fragments = [
            {"id": fid, "position": pos} for pos, fid in enumerate(data["fragment_ids"])
        ]
        query = """
        MATCH (sp:Speaker {speaker_id: $speaker_id})
        MERGE (u:Utterance {utterance_id: $utterance_id})
        SET u.interview_id = $interview_id, u.confidence = $confidence
        MERGE (sp)-[:SPOKE]->(u)
        WITH u
        UNWIND $fragments AS frag
        MATCH (s:Sentence {aggregate_id: frag.id})
        MERGE (s)-[p:PART_OF_UTTERANCE]->(u)
        SET p.position = frag.position
        """
        result = await tx.run(
            query,
            utterance_id=data["utterance_id"],
            speaker_id=data["speaker_id"],
            interview_id=event.aggregate_id,
            confidence=data["confidence"],
            fragments=fragments,
        )
        summary = await result.consume()
        counters = summary.counters
        if (
            counters.nodes_created == 0
            and counters.properties_set == 0
            and counters.relationships_created == 0
        ):
            # Initial MATCH found no Speaker: likely out-of-order delivery.
            # Raise so the base class retries and eventually parks the event
            # instead of silently consuming it.
            raise ValueError(
                f"UtteranceIdentified {data['utterance_id']}: no writes applied "
                f"(Speaker {data['speaker_id']} not yet projected?)"
            )
        logger.info(f"Applied UtteranceIdentified for utterance {data['utterance_id']}")


class InterruptionRecordedHandler(BaseProjectionHandler):
    """Creates INTERRUPTS edge between utterances."""

    async def apply(self, tx, event: EventEnvelope):
        data = event.data
        query = """
        MATCH (a:Utterance {utterance_id: $interrupting})
        MATCH (b:Utterance {utterance_id: $interrupted})
        MERGE (a)-[r:INTERRUPTS]->(b)
        SET r.at_fragment_id = $at_fragment_id
        """
        await tx.run(
            query,
            interrupting=data["interrupting_utterance_id"],
            interrupted=data["interrupted_utterance_id"],
            at_fragment_id=data["at_fragment_id"],
        )


class StitchRemovedHandler(BaseProjectionHandler):
    """Human correction: removes an utterance overlay node entirely."""

    async def apply(self, tx, event: EventEnvelope):
        utterance_id = event.data["utterance_id"]
        query = """
        MATCH (u:Utterance {utterance_id: $utterance_id})
        DETACH DELETE u
        """
        await tx.run(query, utterance_id=utterance_id)
        logger.info(f"Applied StitchRemoved for utterance {utterance_id}")
