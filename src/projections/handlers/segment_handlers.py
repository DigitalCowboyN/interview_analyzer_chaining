"""Projection handlers for Interview-stream segment events (M4.5c).

Overlay node: (:Segment {segment_id, topic, interview_id, confidence});
one CONTAINS edge per fragment in [start_index, end_index].

Ordering guards raise on short RETURN counts (never write counters — edge
MERGEs are no-ops on replay) so the base handler retries and eventually
parks: SegmentIdentified needs every fragment in its range projected;
SegmentRemoved needs its segment projected (if the identify parked, the
remove must park too, or replaying the parked identify would resurrect a
removed segment).
"""

from src.events.envelope import EventEnvelope
from src.projections.handlers.base_handler import BaseProjectionHandler
from src.utils.logger import get_logger

logger = get_logger()


class SegmentIdentifiedHandler(BaseProjectionHandler):
    """MERGE the segment node and rebuild its CONTAINS edges from the range."""

    async def apply(self, tx, event: EventEnvelope):
        d = event.data
        expected = d["end_index"] - d["start_index"] + 1
        query = """
        MERGE (seg:Segment {segment_id: $segment_id})
        SET seg.topic = $topic,
            seg.interview_id = $interview_id,
            seg.confidence = $confidence
        WITH seg
        OPTIONAL MATCH (seg)-[old:CONTAINS]->(:Fragment)
        DELETE old
        WITH DISTINCT seg
        MATCH (i:Interview {interview_id: $interview_id})-[:HAS_SENTENCE]->(f:Fragment)
        WHERE f.sequence_order >= $start_index AND f.sequence_order <= $end_index
        MERGE (seg)-[:CONTAINS]->(f)
        RETURN count(f) AS linked
        """
        result = await tx.run(
            query,
            segment_id=d["segment_id"], topic=d["topic"],
            interview_id=event.aggregate_id, confidence=d["confidence"],
            start_index=d["start_index"], end_index=d["end_index"],
        )
        record = await result.single()
        linked = record["linked"] if record else 0
        if linked < expected:
            raise ValueError(
                f"SegmentIdentified {d['segment_id']}: only {linked}/{expected} "
                f"fragments projected yet"
            )
        logger.info(
            f"Applied SegmentIdentified {d['segment_id']} "
            f"({d['start_index']}..{d['end_index']}) for interview {event.aggregate_id}"
        )


class SegmentRemovedHandler(BaseProjectionHandler):
    """Human correction: DETACH DELETE the segment node."""

    async def apply(self, tx, event: EventEnvelope):
        d = event.data
        query = """
        MATCH (seg:Segment {segment_id: $segment_id})
        WITH seg, count(seg) AS found
        DETACH DELETE seg
        RETURN found
        """
        result = await tx.run(query, segment_id=d["segment_id"])
        record = await result.single()
        found = record["found"] if record else 0
        if found == 0:
            raise ValueError(
                f"SegmentRemoved {d['segment_id']}: segment not projected yet"
            )
        logger.info(f"Applied SegmentRemoved {d['segment_id']}")
