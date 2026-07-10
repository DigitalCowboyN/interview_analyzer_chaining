"""Projection handlers for entity-mention events (Layer 2 enrichment)."""

import logging

from src.events.envelope import EventEnvelope

from .base_handler import BaseProjectionHandler
from .speaker_handlers import _raise_if_no_writes

logger = logging.getLogger(__name__)


class EntitiesExtractedHandler(BaseProjectionHandler):
    """Materializes Entity nodes and span-grounded MENTIONS edges.

    Re-extraction replaces the fragment's MENTIONS set. A guard statement
    stamps the Sentence node first so a missing Sentence (out-of-order
    delivery) raises for retry/park instead of silently dropping — an empty
    extraction on an existing Sentence legitimately creates no edges.
    """

    async def apply(self, tx, event: EventEnvelope):
        data = event.data

        guard_query = """
        MATCH (s:Sentence {aggregate_id: $aggregate_id})
        SET s.entities_extracted_at = datetime($occurred_at),
            s.entities_model = $model,
            s.entities_provider = $provider
        """
        result = await tx.run(
            guard_query,
            aggregate_id=event.aggregate_id,
            occurred_at=event.occurred_at.isoformat(),
            model=data.get("model"),
            provider=data.get("provider"),
        )
        _raise_if_no_writes(await result.consume(), "EntitiesExtracted", event.aggregate_id)

        edges_query = """
        MATCH (s:Sentence {aggregate_id: $aggregate_id})
        OPTIONAL MATCH (s)-[old:MENTIONS]->(:Entity)
        DELETE old
        WITH DISTINCT s
        UNWIND $entities AS ent
        MERGE (e:Entity {surface: toLower(ent.text), entity_type: ent.entity_type})
        MERGE (s)-[m:MENTIONS {start: ent.start, end: ent.end}]->(e)
        SET m.text = ent.text, m.confidence = ent.confidence
        """
        await tx.run(
            edges_query,
            aggregate_id=event.aggregate_id,
            entities=data.get("entities", []),
        )
        logger.info(
            f"Applied EntitiesExtracted for sentence {event.aggregate_id} "
            f"({len(data.get('entities', []))} mentions)"
        )
