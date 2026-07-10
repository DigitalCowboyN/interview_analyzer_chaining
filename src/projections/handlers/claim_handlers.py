"""Projection handlers for claim events (Layer 2 enrichment)."""

import logging

from src.events.envelope import EventEnvelope

from .base_handler import BaseProjectionHandler

logger = logging.getLogger(__name__)


class ClaimExtractedHandler(BaseProjectionHandler):
    """Materializes Claim nodes with MADE_BY speaker and SUPPORTED_BY fragments."""

    async def apply(self, tx, event: EventEnvelope):
        data = event.data
        query = """
        MATCH (sp:Speaker {speaker_id: $speaker_id})
        MATCH (u:Utterance {utterance_id: $utterance_id})
        MERGE (c:Claim {claim_id: $claim_id})
        SET c.text = $text, c.kind = $kind, c.confidence = $confidence,
            c.model = $model, c.provider = $provider, c.interview_id = $interview_id
        MERGE (c)-[:MADE_BY]->(sp)
        WITH c, u
        MATCH (s:Fragment)-[:PART_OF_UTTERANCE]->(u)
        MERGE (c)-[:SUPPORTED_BY]->(s)
        RETURN count(s) AS supported
        """
        result = await tx.run(
            query,
            claim_id=data["claim_id"],
            utterance_id=data["utterance_id"],
            speaker_id=data["speaker_id"],
            text=data["text"],
            kind=data["kind"],
            confidence=data["confidence"],
            model=data["model"],
            provider=data["provider"],
            interview_id=event.aggregate_id,
        )
        record = await result.single()
        supported = record["supported"] if record is not None else 0
        if supported == 0:
            # Speaker/Utterance/fragments arrive on independent subscriptions;
            # raise so retry/park engages instead of sealing a support-less claim.
            raise ValueError(
                f"ClaimExtracted {data['claim_id']}: targets not yet projected "
                f"(speaker {data['speaker_id']} / utterance {data['utterance_id']})"
            )
        logger.info(f"Applied ClaimExtracted {data['claim_id']} ({supported} supports)")
