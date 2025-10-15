"""
Handlers for Sentence-related events.

Handles SentenceCreated, SentenceEdited, AnalysisGenerated, AnalysisOverridden, etc.
"""

import logging
from typing import List

from src.events.envelope import EventEnvelope

from .base_handler import BaseProjectionHandler

logger = logging.getLogger(__name__)


class SentenceCreatedHandler(BaseProjectionHandler):
    """Handler for SentenceCreated events."""

    async def apply(self, tx, event: EventEnvelope):
        """
        Create Sentence node and link to Interview.

        Args:
            tx: Neo4j transaction
            event: SentenceCreated event
        """
        data = event.data

        query = """
        // Match the interview
        MATCH (i:Interview {interview_id: $interview_id})

        // Create sentence
        CREATE (s:Sentence {
            sentence_id: $sentence_id,
            aggregate_id: $aggregate_id,
            text: $text,
            sequence_order: $sequence_order,
            speaker: $speaker,
            start_ms: $start_ms,
            end_ms: $end_ms,
            status: $status,
            is_edited: false,
            created_at: datetime($created_at),
            updated_at: datetime($updated_at),
            event_version: $event_version
        })

        // Link to interview
        MERGE (i)-[:HAS_SENTENCE]->(s)
        """

        await tx.run(
            query,
            interview_id=data.get("interview_id"),
            sentence_id=event.aggregate_id,
            aggregate_id=event.aggregate_id,
            text=data.get("text", ""),
            sequence_order=data.get("index", 0),
            speaker=data.get("speaker"),
            start_ms=data.get("start_ms"),
            end_ms=data.get("end_ms"),
            status=data.get("status", "created"),
            created_at=data.get("created_at", event.occurred_at.isoformat()),
            updated_at=data.get("updated_at", event.occurred_at.isoformat()),
            event_version=event.version,
        )

        logger.info(
            f"Created Sentence node {event.aggregate_id} for interview "
            f"{data.get('interview_id')}"
        )


class SentenceEditedHandler(BaseProjectionHandler):
    """Handler for SentenceEdited events."""

    async def apply(self, tx, event: EventEnvelope):
        """
        Update Sentence text and set edited flag.

        Args:
            tx: Neo4j transaction
            event: SentenceEdited event
        """
        data = event.data

        query = """
        MATCH (s:Sentence {aggregate_id: $aggregate_id})
        SET
            s.text = $new_text,
            s.is_edited = true,
            s.edited_by = $editor_type,
            s.updated_at = datetime($updated_at)
        """

        await tx.run(
            query,
            aggregate_id=event.aggregate_id,
            new_text=data.get("new_text", ""),
            editor_type=data.get("editor_type", "unknown"),
            updated_at=event.occurred_at.isoformat(),
        )

        logger.info(f"Updated Sentence {event.aggregate_id} text (edited)")


class AnalysisGeneratedHandler(BaseProjectionHandler):
    """Handler for AnalysisGenerated events."""

    async def apply(self, tx, event: EventEnvelope):
        """
        Create Analysis node and link dimension nodes.

        Args:
            tx: Neo4j transaction
            event: AnalysisGenerated event
        """
        data = event.data

        # Create Analysis node
        query_analysis = """
        MATCH (s:Sentence {aggregate_id: $aggregate_id})

        CREATE (a:Analysis {
            analysis_id: $analysis_id,
            model: $model,
            model_version: $model_version,
            confidence: $confidence,
            raw_ref: $raw_ref,
            is_overridden: false,
            created_at: datetime($created_at)
        })

        MERGE (s)-[:HAS_ANALYSIS]->(a)
        """

        analysis_id = f"{event.aggregate_id}-analysis-{event.version}"

        await tx.run(
            query_analysis,
            aggregate_id=event.aggregate_id,
            analysis_id=analysis_id,
            model=data.get("model", "unknown"),
            model_version=data.get("model_version", "unknown"),
            confidence=data.get("confidence"),
            raw_ref=data.get("raw_ref"),
            created_at=event.occurred_at.isoformat(),
        )

        # Link classification dimensions
        classification = data.get("classification", {})
        await self._link_classification(tx, event.aggregate_id, classification)

        # Link keywords
        keywords = data.get("keywords", [])
        await self._link_dimension_list(
            tx, event.aggregate_id, "Keyword", "MENTIONS_OVERALL_KEYWORD", keywords
        )

        # Link topics
        topics = data.get("topics", [])
        await self._link_dimension_list(
            tx, event.aggregate_id, "Topic", "MENTIONS_TOPIC", topics
        )

        # Link domain keywords
        domain_keywords = data.get("domain_keywords", [])
        await self._link_dimension_list(
            tx, event.aggregate_id, "DomainKeyword", "MENTIONS_DOMAIN_KEYWORD", domain_keywords
        )

        logger.info(f"Generated analysis for Sentence {event.aggregate_id}")

    async def _link_classification(self, tx, sentence_id: str, classification: dict):
        """Link classification dimension nodes."""
        # Function type
        function_type = classification.get("function_type")
        if function_type:
            query = """
            MATCH (s:Sentence {aggregate_id: $sentence_id})
            MERGE (ft:FunctionType {name: $name})
            MERGE (s)-[:HAS_FUNCTION_TYPE]->(ft)
            """
            await tx.run(query, sentence_id=sentence_id, name=function_type)

        # Structure type
        structure_type = classification.get("structure_type")
        if structure_type:
            query = """
            MATCH (s:Sentence {aggregate_id: $sentence_id})
            MERGE (st:StructureType {name: $name})
            MERGE (s)-[:HAS_STRUCTURE_TYPE]->(st)
            """
            await tx.run(query, sentence_id=sentence_id, name=structure_type)

        # Purpose
        purpose = classification.get("purpose")
        if purpose:
            query = """
            MATCH (s:Sentence {aggregate_id: $sentence_id})
            MERGE (p:Purpose {name: $name})
            MERGE (s)-[:HAS_PURPOSE]->(p)
            """
            await tx.run(query, sentence_id=sentence_id, name=purpose)

    async def _link_dimension_list(
        self, tx, sentence_id: str, node_label: str, rel_type: str, values: List[str]
    ):
        """Link a list of dimension nodes."""
        if not values:
            return

        for value in values:
            query = f"""
            MATCH (s:Sentence {{aggregate_id: $sentence_id}})
            MERGE (d:{node_label} {{name: $name}})
            MERGE (s)-[:{rel_type}]->(d)
            """
            await tx.run(query, sentence_id=sentence_id, name=value)


class AnalysisOverriddenHandler(BaseProjectionHandler):
    """Handler for AnalysisOverridden events."""

    async def apply(self, tx, event: EventEnvelope):
        """
        Update Analysis with override data.

        Args:
            tx: Neo4j transaction
            event: AnalysisOverridden event
        """
        data = event.data

        # Mark analysis as overridden
        query_analysis = """
        MATCH (s:Sentence {aggregate_id: $aggregate_id})-[:HAS_ANALYSIS]->(a:Analysis)
        SET
            a.is_overridden = true,
            a.override_note = $note,
            a.overridden_at = datetime($overridden_at),
            a.overridden_by = $actor
        """

        actor_info = event.actor.display if event.actor else "unknown"

        await tx.run(
            query_analysis,
            aggregate_id=event.aggregate_id,
            note=data.get("note", ""),
            overridden_at=event.occurred_at.isoformat(),
            actor=actor_info,
        )

        # Update overridden fields
        # For simplicity, we'll store the override data as JSON
        # In a real implementation, you'd update the specific dimension relationships

        logger.info(
            f"Overrode analysis for Sentence {event.aggregate_id} "
            f"(by: {actor_info})"
        )
