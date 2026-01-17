"""
Handlers for Sentence-related events.

Handles SentenceCreated, SentenceEdited, AnalysisGenerated, AnalysisOverridden, etc.
"""

import logging
from typing import List

from src.events.envelope import EventEnvelope
from src.utils.metrics import metrics_tracker

from .base_handler import BaseProjectionHandler

logger = logging.getLogger(__name__)


class SentenceCreatedHandler(BaseProjectionHandler):
    """Handler for SentenceCreated events."""

    async def apply(self, tx, event: EventEnvelope):
        """
        Create Sentence node and link to Interview.

        Uses MERGE instead of CREATE to handle dual-write scenario where pipeline
        may have already written sentence directly to Neo4j. Only updates if the
        event version is newer than what's currently stored.

        Args:
            tx: Neo4j transaction
            event: SentenceCreated event
        """
        data = event.data

        query = """
        // Match the interview
        MATCH (i:Interview {interview_id: $interview_id})

        // MERGE instead of CREATE for dual-write safety
        // This allows both direct writes and projection writes to coexist
        MERGE (s:Sentence {sentence_id: $sentence_id})

        // Only update if event version is newer (or not set)
        // This ensures projection service doesn't overwrite newer data
        WITH s, i
        WHERE s.event_version IS NULL OR s.event_version < $event_version

        SET
            s.aggregate_id = $aggregate_id,
            s.text = $text,
            s.sequence_order = $sequence_order,
            s.speaker = $speaker,
            s.start_ms = $start_ms,
            s.end_ms = $end_ms,
            s.status = $status,
            s.is_edited = false,
            s.created_at = datetime($created_at),
            s.updated_at = datetime($updated_at),
            s.event_version = $event_version,
            s.source = $source

        // Link to interview
        MERGE (i)-[:HAS_SENTENCE]->(s)
        """

        result = await tx.run(
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
            source="projection_service",  # Track that projection wrote this
        )

        # Track deduplication metrics
        # Check if properties were actually updated (WHERE clause passed)
        summary = await result.consume()
        if summary.counters.properties_set > 0:
            # Properties were updated - either new node or version was newer
            logger.info(f"Applied SentenceCreated event (v{event.version}) for sentence {event.aggregate_id}")
        else:
            # No properties updated - likely a duplicate with same/newer version already present
            logger.debug(
                f"Skipped SentenceCreated event (v{event.version}) for sentence {event.aggregate_id} "
                f"- existing version is equal or newer (deduplication)"
            )
            metrics_tracker.increment_projection_duplicate_skipped("Sentence")


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

        # Delete old Analysis node and create new one (overwrite behavior)
        # M2.8: Multiple AnalysisGenerated events can occur for same sentence.
        # Latest event should replace previous analysis.
        # BUT: Preserve user edits (relationships with is_edited=true)

        # First, check for user-edited relationships before deleting
        query_find_edited = """
        MATCH (s:Sentence {aggregate_id: $aggregate_id})-[:HAS_ANALYSIS]->(old_a:Analysis)
        OPTIONAL MATCH (old_a)-[r_func:HAS_FUNCTION {is_edited: true}]->(ft:FunctionType)
        OPTIONAL MATCH (old_a)-[r_struct:HAS_STRUCTURE {is_edited: true}]->(st:StructureType)
        OPTIONAL MATCH (old_a)-[r_purp:HAS_PURPOSE {is_edited: true}]->(p:Purpose)
        RETURN
            ft.name as edited_function,
            st.name as edited_structure,
            p.name as edited_purpose
        """
        result = await tx.run(query_find_edited, aggregate_id=event.aggregate_id)
        record = await result.single()

        # Store edited values if they exist
        edited_dims = {}
        if record:
            if record["edited_function"]:
                edited_dims["function_type"] = record["edited_function"]
            if record["edited_structure"]:
                edited_dims["structure_type"] = record["edited_structure"]
            if record["edited_purpose"]:
                edited_dims["purpose"] = record["edited_purpose"]

        # Now delete any existing Analysis nodes
        query_delete_old = """
        MATCH (s:Sentence {aggregate_id: $aggregate_id})-[:HAS_ANALYSIS]->(old_a:Analysis)
        DETACH DELETE old_a
        """
        result = await tx.run(query_delete_old, aggregate_id=event.aggregate_id)
        summary = await result.consume()
        if summary.counters.nodes_deleted > 0:
            logger.info(f"Deleted {summary.counters.nodes_deleted} old Analysis node(s) for sentence {event.aggregate_id}")

        # Then create new Analysis node
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

        # Link classification dimensions using specific analysis_id
        # Pass edited_dims to preserve user edits
        classification = data.get("classification", {})
        await self._link_classification(tx, event.aggregate_id, analysis_id, classification, edited_dims)

        # Link keywords
        keywords = data.get("keywords", [])
        await self._link_dimension_list(tx, event.aggregate_id, analysis_id, "Keyword", "MENTIONS_OVERALL_KEYWORD", keywords)

        # Link topics
        topics = data.get("topics", [])
        await self._link_dimension_list(tx, event.aggregate_id, analysis_id, "Topic", "MENTIONS_TOPIC", topics)

        # Link domain keywords
        domain_keywords = data.get("domain_keywords", [])
        await self._link_dimension_list(
            tx, event.aggregate_id, analysis_id, "DomainKeyword", "MENTIONS_DOMAIN_KEYWORD", domain_keywords
        )

        logger.info(f"Generated analysis for Sentence {event.aggregate_id}")

    async def _link_classification(self, tx, sentence_id: str, analysis_id: str, classification: dict, edited_dims: dict = None):
        """Link classification dimension nodes to specific Analysis, preserving user edits."""
        edited_dims = edited_dims or {}

        # Function type - use edited value if present, otherwise use event value
        function_type = edited_dims.get("function_type") or classification.get("function_type")
        is_func_edited = "function_type" in edited_dims
        if function_type:
            query = """
            MATCH (a:Analysis {analysis_id: $analysis_id})
            MERGE (ft:FunctionType {name: $name})
            MERGE (a)-[r:HAS_FUNCTION]->(ft)
            SET r.is_edited = $is_edited
            """
            await tx.run(query, analysis_id=analysis_id, name=function_type, is_edited=is_func_edited)

        # Structure type - use edited value if present, otherwise use event value
        structure_type = edited_dims.get("structure_type") or classification.get("structure_type")
        is_struct_edited = "structure_type" in edited_dims
        if structure_type:
            query = """
            MATCH (a:Analysis {analysis_id: $analysis_id})
            MERGE (st:StructureType {name: $name})
            MERGE (a)-[r:HAS_STRUCTURE]->(st)
            SET r.is_edited = $is_edited
            """
            await tx.run(query, analysis_id=analysis_id, name=structure_type, is_edited=is_struct_edited)

        # Purpose - use edited value if present, otherwise use event value
        purpose = edited_dims.get("purpose") or classification.get("purpose")
        is_purp_edited = "purpose" in edited_dims
        if purpose:
            query = """
            MATCH (a:Analysis {analysis_id: $analysis_id})
            MERGE (p:Purpose {name: $name})
            MERGE (a)-[r:HAS_PURPOSE]->(p)
            SET r.is_edited = $is_edited
            """
            await tx.run(query, analysis_id=analysis_id, name=purpose, is_edited=is_purp_edited)

    async def _link_dimension_list(self, tx, sentence_id: str, analysis_id: str, node_label: str, rel_type: str, values: List[str]):
        """Link a list of dimension nodes from specific Analysis."""
        if not values:
            return

        # Determine property name based on node label
        # Keyword and DomainKeyword use 'text', others use 'name'
        prop_name = "text" if node_label in ["Keyword", "DomainKeyword"] else "name"

        for value in values:
            # Set additional properties based on node type
            if node_label == "DomainKeyword":
                query = f"""
                MATCH (a:Analysis {{analysis_id: $analysis_id}})
                MERGE (d:{node_label} {{{prop_name}: $value}})
                ON CREATE SET d.is_custom = false
                MERGE (a)-[r:{rel_type} {{is_edited: false}}]->(d)
                """
            else:
                query = f"""
                MATCH (a:Analysis {{analysis_id: $analysis_id}})
                MERGE (d:{node_label} {{{prop_name}: $value}})
                MERGE (a)-[r:{rel_type} {{is_edited: false}}]->(d)
                """
            await tx.run(query, analysis_id=analysis_id, value=value)


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

        logger.info(f"Overrode analysis for Sentence {event.aggregate_id} " f"(by: {actor_info})")
