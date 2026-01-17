"""
Handlers for Interview-related events.

Handles InterviewCreated, InterviewUpdated, StatusChanged, etc.
"""

import logging

from src.events.envelope import EventEnvelope
from src.utils.metrics import metrics_tracker

from .base_handler import BaseProjectionHandler

logger = logging.getLogger(__name__)


class InterviewCreatedHandler(BaseProjectionHandler):
    """Handler for InterviewCreated events."""

    async def apply(self, tx, event: EventEnvelope):
        """
        Create Interview and Project nodes.

        Uses MERGE instead of CREATE to handle dual-write scenario where pipeline
        may have already created interview directly in Neo4j. Only updates if the
        event version is newer than what's currently stored.

        Args:
            tx: Neo4j transaction
            event: InterviewCreated event
        """
        data = event.data

        query = """
        // Create or match project
        MERGE (p:Project {project_id: $project_id})
        ON CREATE SET
            p.created_at = datetime($created_at)

        // MERGE instead of CREATE for dual-write safety
        MERGE (i:Interview {interview_id: $interview_id})

        // Only update if event version is newer (or not set)
        WITH i, p
        WHERE i.event_version IS NULL OR i.event_version < $event_version

        SET
            i.aggregate_id = $aggregate_id,
            i.title = $title,
            i.source = $source,
            i.language = $language,
            i.status = $status,
            i.created_at = datetime($created_at),
            i.updated_at = datetime($updated_at),
            i.event_version = $event_version,
            i.source_writer = $source_writer

        // Link to project
        MERGE (p)-[:CONTAINS_INTERVIEW]->(i)
        """

        result = await tx.run(
            query,
            project_id=data.get("project_id", "default"),
            interview_id=event.aggregate_id,
            aggregate_id=event.aggregate_id,
            title=data.get("title", "Untitled"),
            source=data.get("source", ""),
            language=data.get("language"),
            status=data.get("status", "created"),
            created_at=data.get("created_at", event.occurred_at.isoformat()),
            updated_at=data.get("updated_at", event.occurred_at.isoformat()),
            event_version=event.version,
            source_writer="projection_service",  # Track that projection wrote this
        )

        # Track deduplication metrics
        # Check if properties were actually updated (WHERE clause passed)
        summary = await result.consume()
        if summary.counters.properties_set > 0:
            # Properties were updated - either new node or version was newer
            logger.info(f"Applied InterviewCreated event (v{event.version}) for interview {event.aggregate_id}")
        else:
            # No properties updated - likely a duplicate with same/newer version already present
            logger.debug(
                f"Skipped InterviewCreated event (v{event.version}) for interview {event.aggregate_id} "
                f"- existing version is equal or newer (deduplication)"
            )
            metrics_tracker.increment_projection_duplicate_skipped("Interview")


class InterviewMetadataUpdatedHandler(BaseProjectionHandler):
    """Handler for InterviewMetadataUpdated events."""

    async def apply(self, tx, event: EventEnvelope):
        """
        Update Interview metadata.

        Args:
            tx: Neo4j transaction
            event: InterviewMetadataUpdated event
        """
        data = event.data

        # Build SET clause dynamically based on what's provided
        updates = []
        params = {
            "aggregate_id": event.aggregate_id,
            "updated_at": event.occurred_at.isoformat(),
        }

        if "title" in data and data["title"] is not None:
            updates.append("i.title = $title")
            params["title"] = data["title"]

        if "language" in data and data["language"] is not None:
            updates.append("i.language = $language")
            params["language"] = data["language"]

        if "metadata_diff" in data:
            # For now, we'll skip complex metadata updates
            # In a real implementation, you'd merge the metadata JSON
            pass

        if not updates:
            logger.debug(f"No updates to apply for event {event.event_id}")
            return

        updates.append("i.updated_at = datetime($updated_at)")
        set_clause = ", ".join(updates)

        query = f"""
        MATCH (i:Interview {{aggregate_id: $aggregate_id}})
        SET {set_clause}
        """

        await tx.run(query, **params)

        logger.info(f"Updated Interview metadata for {event.aggregate_id}")


class InterviewStatusChangedHandler(BaseProjectionHandler):
    """Handler for StatusChanged events."""

    async def apply(self, tx, event: EventEnvelope):
        """
        Update Interview status.

        Args:
            tx: Neo4j transaction
            event: StatusChanged event
        """
        data = event.data

        query = """
        MATCH (i:Interview {aggregate_id: $aggregate_id})
        SET
            i.status = $new_status,
            i.updated_at = datetime($updated_at)
        """

        await tx.run(
            query,
            aggregate_id=event.aggregate_id,
            new_status=data.get("new_status", "unknown"),
            updated_at=event.occurred_at.isoformat(),
        )

        logger.info(f"Changed Interview {event.aggregate_id} status to " f"{data.get('new_status')}")
