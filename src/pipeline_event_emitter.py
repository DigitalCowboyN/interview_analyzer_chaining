"""
Pipeline Event Emitter for Dual-Write Phase.

Lightweight wrapper that emits events to EventStoreDB during pipeline processing.
Used during dual-write phase to emit events alongside Neo4j writes.

This is NOT the command layer - it's a direct event emitter for system-generated
events during file processing. User-driven events use the command layer.
"""

import asyncio
import uuid
from typing import Any, Dict, Optional

from src.agents.agent_factory import agent
from src.events.envelope import Actor, ActorType
from src.events.interview_events import (
    InterviewStatus,
    create_interview_created_event,
    create_status_changed_event,
)
from src.events.sentence_events import (
    create_analysis_generated_event,
    create_sentence_created_event,
)
from src.events.store import EventStoreClient, StreamState
from src.utils.logger import get_logger
from src.utils.metrics import metrics_tracker

logger = get_logger()


class PipelineEventEmitter:
    """
    Emits events to EventStoreDB during pipeline processing.

    Lightweight wrapper that creates events directly (no commands/aggregates).
    Used for dual-write phase to emit events alongside Neo4j writes.

    Key Design Decisions:
    - Non-blocking: Catches all exceptions, logs errors, continues
    - Fire-and-forget: Uses asyncio.create_task() for async execution
    - Deterministic IDs: Uses uuid5 for sentence IDs (interview_id:index)
    - Actor tracking: All events tagged as SYSTEM actor with "pipeline" user_id
    """

    def __init__(self, event_store_client: EventStoreClient):
        """
        Initialize the event emitter.

        Args:
            event_store_client: EventStoreDB client for event persistence
        """
        self.client = event_store_client
        self.logger = get_logger()

    def _generate_sentence_uuid(self, interview_id: str, index: int) -> str:
        """
        Generate deterministic UUID for sentence.

        Uses uuid5 to create consistent UUIDs based on interview_id and index.
        This ensures the same sentence always gets the same UUID.

        Args:
            interview_id: Interview UUID
            index: Sentence index (sequence_order)

        Returns:
            UUID string for the sentence
        """
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{interview_id}:{index}"))

    async def emit_interview_created(
        self,
        interview_id: str,
        project_id: str,
        title: str,
        source: str,
        language: str = "en",
        correlation_id: Optional[str] = None,
    ):
        """
        Emit InterviewCreated event (BLOCKING - raises on failure).

        Event-first dual-write: This method raises exceptions on failure.
        The caller (pipeline) must handle the exception and abort the operation.

        Args:
            interview_id: Interview UUID
            project_id: Project UUID
            title: Interview title (usually filename)
            source: Source file path
            language: Language code (default: "en")
            correlation_id: Correlation ID for event tracking

        Raises:
            Exception: If event emission fails (must be handled by caller)
        """
        try:
            event = create_interview_created_event(
                aggregate_id=interview_id,
                version=0,  # First event for this interview
                title=title,
                source=source,
                language=language,
                actor=Actor(actor_type=ActorType.SYSTEM, user_id="pipeline"),
                correlation_id=correlation_id,
                project_id=project_id,
            )

            stream_name = f"Interview-{interview_id}"
            await self.client.append_events(stream_name, [event], expected_version=-1)
            self.logger.debug(
                f"Emitted InterviewCreated event for interview {interview_id} " f"(correlation_id={correlation_id})"
            )

            # Track success metric
            metrics_tracker.increment_event_emission_success("InterviewCreated")

        except Exception as e:
            # Track failure metric
            metrics_tracker.increment_event_emission_failure("InterviewCreated")

            # Log error with context
            self.logger.error(
                f"Failed to emit InterviewCreated event for interview {interview_id}: {e}",
                exc_info=True,
                extra={
                    "interview_id": interview_id,
                    "project_id": project_id,
                    "correlation_id": correlation_id,
                    "event_type": "InterviewCreated",
                },
            )

            # RE-RAISE for event-first dual-write behavior
            # Caller must handle and abort operation
            raise

    async def emit_interview_status_changed(
        self,
        interview_id: str,
        new_status: str,
        correlation_id: Optional[str] = None,
    ):
        """
        Emit InterviewStatusChanged event (non-blocking).

        Args:
            interview_id: Interview UUID
            new_status: New status (e.g., "completed", "failed")
            correlation_id: Correlation ID for event tracking
        """
        try:
            # Note: We don't track versions in dual-write mode
            # Use version 1+ for status change (assuming InterviewCreated was version 0)
            # Convert string status to enum
            to_status_enum = InterviewStatus(new_status) if isinstance(new_status, str) else new_status
            event = create_status_changed_event(
                aggregate_id=interview_id,
                version=1,  # TODO: Track actual version in dual-write
                from_status=InterviewStatus.PROCESSING,  # Assumption
                to_status=to_status_enum,
                actor=Actor(actor_type=ActorType.SYSTEM, user_id="pipeline"),
                correlation_id=correlation_id,
            )

            stream_name = f"Interview-{interview_id}"
            # Don't use expected_version here - we don't know the current version
            await self.client.append_events(stream_name, [event], expected_version=-1)
            self.logger.debug(
                f"Emitted InterviewStatusChanged event for interview {interview_id} "
                f"(status={new_status}, correlation_id={correlation_id})"
            )

        except Exception as e:
            self.logger.error(
                f"Failed to emit InterviewStatusChanged event for interview {interview_id}: {e}",
                exc_info=True,
                extra={
                    "interview_id": interview_id,
                    "new_status": new_status,
                    "correlation_id": correlation_id,
                    "event_type": "InterviewStatusChanged",
                },
            )

    async def emit_sentence_created(
        self,
        interview_id: str,
        index: int,
        text: str,
        speaker: Optional[str] = None,
        start_ms: Optional[int] = None,
        end_ms: Optional[int] = None,
        correlation_id: Optional[str] = None,
    ):
        """
        Emit SentenceCreated event (BLOCKING - raises on failure).

        Event-first dual-write: This method raises exceptions on failure.
        The caller (neo4j_map_storage) must handle the exception and abort the operation.

        Args:
            interview_id: Interview UUID
            index: Sentence index (sequence_order)
            text: Sentence text
            speaker: Optional speaker identifier
            start_ms: Optional start time in milliseconds
            end_ms: Optional end time in milliseconds
            correlation_id: Correlation ID for event tracking

        Raises:
            Exception: If event emission fails (must be handled by caller)
        """
        try:
            # Generate deterministic UUID for sentence
            sentence_id = self._generate_sentence_uuid(interview_id, index)

            event = create_sentence_created_event(
                aggregate_id=sentence_id,
                version=0,  # First event for this sentence
                interview_id=interview_id,
                index=index,
                text=text,
                speaker=speaker,
                start_ms=start_ms,
                end_ms=end_ms,
                actor=Actor(actor_type=ActorType.SYSTEM, user_id="pipeline"),
                correlation_id=correlation_id,
            )

            stream_name = f"Sentence-{sentence_id}"
            await self.client.append_events(stream_name, [event], expected_version=-1)
            self.logger.debug(
                f"Emitted SentenceCreated event for sentence {sentence_id} "
                f"(interview={interview_id}, index={index}, correlation_id={correlation_id})"
            )

            # Track success metric
            metrics_tracker.increment_event_emission_success("SentenceCreated")

        except Exception as e:
            # Track failure metric
            metrics_tracker.increment_event_emission_failure("SentenceCreated")

            # Log error with context
            self.logger.error(
                f"Failed to emit SentenceCreated event for interview {interview_id}, index {index}: {e}",
                exc_info=True,
                extra={
                    "interview_id": interview_id,
                    "index": index,
                    "correlation_id": correlation_id,
                    "event_type": "SentenceCreated",
                },
            )

            # RE-RAISE for event-first dual-write behavior
            # Caller must handle and abort operation
            raise

    async def emit_analysis_generated(
        self,
        interview_id: str,
        sentence_index: int,
        analysis_data: Dict[str, Any],
        correlation_id: Optional[str] = None,
    ):
        """
        Emit AnalysisGenerated event (BLOCKING - raises on failure).

        Event-first dual-write: This method raises exceptions on failure.
        The caller (neo4j_analysis_writer) must handle the exception and abort the operation.

        Args:
            interview_id: Interview UUID
            sentence_index: Sentence index (sequence_order)
            analysis_data: Analysis result dictionary from SentenceAnalyzer
            correlation_id: Correlation ID for event tracking

        Raises:
            Exception: If event emission fails (must be handled by caller)
        """
        try:
            # Generate deterministic UUID for sentence
            sentence_id = self._generate_sentence_uuid(interview_id, sentence_index)

            # Extract analysis fields from result
            classification = {
                "function_type": analysis_data.get("function_type"),
                "structure_type": analysis_data.get("structure_type"),
                "purpose": analysis_data.get("purpose"),
            }

            # M2.8: Apply cardinality limits at event emission time
            # Keywords have a default limit of 6, others are unlimited
            keywords = analysis_data.get("overall_keywords", [])
            # First deduplicate (preserving order), then apply limit
            keywords = list(dict.fromkeys(keywords))[:6] if keywords else []

            # Determine event version by reading current stream
            # This ensures each AnalysisGenerated event gets unique version
            stream_name = f"Sentence-{sentence_id}"
            try:
                existing_events = await self.client.read_stream(stream_name)
                # Count existing events to determine next version
                next_version = len(existing_events)
            except Exception:
                # Stream doesn't exist or read failed, assume first event after SentenceCreated
                next_version = 1

            event = create_analysis_generated_event(
                aggregate_id=sentence_id,
                version=next_version,  # Increments with each analysis
                model=analysis_data.get("model", agent.get_model_name()),
                model_version=analysis_data.get("model_version", "1.0"),
                classification=classification,
                keywords=keywords,  # Limited to 6
                topics=analysis_data.get("topics", []),  # Unlimited
                domain_keywords=analysis_data.get("domain_keywords", []),  # Unlimited
                confidence=analysis_data.get("confidence", 1.0),
                actor=Actor(actor_type=ActorType.SYSTEM, user_id="pipeline"),
                correlation_id=correlation_id,
            )

            # Use StreamState.ANY to allow multiple analyses for the same sentence
            # M2.8: Multiple analyses can be generated (overwrite scenario in tests)
            await self.client.append_events(stream_name, [event], expected_version=StreamState.ANY)
            self.logger.debug(
                f"Emitted AnalysisGenerated event for sentence {sentence_id} "
                f"(interview={interview_id}, index={sentence_index}, correlation_id={correlation_id})"
            )

            # Track success metric
            metrics_tracker.increment_event_emission_success("AnalysisGenerated")

        except Exception as e:
            # Track failure metric
            metrics_tracker.increment_event_emission_failure("AnalysisGenerated")

            # Log error with context
            self.logger.error(
                f"Failed to emit AnalysisGenerated event for interview {interview_id}, "
                f"sentence index {sentence_index}: {e}",
                exc_info=True,
                extra={
                    "interview_id": interview_id,
                    "sentence_index": sentence_index,
                    "correlation_id": correlation_id,
                    "event_type": "AnalysisGenerated",
                },
            )

            # RE-RAISE for event-first dual-write behavior
            # Caller must handle and abort operation
            raise
