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
from src.events.store import EventStoreClient
from src.utils.logger import get_logger

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
        Emit InterviewCreated event (non-blocking).

        Args:
            interview_id: Interview UUID
            project_id: Project UUID
            title: Interview title (usually filename)
            source: Source file path
            language: Language code (default: "en")
            correlation_id: Correlation ID for event tracking
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

        except Exception as e:
            # Log but don't raise - Neo4j write already succeeded
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
        Emit SentenceCreated event (non-blocking).

        Args:
            interview_id: Interview UUID
            index: Sentence index (sequence_order)
            text: Sentence text
            speaker: Optional speaker identifier
            start_ms: Optional start time in milliseconds
            end_ms: Optional end time in milliseconds
            correlation_id: Correlation ID for event tracking
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

        except Exception as e:
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

    async def emit_analysis_generated(
        self,
        interview_id: str,
        sentence_index: int,
        analysis_data: Dict[str, Any],
        correlation_id: Optional[str] = None,
    ):
        """
        Emit AnalysisGenerated event (non-blocking).

        Args:
            interview_id: Interview UUID
            sentence_index: Sentence index (sequence_order)
            analysis_data: Analysis result dictionary from SentenceAnalyzer
            correlation_id: Correlation ID for event tracking
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

            event = create_analysis_generated_event(
                aggregate_id=sentence_id,
                version=1,  # Assumes SentenceCreated was version 0
                model=analysis_data.get("model", "gpt-4"),
                model_version=analysis_data.get("model_version", "1.0"),
                classification=classification,
                keywords=analysis_data.get("overall_keywords", []),
                topics=analysis_data.get("topics", []),
                domain_keywords=analysis_data.get("domain_keywords", []),
                confidence=analysis_data.get("confidence", 1.0),
                actor=Actor(actor_type=ActorType.SYSTEM, user_id="pipeline"),
                correlation_id=correlation_id,
            )

            stream_name = f"Sentence-{sentence_id}"
            # Don't use expected_version - we don't track versions during dual-write
            await self.client.append_events(stream_name, [event], expected_version=-1)
            self.logger.debug(
                f"Emitted AnalysisGenerated event for sentence {sentence_id} "
                f"(interview={interview_id}, index={sentence_index}, correlation_id={correlation_id})"
            )

        except Exception as e:
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

    async def emit_interview_created_async(self, *args, **kwargs):
        """Fire-and-forget wrapper for emit_interview_created."""
        asyncio.create_task(self.emit_interview_created(*args, **kwargs))

    async def emit_sentence_created_async(self, *args, **kwargs):
        """Fire-and-forget wrapper for emit_sentence_created."""
        asyncio.create_task(self.emit_sentence_created(*args, **kwargs))

    async def emit_analysis_generated_async(self, *args, **kwargs):
        """Fire-and-forget wrapper for emit_analysis_generated."""
        asyncio.create_task(self.emit_analysis_generated(*args, **kwargs))
