"""
Parked events (Dead Letter Queue) management.

Events that fail after all retry attempts are parked for manual review
and potential replay.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.events.envelope import EventEnvelope
from src.events.store import EventStoreClient, get_event_store_client

from .config import get_parked_stream_name

logger = logging.getLogger(__name__)


class ParkedEvent:
    """Represents a parked event with error context."""

    def __init__(
        self,
        original_event: EventEnvelope,
        error_message: str,
        error_type: str,
        retry_count: int,
        parked_at: datetime,
        lane_id: int,
        stack_trace: Optional[str] = None,
    ):
        """
        Initialize a parked event.

        Args:
            original_event: The event that failed
            error_message: Error message
            error_type: Type of error that occurred
            retry_count: Number of retry attempts
            parked_at: When the event was parked
            lane_id: Lane that was processing the event
            stack_trace: Optional stack trace
        """
        self.original_event = original_event
        self.error_message = error_message
        self.error_type = error_type
        self.retry_count = retry_count
        self.parked_at = parked_at
        self.lane_id = lane_id
        self.stack_trace = stack_trace

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "original_event": {
                "event_id": self.original_event.event_id,
                "event_type": self.original_event.event_type,
                "aggregate_type": self.original_event.aggregate_type,
                "aggregate_id": self.original_event.aggregate_id,
                "version": self.original_event.version,
                "occurred_at": self.original_event.occurred_at.isoformat(),
                "data": self.original_event.data,
                "actor": self.original_event.actor.model_dump() if self.original_event.actor else None,
                "correlation_id": self.original_event.correlation_id,
                "source": self.original_event.source,
            },
            "error": {
                "message": self.error_message,
                "type": self.error_type,
                "stack_trace": self.stack_trace,
            },
            "retry_count": self.retry_count,
            "parked_at": self.parked_at.isoformat(),
            "lane_id": self.lane_id,
        }


class ParkedEventsManager:
    """Manages parked events (DLQ)."""

    def __init__(self, event_store: Optional[EventStoreClient] = None):
        """
        Initialize the parked events manager.

        Args:
            event_store: EventStore client (uses global if not provided)
        """
        self.event_store = event_store or get_event_store_client()

    async def park_event(
        self,
        event: EventEnvelope,
        error: Exception,
        retry_count: int,
        lane_id: int,
    ):
        """
        Park an event that has failed permanently.

        Args:
            event: Event that failed
            error: Exception that caused the failure
            retry_count: Number of retry attempts
            lane_id: Lane that was processing the event
        """
        import traceback

        parked_event = ParkedEvent(
            original_event=event,
            error_message=str(error),
            error_type=type(error).__name__,
            retry_count=retry_count,
            parked_at=datetime.now(timezone.utc),
            lane_id=lane_id,
            stack_trace=traceback.format_exc(),
        )

        # Get parked stream name based on aggregate type
        stream_name = get_parked_stream_name(event.aggregate_type)

        # Store in ESDB
        try:
            # Create a simple event for the parked event
            parked_data = parked_event.to_dict()

            # We'll use the event store's append method, but we need to create
            # a proper event envelope for it
            from src.events.envelope import EventEnvelope as Envelope

            import uuid as uuid_module

            parked_envelope = Envelope(
                event_id=str(uuid_module.uuid4()),
                event_type="EventParked",
                aggregate_type=event.aggregate_type,
                aggregate_id=event.aggregate_id,
                version=0,  # Parked events don't have versions
                data=parked_data,
                tags=[
                    f"original_event_id:{event.event_id}",
                    f"original_event_type:{event.event_type}",
                ],
            )

            await self.event_store.append_events(stream_name, [parked_envelope])

            logger.error(
                f"Parked event {event.event_id} (type: {event.event_type}, "
                f"aggregate: {event.aggregate_id}) after {retry_count} retries. "
                f"Error: {error}. Stream: {stream_name}"
            )

        except Exception as e:
            logger.error(f"Failed to park event {event.event_id}: {e}", exc_info=True)

    async def get_parked_events(
        self,
        aggregate_type: str,
        max_count: Optional[int] = None,
    ) -> List[ParkedEvent]:
        """
        Get parked events for an aggregate type.

        Args:
            aggregate_type: Type of aggregate to get parked events for
            max_count: Maximum number of events to retrieve

        Returns:
            List[ParkedEvent]: Parked events
        """
        stream_name = get_parked_stream_name(aggregate_type)

        try:
            events = await self.event_store.read_stream(stream_name, from_version=0, max_count=max_count)

            parked_events = []
            for envelope in events:
                if envelope.event_type == "EventParked":
                    # Reconstruct ParkedEvent from stored data
                    data = envelope.data
                    original_data = data["original_event"]

                    # Reconstruct original event envelope
                    from src.events.envelope import Actor, EventEnvelope as Envelope

                    # Reconstruct actor if present
                    actor = None
                    if original_data.get("actor"):
                        actor = Actor(**original_data["actor"])

                    original_event = Envelope(
                        event_id=original_data["event_id"],
                        event_type=original_data["event_type"],
                        aggregate_type=original_data["aggregate_type"],
                        aggregate_id=original_data["aggregate_id"],
                        version=original_data["version"],
                        data=original_data["data"],
                        actor=actor,
                        correlation_id=original_data.get("correlation_id"),
                        source=original_data.get("source"),
                    )

                    parked_event = ParkedEvent(
                        original_event=original_event,
                        error_message=data["error"]["message"],
                        error_type=data["error"]["type"],
                        retry_count=data["retry_count"],
                        parked_at=datetime.fromisoformat(data["parked_at"]),
                        lane_id=data["lane_id"],
                        stack_trace=data["error"].get("stack_trace"),
                    )

                    parked_events.append(parked_event)

            return parked_events

        except Exception as e:
            logger.error(f"Failed to retrieve parked events for {aggregate_type}: {e}", exc_info=True)
            return []

    async def get_parked_count(self, aggregate_type: str) -> int:
        """
        Get count of parked events for an aggregate type.

        Args:
            aggregate_type: Type of aggregate

        Returns:
            int: Count of parked events
        """
        events = await self.get_parked_events(aggregate_type)
        return len(events)
