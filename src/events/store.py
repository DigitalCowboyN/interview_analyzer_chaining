"""
EventStoreDB client and connection management.

Provides a high-level interface to EventStoreDB with connection pooling,
error handling, and retry logic. Handles both append and read operations
with proper stream management.
"""

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, List, Optional

from esdbclient import (  # type: ignore[import-untyped]
    EventStoreDBClient,
    NewEvent,
    RecordedEvent,
)
from esdbclient.exceptions import (  # type: ignore[import-untyped]
    EventStoreDBClientException,
    NotFound,
    WrongCurrentVersion,
)

from .envelope import EventEnvelope

logger = logging.getLogger(__name__)


class EventStoreError(Exception):
    """Base exception for EventStore operations."""

    pass


class StreamNotFoundError(EventStoreError):
    """Raised when trying to read from a non-existent stream."""

    pass


class ConcurrencyError(EventStoreError):
    """Raised when optimistic concurrency control fails."""

    def __init__(self, message: str, expected_version: int, actual_version: int):
        super().__init__(message)
        self.expected_version = expected_version
        self.actual_version = actual_version


class EventStoreClient:
    """
    High-level EventStoreDB client with connection management and error handling.

    Provides async context manager support and handles connection pooling,
    retries, and stream operations with proper error translation.
    """

    def __init__(
        self,
        connection_string: str = "esdb://localhost:2113?tls=false",
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Initialize the EventStore client.

        Args:
            connection_string: ESDB connection string
            max_retries: Maximum number of retry attempts for failed operations
            retry_delay: Delay between retry attempts in seconds
        """
        self.connection_string = connection_string
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._client: Optional[EventStoreDBClient] = None
        self._connection_lock = asyncio.Lock()

    async def connect(self) -> None:
        """Establish connection to EventStoreDB."""
        async with self._connection_lock:
            if self._client is None:
                try:
                    self._client = EventStoreDBClient(uri=self.connection_string)
                    logger.info(f"Connected to EventStoreDB at {self.connection_string}")
                except Exception as e:
                    logger.error(f"Failed to connect to EventStoreDB: {e}")
                    raise EventStoreError(f"Connection failed: {e}")

    async def disconnect(self) -> None:
        """Close connection to EventStoreDB."""
        async with self._connection_lock:
            if self._client is not None:
                try:
                    self._client.close()
                    self._client = None
                    logger.info("Disconnected from EventStoreDB")
                except Exception as e:
                    logger.warning(f"Error during disconnect: {e}")

    @asynccontextmanager
    async def get_client(self) -> AsyncGenerator[EventStoreDBClient, None]:
        """
        Get a connected client with automatic connection management.

        Yields:
            EventStoreDBClient: Connected client instance

        Raises:
            EventStoreError: If connection fails
        """
        if self._client is None:
            await self.connect()

        if self._client is None:
            raise EventStoreError("Failed to establish connection")

        try:
            yield self._client
        except EventStoreDBClientException as e:
            logger.error(f"EventStoreDB client error: {e}")
            raise EventStoreError(f"Client operation failed: {e}")

    async def append_events(
        self,
        stream_name: str,
        events: List[EventEnvelope],
        expected_version: Optional[int] = None,
    ) -> int:
        """
        Append events to a stream with optimistic concurrency control.

        Args:
            stream_name: Name of the stream to append to
            events: List of events to append
            expected_version: Expected current version of stream (None for new stream)

        Returns:
            int: New version number after append

        Raises:
            ConcurrencyError: If expected version doesn't match actual version
            EventStoreError: For other append failures
        """
        if not events:
            raise ValueError("Cannot append empty event list")

        new_events = []
        for event in events:
            # Convert EventEnvelope to NewEvent
            new_event = NewEvent(
                type=event.event_type,
                data=json.dumps(event.data).encode("utf-8"),
                metadata=json.dumps(
                    {
                        "event_id": event.event_id,
                        "aggregate_type": event.aggregate_type,
                        "aggregate_id": event.aggregate_id,
                        "version": event.version,
                        "occurred_at": event.occurred_at.isoformat(),
                        "schema_version": event.schema_version,
                        "actor": event.actor.dict() if event.actor else None,
                        "correlation_id": event.correlation_id,
                        "causation_id": event.causation_id,
                        "source": event.source,
                        "trace_id": event.trace_id,
                        "project_id": event.project_id,
                        "tags": event.tags,
                    }
                ).encode("utf-8"),
                id=event.event_id,
            )
            new_events.append(new_event)

        for attempt in range(self.max_retries + 1):
            try:
                async with self.get_client() as client:
                    if expected_version is None:
                        # New stream or don't care about version
                        commit_position = await client.append_to_stream(stream_name=stream_name, events=new_events)
                    else:
                        # Optimistic concurrency control
                        commit_position = await client.append_to_stream(
                            stream_name=stream_name, events=new_events, current_version=expected_version
                        )

                    # Calculate new version (expected_version + number of events)
                    new_version = (expected_version or -1) + len(events)

                    logger.debug(
                        f"Appended {len(events)} events to stream '{stream_name}' "
                        f"at commit position {commit_position}, new version: {new_version}"
                    )
                    return new_version

            except WrongCurrentVersion as e:
                # Extract actual version from exception if possible
                actual_version = getattr(e, "actual_version", -1)
                raise ConcurrencyError(
                    f"Concurrency conflict on stream '{stream_name}': "
                    f"expected version {expected_version}, actual version {actual_version}",
                    expected_version or -1,
                    actual_version,
                )
            except EventStoreDBClientException as e:
                if attempt < self.max_retries:
                    logger.warning(
                        f"Append attempt {attempt + 1} failed for stream '{stream_name}': {e}. "
                        f"Retrying in {self.retry_delay}s..."
                    )
                    await asyncio.sleep(self.retry_delay)
                    continue
                else:
                    logger.error(f"Failed to append to stream '{stream_name}' after {self.max_retries} retries: {e}")
                    raise EventStoreError(f"Append failed: {e}")

        raise EventStoreError(f"Append failed after {self.max_retries} retries")

    async def read_stream(
        self,
        stream_name: str,
        from_version: int = 0,
        max_count: Optional[int] = None,
    ) -> List[EventEnvelope]:
        """
        Read events from a stream.

        Args:
            stream_name: Name of the stream to read from
            from_version: Version to start reading from (0 = from beginning)
            max_count: Maximum number of events to read (None = all)

        Returns:
            List[EventEnvelope]: Events from the stream

        Raises:
            StreamNotFoundError: If stream doesn't exist
            EventStoreError: For other read failures
        """
        events = []

        try:
            async with self.get_client() as client:
                recorded_events = client.read_stream(
                    stream_name=stream_name,
                    stream_position=from_version,
                    limit=max_count if max_count is not None else 9223372036854775807
                )

                for recorded_event in recorded_events:
                    event_envelope = self._recorded_event_to_envelope(recorded_event)
                    events.append(event_envelope)

                logger.debug(f"Read {len(events)} events from stream '{stream_name}'")
                return events

        except NotFound:
            raise StreamNotFoundError(f"Stream '{stream_name}' not found")
        except EventStoreDBClientException as e:
            logger.error(f"Failed to read from stream '{stream_name}': {e}")
            raise EventStoreError(f"Read failed: {e}")

    async def get_stream_version(self, stream_name: str) -> Optional[int]:
        """
        Get the current version of a stream.

        Args:
            stream_name: Name of the stream

        Returns:
            int: Current version of the stream, or None if stream doesn't exist
        """
        try:
            events = await self.read_stream(stream_name, max_count=1)
            if not events:
                return None

            # Read all events to get the last version
            all_events = await self.read_stream(stream_name)
            return len(all_events) - 1  # Version is 0-based

        except StreamNotFoundError:
            return None

    def _recorded_event_to_envelope(self, recorded_event: RecordedEvent) -> EventEnvelope:
        """
        Convert a RecordedEvent to an EventEnvelope.

        Args:
            recorded_event: EventStoreDB RecordedEvent

        Returns:
            EventEnvelope: Converted event envelope
        """
        # Parse the event data
        data = json.loads(recorded_event.data.decode("utf-8"))

        # Parse the metadata
        metadata_dict = json.loads(recorded_event.metadata.decode("utf-8"))

        # Reconstruct the EventEnvelope
        return EventEnvelope(
            event_id=metadata_dict["event_id"],
            event_type=recorded_event.type,
            aggregate_type=metadata_dict["aggregate_type"],
            aggregate_id=metadata_dict["aggregate_id"],
            version=metadata_dict["version"],
            occurred_at=metadata_dict["occurred_at"],
            schema_version=metadata_dict["schema_version"],
            data=data,
            actor=metadata_dict.get("actor"),
            correlation_id=metadata_dict.get("correlation_id"),
            causation_id=metadata_dict.get("causation_id"),
            source=metadata_dict.get("source"),
            trace_id=metadata_dict.get("trace_id"),
            project_id=metadata_dict.get("project_id"),
            tags=metadata_dict.get("tags", []),
        )


# Global client instance for application use
_global_client: Optional[EventStoreClient] = None


def get_event_store_client(
    connection_string: Optional[str] = None,
    max_retries: int = 3,
    retry_delay: float = 1.0,
) -> EventStoreClient:
    """
    Get the global EventStore client instance.

    Args:
        connection_string: ESDB connection string (only used on first call)
        max_retries: Maximum retry attempts
        retry_delay: Delay between retries

    Returns:
        EventStoreClient: Global client instance
    """
    global _global_client

    if _global_client is None:
        conn_str = connection_string or "esdb://localhost:2113?tls=false"
        _global_client = EventStoreClient(connection_string=conn_str, max_retries=max_retries, retry_delay=retry_delay)

    return _global_client


async def close_global_client() -> None:
    """Close the global EventStore client connection."""
    global _global_client

    if _global_client is not None:
        await _global_client.disconnect()
        _global_client = None
