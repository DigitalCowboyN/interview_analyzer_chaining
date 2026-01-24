"""
Unit tests for ParkedEventsManager (src/projections/parked_events.py).

Tests the dead letter queue functionality for events that fail after all retries.
"""

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.events.envelope import Actor, ActorType, EventEnvelope
from src.projections.parked_events import ParkedEvent, ParkedEventsManager


def create_test_envelope(
    event_type: str = "TestEvent",
    aggregate_type: str = "Interview",
    aggregate_id: str = None,
    version: int = 0,
    data: dict = None,
) -> EventEnvelope:
    """Helper to create test event envelopes."""
    return EventEnvelope(
        event_type=event_type,
        aggregate_type=aggregate_type,
        aggregate_id=aggregate_id or str(uuid.uuid4()),
        version=version,
        data=data or {"test": "data"},
        actor=Actor(actor_type=ActorType.SYSTEM),
        correlation_id=str(uuid.uuid4()),
    )


class TestParkedEvent:
    """Test ParkedEvent dataclass."""

    def test_parked_event_initialization(self):
        """Test ParkedEvent initialization with all fields."""
        original_event = create_test_envelope()
        parked_at = datetime.now(timezone.utc)

        parked = ParkedEvent(
            original_event=original_event,
            error_message="Test error",
            error_type="ValueError",
            retry_count=3,
            parked_at=parked_at,
            lane_id=5,
            stack_trace="Traceback...",
        )

        assert parked.original_event is original_event
        assert parked.error_message == "Test error"
        assert parked.error_type == "ValueError"
        assert parked.retry_count == 3
        assert parked.parked_at == parked_at
        assert parked.lane_id == 5
        assert parked.stack_trace == "Traceback..."

    def test_parked_event_without_stack_trace(self):
        """Test ParkedEvent initialization without stack trace."""
        original_event = create_test_envelope()
        parked_at = datetime.now(timezone.utc)

        parked = ParkedEvent(
            original_event=original_event,
            error_message="Test error",
            error_type="ValueError",
            retry_count=3,
            parked_at=parked_at,
            lane_id=5,
        )

        assert parked.stack_trace is None

    def test_parked_event_to_dict(self):
        """Test converting ParkedEvent to dictionary."""
        aggregate_id = str(uuid.uuid4())
        original_event = create_test_envelope(
            event_type="InterviewCreated",
            aggregate_type="Interview",
            aggregate_id=aggregate_id,
            data={"title": "Test Interview"},
        )
        parked_at = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)

        parked = ParkedEvent(
            original_event=original_event,
            error_message="Connection failed",
            error_type="ConnectionError",
            retry_count=5,
            parked_at=parked_at,
            lane_id=2,
            stack_trace="File...\nLine...",
        )

        result = parked.to_dict()

        # Check original event fields
        assert result["original_event"]["event_id"] == original_event.event_id
        assert result["original_event"]["event_type"] == "InterviewCreated"
        assert result["original_event"]["aggregate_type"] == "Interview"
        assert result["original_event"]["aggregate_id"] == aggregate_id
        assert result["original_event"]["version"] == 0
        assert result["original_event"]["data"] == {"title": "Test Interview"}

        # Check actor is serialized (from SYSTEM actor)
        assert result["original_event"]["actor"] is not None
        assert result["original_event"]["actor"]["actor_type"] == "system"

        # Check correlation_id and source are included
        assert "correlation_id" in result["original_event"]
        assert "source" in result["original_event"]

        # Check error fields
        assert result["error"]["message"] == "Connection failed"
        assert result["error"]["type"] == "ConnectionError"
        assert result["error"]["stack_trace"] == "File...\nLine..."

        # Check metadata fields
        assert result["retry_count"] == 5
        assert result["parked_at"] == "2024-01-15T10:30:00+00:00"
        assert result["lane_id"] == 2

    def test_parked_event_to_dict_without_stack_trace(self):
        """Test to_dict with None stack trace."""
        original_event = create_test_envelope()
        parked_at = datetime.now(timezone.utc)

        parked = ParkedEvent(
            original_event=original_event,
            error_message="Error",
            error_type="Exception",
            retry_count=1,
            parked_at=parked_at,
            lane_id=0,
        )

        result = parked.to_dict()

        assert result["error"]["stack_trace"] is None


class TestParkedEventsManagerInit:
    """Test ParkedEventsManager initialization."""

    def test_init_with_provided_event_store(self):
        """Test initialization with provided event store."""
        mock_store = MagicMock()
        manager = ParkedEventsManager(event_store=mock_store)

        assert manager.event_store is mock_store

    def test_init_uses_global_event_store(self):
        """Test initialization uses global event store when none provided."""
        mock_store = MagicMock()

        with patch("src.projections.parked_events.get_event_store_client", return_value=mock_store):
            manager = ParkedEventsManager()

            assert manager.event_store is mock_store


@pytest.mark.asyncio
class TestParkedEventsManagerParkEvent:
    """Test ParkedEventsManager.park_event method."""

    async def test_park_event_success(self):
        """Test successfully parking an event."""
        mock_store = AsyncMock()
        mock_store.append_events = AsyncMock()
        manager = ParkedEventsManager(event_store=mock_store)

        original_event = create_test_envelope(
            event_type="SentenceCreated",
            aggregate_type="Sentence",
        )
        error = ValueError("Neo4j connection failed")

        await manager.park_event(
            event=original_event,
            error=error,
            retry_count=3,
            lane_id=5,
        )

        # Verify append_events was called
        mock_store.append_events.assert_called_once()

        # Check the stream name
        call_args = mock_store.append_events.call_args
        stream_name = call_args.args[0]
        assert "parked" in stream_name.lower() or "Sentence" in stream_name

        # Check the parked envelope
        parked_envelopes = call_args.args[1]
        assert len(parked_envelopes) == 1
        parked_envelope = parked_envelopes[0]
        assert parked_envelope.event_type == "EventParked"
        assert parked_envelope.aggregate_type == "Sentence"
        assert parked_envelope.data["error"]["message"] == "Neo4j connection failed"
        assert parked_envelope.data["error"]["type"] == "ValueError"
        assert parked_envelope.data["retry_count"] == 3
        assert parked_envelope.data["lane_id"] == 5

    async def test_park_event_includes_stack_trace(self):
        """Test that parking includes stack trace."""
        mock_store = AsyncMock()
        mock_store.append_events = AsyncMock()
        manager = ParkedEventsManager(event_store=mock_store)

        original_event = create_test_envelope()
        error = Exception("Test error")

        await manager.park_event(
            event=original_event,
            error=error,
            retry_count=1,
            lane_id=0,
        )

        parked_envelope = mock_store.append_events.call_args.args[1][0]
        assert parked_envelope.data["error"]["stack_trace"] is not None

    async def test_park_event_handles_append_failure(self):
        """Test handling of append failure when parking."""
        mock_store = AsyncMock()
        mock_store.append_events = AsyncMock(side_effect=Exception("Store unavailable"))
        manager = ParkedEventsManager(event_store=mock_store)

        original_event = create_test_envelope()
        error = ValueError("Original error")

        # Should not raise, just log
        await manager.park_event(
            event=original_event,
            error=error,
            retry_count=1,
            lane_id=0,
        )

    async def test_park_event_for_interview(self):
        """Test parking an interview event."""
        mock_store = AsyncMock()
        mock_store.append_events = AsyncMock()
        manager = ParkedEventsManager(event_store=mock_store)

        original_event = create_test_envelope(
            event_type="InterviewCreated",
            aggregate_type="Interview",
        )

        await manager.park_event(
            event=original_event,
            error=RuntimeError("Processing failed"),
            retry_count=5,
            lane_id=3,
        )

        parked_envelope = mock_store.append_events.call_args.args[1][0]
        assert parked_envelope.aggregate_type == "Interview"

    async def test_park_event_metadata(self):
        """Test that parked event includes correct metadata in tags."""
        mock_store = AsyncMock()
        mock_store.append_events = AsyncMock()
        manager = ParkedEventsManager(event_store=mock_store)

        original_event = create_test_envelope()

        await manager.park_event(
            event=original_event,
            error=Exception("Error"),
            retry_count=2,
            lane_id=1,
        )

        parked_envelope = mock_store.append_events.call_args.args[1][0]
        # Metadata is now stored in tags
        tags = parked_envelope.tags
        assert f"original_event_id:{original_event.event_id}" in tags
        assert f"original_event_type:{original_event.event_type}" in tags


@pytest.mark.asyncio
class TestParkedEventsManagerGetParkedEvents:
    """Test ParkedEventsManager.get_parked_events method."""

    async def test_get_parked_events_success(self):
        """Test successfully retrieving parked events."""
        mock_store = AsyncMock()
        manager = ParkedEventsManager(event_store=mock_store)

        # Create mock parked event data (matching the new serialization format)
        original_event_id = str(uuid.uuid4())
        aggregate_id = str(uuid.uuid4())
        parked_at = datetime.now(timezone.utc).isoformat()

        parked_data = {
            "original_event": {
                "event_id": original_event_id,
                "event_type": "SentenceCreated",
                "aggregate_type": "Sentence",
                "aggregate_id": aggregate_id,
                "version": 0,
                "data": {"text": "Test sentence."},
                "actor": {"actor_type": "system", "user_id": None, "display": None},
                "correlation_id": None,
                "source": "interview_analyzer",
            },
            "error": {
                "message": "Processing failed",
                "type": "RuntimeError",
                "stack_trace": "Traceback...",
            },
            "retry_count": 3,
            "parked_at": parked_at,
            "lane_id": 2,
        }

        mock_envelope = MagicMock()
        mock_envelope.event_type = "EventParked"
        mock_envelope.data = parked_data

        mock_store.read_stream = AsyncMock(return_value=[mock_envelope])

        result = await manager.get_parked_events("Sentence")

        assert len(result) == 1
        parked = result[0]
        assert parked.original_event.event_id == original_event_id
        assert parked.original_event.event_type == "SentenceCreated"
        assert parked.error_message == "Processing failed"
        assert parked.error_type == "RuntimeError"
        assert parked.retry_count == 3
        assert parked.lane_id == 2
        assert parked.stack_trace == "Traceback..."

    async def test_get_parked_events_empty_stream(self):
        """Test getting parked events from empty stream."""
        mock_store = AsyncMock()
        mock_store.read_stream = AsyncMock(return_value=[])
        manager = ParkedEventsManager(event_store=mock_store)

        result = await manager.get_parked_events("Interview")

        assert result == []

    async def test_get_parked_events_with_max_count(self):
        """Test getting parked events with max count limit."""
        mock_store = AsyncMock()
        mock_store.read_stream = AsyncMock(return_value=[])
        manager = ParkedEventsManager(event_store=mock_store)

        await manager.get_parked_events("Sentence", max_count=10)

        mock_store.read_stream.assert_called_once()
        call_kwargs = mock_store.read_stream.call_args.kwargs
        assert call_kwargs.get("max_count") == 10

    async def test_get_parked_events_filters_non_parked_events(self):
        """Test that non-EventParked events are filtered out."""
        mock_store = AsyncMock()
        manager = ParkedEventsManager(event_store=mock_store)

        # Mix of EventParked and other events
        parked_envelope = MagicMock()
        parked_envelope.event_type = "EventParked"
        parked_envelope.data = {
            "original_event": {
                "event_id": str(uuid.uuid4()),
                "event_type": "SentenceCreated",
                "aggregate_type": "Sentence",
                "aggregate_id": str(uuid.uuid4()),
                "version": 0,
                "data": {},
                "actor": None,
                "correlation_id": None,
                "source": None,
            },
            "error": {"message": "Error", "type": "Exception"},
            "retry_count": 1,
            "parked_at": datetime.now(timezone.utc).isoformat(),
            "lane_id": 0,
        }

        other_envelope = MagicMock()
        other_envelope.event_type = "SomeOtherEvent"

        mock_store.read_stream = AsyncMock(return_value=[parked_envelope, other_envelope])

        result = await manager.get_parked_events("Sentence")

        # Should only return the EventParked event
        assert len(result) == 1

    async def test_get_parked_events_handles_read_failure(self):
        """Test handling of read failure."""
        mock_store = AsyncMock()
        mock_store.read_stream = AsyncMock(side_effect=Exception("Read failed"))
        manager = ParkedEventsManager(event_store=mock_store)

        result = await manager.get_parked_events("Interview")

        # Should return empty list on error
        assert result == []

    async def test_get_parked_events_reconstructs_original_event(self):
        """Test that original event is correctly reconstructed."""
        mock_store = AsyncMock()
        manager = ParkedEventsManager(event_store=mock_store)

        # Use valid UUIDs for event_id and aggregate_id
        event_id = str(uuid.uuid4())
        aggregate_id = str(uuid.uuid4())

        original_data = {
            "event_id": event_id,
            "event_type": "InterviewCreated",
            "aggregate_type": "Interview",
            "aggregate_id": aggregate_id,
            "version": 5,
            "data": {"title": "Test", "source": "file.txt"},
            "actor": {"actor_type": "human", "user_id": "user-1", "display": "Test User"},
            "correlation_id": "corr-123",
            "source": "test_source",
        }

        parked_data = {
            "original_event": original_data,
            "error": {"message": "Error", "type": "Exception"},
            "retry_count": 2,
            "parked_at": datetime.now(timezone.utc).isoformat(),
            "lane_id": 1,
        }

        mock_envelope = MagicMock()
        mock_envelope.event_type = "EventParked"
        mock_envelope.data = parked_data

        mock_store.read_stream = AsyncMock(return_value=[mock_envelope])

        result = await manager.get_parked_events("Interview")

        assert len(result) == 1
        original = result[0].original_event
        assert original.event_id == event_id
        assert original.event_type == "InterviewCreated"
        assert original.aggregate_type == "Interview"
        assert original.aggregate_id == aggregate_id
        assert original.version == 5
        assert original.data == {"title": "Test", "source": "file.txt"}
        # Verify reconstructed metadata fields
        assert original.actor is not None
        assert original.actor.user_id == "user-1"
        assert original.correlation_id == "corr-123"
        assert original.source == "test_source"


@pytest.mark.asyncio
class TestParkedEventsManagerGetParkedCount:
    """Test ParkedEventsManager.get_parked_count method."""

    async def test_get_parked_count_returns_count(self):
        """Test getting count of parked events."""
        mock_store = AsyncMock()
        manager = ParkedEventsManager(event_store=mock_store)

        # Mock get_parked_events to return 3 events
        manager.get_parked_events = AsyncMock(return_value=[
            MagicMock(), MagicMock(), MagicMock()
        ])

        count = await manager.get_parked_count("Sentence")

        assert count == 3
        manager.get_parked_events.assert_called_once_with("Sentence")

    async def test_get_parked_count_zero(self):
        """Test getting count when no parked events."""
        mock_store = AsyncMock()
        manager = ParkedEventsManager(event_store=mock_store)

        manager.get_parked_events = AsyncMock(return_value=[])

        count = await manager.get_parked_count("Interview")

        assert count == 0

    async def test_get_parked_count_for_different_aggregates(self):
        """Test count for different aggregate types."""
        mock_store = AsyncMock()
        manager = ParkedEventsManager(event_store=mock_store)

        # Different counts for different aggregates
        async def mock_get_parked(aggregate_type):
            if aggregate_type == "Interview":
                return [MagicMock()]
            elif aggregate_type == "Sentence":
                return [MagicMock(), MagicMock()]
            return []

        manager.get_parked_events = mock_get_parked

        interview_count = await manager.get_parked_count("Interview")
        sentence_count = await manager.get_parked_count("Sentence")

        assert interview_count == 1
        assert sentence_count == 2
