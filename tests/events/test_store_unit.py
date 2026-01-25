"""
Unit tests for EventStoreDB client (src/events/store.py).

Tests connection management, event append/read operations, retry logic,
and error handling without requiring actual EventStoreDB connection.
"""

import json
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.events.envelope import Actor, ActorType, EventEnvelope
from src.events.store import (
    ConcurrencyError,
    EventStoreClient,
    EventStoreError,
    StreamNotFoundError,
    close_global_client,
    get_event_store_client,
)


def create_test_envelope(
    event_type: str = "TestEvent",
    aggregate_type: str = "Interview",  # Must be valid AggregateType enum value
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


class TestEventStoreClientInit:
    """Test EventStoreClient initialization."""

    def test_init_with_defaults(self):
        """Test client initialization with default values."""
        client = EventStoreClient()

        assert client.connection_string == "esdb://localhost:2113?tls=false"
        assert client.max_retries == 3
        assert client.retry_delay == 1.0
        assert client._client is None

    def test_init_with_custom_values(self):
        """Test client initialization with custom values."""
        client = EventStoreClient(
            connection_string="esdb://custom:2114?tls=true",
            max_retries=5,
            retry_delay=2.0,
        )

        assert client.connection_string == "esdb://custom:2114?tls=true"
        assert client.max_retries == 5
        assert client.retry_delay == 2.0


@pytest.mark.asyncio
class TestEventStoreClientConnect:
    """Test connection management."""

    async def test_connect_success(self):
        """Test successful connection to EventStoreDB."""
        client = EventStoreClient()

        with patch("src.events.store.EventStoreDBClient") as mock_esdb:
            mock_esdb.return_value = MagicMock()
            await client.connect()

            mock_esdb.assert_called_once_with(uri=client.connection_string)
            assert client._client is not None

    async def test_connect_failure_raises_event_store_error(self):
        """Test that connection failure raises EventStoreError."""
        client = EventStoreClient()

        with patch("src.events.store.EventStoreDBClient") as mock_esdb:
            mock_esdb.side_effect = Exception("Connection refused")

            with pytest.raises(EventStoreError, match="Connection failed"):
                await client.connect()

    async def test_connect_is_idempotent(self):
        """Test that multiple connect calls don't create multiple connections."""
        client = EventStoreClient()

        with patch("src.events.store.EventStoreDBClient") as mock_esdb:
            mock_instance = MagicMock()
            mock_esdb.return_value = mock_instance

            await client.connect()
            await client.connect()

            # Should only create one connection
            mock_esdb.assert_called_once()

    async def test_disconnect_closes_connection(self):
        """Test that disconnect closes the connection."""
        client = EventStoreClient()

        with patch("src.events.store.EventStoreDBClient") as mock_esdb:
            mock_instance = MagicMock()
            mock_esdb.return_value = mock_instance

            await client.connect()
            await client.disconnect()

            mock_instance.close.assert_called_once()
            assert client._client is None

    async def test_disconnect_handles_error_gracefully(self):
        """Test that disconnect handles errors without raising."""
        client = EventStoreClient()

        with patch("src.events.store.EventStoreDBClient") as mock_esdb:
            mock_instance = MagicMock()
            mock_instance.close.side_effect = Exception("Close failed")
            mock_esdb.return_value = mock_instance

            await client.connect()
            # Should not raise
            await client.disconnect()

    async def test_disconnect_when_not_connected(self):
        """Test that disconnect when not connected is safe."""
        client = EventStoreClient()
        # Should not raise
        await client.disconnect()


@pytest.mark.asyncio
class TestEventStoreClientGetClient:
    """Test get_client context manager."""

    async def test_get_client_auto_connects(self):
        """Test that get_client automatically connects if not connected."""
        client = EventStoreClient()

        with patch("src.events.store.EventStoreDBClient") as mock_esdb:
            mock_instance = MagicMock()
            mock_esdb.return_value = mock_instance

            async with client.get_client() as esdb_client:
                assert esdb_client is mock_instance

    async def test_get_client_raises_if_connection_fails(self):
        """Test that get_client raises if connection cannot be established."""
        client = EventStoreClient()

        with patch("src.events.store.EventStoreDBClient") as mock_esdb:
            mock_esdb.side_effect = Exception("Connection refused")

            with pytest.raises(EventStoreError, match="Connection failed"):
                async with client.get_client():
                    pass

    async def test_get_client_wraps_client_exceptions(self):
        """Test that client exceptions are wrapped in EventStoreError."""
        from esdbclient.exceptions import EventStoreDBClientException

        client = EventStoreClient()

        with patch("src.events.store.EventStoreDBClient") as mock_esdb:
            mock_instance = MagicMock()
            mock_esdb.return_value = mock_instance

            await client.connect()

            # Simulate exception during context
            async with client.get_client() as esdb_client:
                # The exception needs to be raised in the context
                pass

            # Test exception wrapping by mocking an operation
            mock_instance.read_stream.side_effect = EventStoreDBClientException("Test error")

            with pytest.raises(EventStoreError, match="Client operation failed"):
                async with client.get_client() as esdb_client:
                    esdb_client.read_stream("test")


@pytest.mark.asyncio
class TestEventStoreClientAppendEvents:
    """Test event append operations."""

    async def test_append_events_success(self):
        """Test successful event append."""
        client = EventStoreClient()
        envelope = create_test_envelope()

        with patch("src.events.store.EventStoreDBClient") as mock_esdb:
            mock_instance = MagicMock()
            mock_instance.append_to_stream.return_value = 12345  # commit position
            mock_esdb.return_value = mock_instance

            await client.connect()
            new_version = await client.append_events(
                stream_name="Test-123",
                events=[envelope],
                expected_version=-1,  # New stream
            )

            mock_instance.append_to_stream.assert_called_once()
            assert new_version == 0  # -1 + 1 event = 0

    async def test_append_events_empty_list_raises(self):
        """Test that appending empty list raises ValueError."""
        client = EventStoreClient()

        with pytest.raises(ValueError, match="Cannot append empty event list"):
            await client.append_events("Test-123", events=[], expected_version=-1)

    async def test_append_events_concurrency_error(self):
        """Test that WrongCurrentVersion raises ConcurrencyError."""
        from contextlib import asynccontextmanager

        from esdbclient.exceptions import WrongCurrentVersion

        client = EventStoreClient()
        envelope = create_test_envelope()

        # Create a mock that raises WrongCurrentVersion
        mock_esdb_client = MagicMock()
        mock_esdb_client.append_to_stream.side_effect = WrongCurrentVersion("Version mismatch")

        # Patch get_client to yield mock without wrapping exceptions
        @asynccontextmanager
        async def mock_get_client():
            yield mock_esdb_client

        with patch.object(client, "get_client", mock_get_client):
            with pytest.raises(ConcurrencyError) as exc_info:
                await client.append_events(
                    stream_name="Test-123",
                    events=[envelope],
                    expected_version=5,
                )

            assert exc_info.value.expected_version == 5

    async def test_append_events_retries_on_transient_error(self):
        """Test that append retries on transient errors."""
        from contextlib import asynccontextmanager

        from esdbclient.exceptions import EventStoreDBClientException

        client = EventStoreClient(max_retries=2, retry_delay=0.01)
        envelope = create_test_envelope()

        # Mock the client that fails twice then succeeds
        mock_esdb_client = MagicMock()
        mock_esdb_client.append_to_stream.side_effect = [
            EventStoreDBClientException("Transient error"),
            EventStoreDBClientException("Transient error"),
            12345,  # Success on third attempt
        ]

        # Patch get_client to yield mock without wrapping exceptions
        @asynccontextmanager
        async def mock_get_client():
            yield mock_esdb_client

        with patch.object(client, "get_client", mock_get_client):
            new_version = await client.append_events(
                stream_name="Test-123",
                events=[envelope],
                expected_version=-1,
            )

            assert mock_esdb_client.append_to_stream.call_count == 3
            assert new_version == 0

    async def test_append_events_exhausts_retries(self):
        """Test that append raises after exhausting retries."""
        from contextlib import asynccontextmanager

        from esdbclient.exceptions import EventStoreDBClientException

        client = EventStoreClient(max_retries=2, retry_delay=0.01)
        envelope = create_test_envelope()

        # Mock the client that always fails
        mock_esdb_client = MagicMock()
        mock_esdb_client.append_to_stream.side_effect = EventStoreDBClientException("Persistent error")

        # Patch get_client to yield mock without wrapping exceptions
        @asynccontextmanager
        async def mock_get_client():
            yield mock_esdb_client

        with patch.object(client, "get_client", mock_get_client):
            with pytest.raises(EventStoreError, match="Append failed"):
                await client.append_events(
                    stream_name="Test-123",
                    events=[envelope],
                    expected_version=-1,
                )

            # Initial attempt + 2 retries = 3 calls
            assert mock_esdb_client.append_to_stream.call_count == 3

    async def test_append_events_with_existing_stream_version(self):
        """Test append with existing stream version."""
        client = EventStoreClient()
        envelope = create_test_envelope(version=5)

        with patch("src.events.store.EventStoreDBClient") as mock_esdb:
            mock_instance = MagicMock()
            mock_instance.append_to_stream.return_value = 12345
            mock_esdb.return_value = mock_instance

            await client.connect()
            new_version = await client.append_events(
                stream_name="Test-123",
                events=[envelope],
                expected_version=4,  # Existing stream at version 4
            )

            assert new_version == 5  # 4 + 1 event = 5

    async def test_append_events_multiple_events(self):
        """Test appending multiple events at once."""
        client = EventStoreClient()
        envelopes = [create_test_envelope(version=i) for i in range(3)]

        with patch("src.events.store.EventStoreDBClient") as mock_esdb:
            mock_instance = MagicMock()
            mock_instance.append_to_stream.return_value = 12345
            mock_esdb.return_value = mock_instance

            await client.connect()
            new_version = await client.append_events(
                stream_name="Test-123",
                events=envelopes,
                expected_version=-1,
            )

            # -1 + 3 events = 2 (0-indexed)
            assert new_version == 2


@pytest.mark.asyncio
class TestEventStoreClientReadStream:
    """Test event read operations."""

    async def test_read_stream_success(self):
        """Test successful stream read."""
        client = EventStoreClient()
        aggregate_id = str(uuid.uuid4())

        # Create mock recorded event
        mock_recorded = MagicMock()
        mock_recorded.type = "TestEvent"
        mock_recorded.data = json.dumps({"test": "data"}).encode("utf-8")
        mock_recorded.metadata = json.dumps({
            "event_id": str(uuid.uuid4()),
            "aggregate_type": "Interview",
            "aggregate_id": aggregate_id,
            "version": 0,
            "occurred_at": datetime.now(timezone.utc).isoformat(),
            "schema_version": "1.0.0",
            "actor": {"actor_type": "system"},
            "correlation_id": str(uuid.uuid4()),
            "causation_id": None,
            "source": "test",
            "trace_id": None,
            "project_id": "default",
            "tags": [],
        }).encode("utf-8")

        with patch("src.events.store.EventStoreDBClient") as mock_esdb:
            mock_instance = MagicMock()
            mock_instance.read_stream.return_value = [mock_recorded]
            mock_esdb.return_value = mock_instance

            await client.connect()
            events = await client.read_stream("Test-123")

            assert len(events) == 1
            assert events[0].event_type == "TestEvent"
            assert events[0].aggregate_id == aggregate_id

    async def test_read_stream_not_found(self):
        """Test reading from non-existent stream raises StreamNotFoundError."""
        from contextlib import asynccontextmanager

        from esdbclient.exceptions import NotFound

        client = EventStoreClient()

        # Mock the client that raises NotFound
        mock_esdb_client = MagicMock()
        mock_esdb_client.read_stream.side_effect = NotFound("Stream not found")

        # Patch get_client to yield mock without wrapping exceptions
        @asynccontextmanager
        async def mock_get_client():
            yield mock_esdb_client

        with patch.object(client, "get_client", mock_get_client):
            with pytest.raises(StreamNotFoundError, match="not found"):
                await client.read_stream("NonExistent-123")

    async def test_read_stream_with_from_version(self):
        """Test reading from specific version."""
        client = EventStoreClient()

        with patch("src.events.store.EventStoreDBClient") as mock_esdb:
            mock_instance = MagicMock()
            mock_instance.read_stream.return_value = []
            mock_esdb.return_value = mock_instance

            await client.connect()
            await client.read_stream("Test-123", from_version=5)

            mock_instance.read_stream.assert_called_once()
            call_kwargs = mock_instance.read_stream.call_args
            assert call_kwargs.kwargs["stream_position"] == 5

    async def test_read_stream_with_max_count(self):
        """Test reading with max count limit."""
        client = EventStoreClient()

        with patch("src.events.store.EventStoreDBClient") as mock_esdb:
            mock_instance = MagicMock()
            mock_instance.read_stream.return_value = []
            mock_esdb.return_value = mock_instance

            await client.connect()
            await client.read_stream("Test-123", max_count=10)

            mock_instance.read_stream.assert_called_once()
            call_kwargs = mock_instance.read_stream.call_args
            assert call_kwargs.kwargs["limit"] == 10

    async def test_read_stream_client_exception(self):
        """Test that client exceptions are wrapped."""
        from esdbclient.exceptions import EventStoreDBClientException

        client = EventStoreClient()

        with patch("src.events.store.EventStoreDBClient") as mock_esdb:
            mock_instance = MagicMock()
            mock_instance.read_stream.side_effect = EventStoreDBClientException("Read failed")
            mock_esdb.return_value = mock_instance

            await client.connect()

            with pytest.raises(EventStoreError, match="Read failed"):
                await client.read_stream("Test-123")


@pytest.mark.asyncio
class TestEventStoreClientGetStreamVersion:
    """Test get_stream_version method."""

    async def test_get_stream_version_found(self):
        """Test getting version of existing stream."""
        client = EventStoreClient()
        aggregate_id = str(uuid.uuid4())

        # Create mock events
        def create_mock_recorded(version):
            mock = MagicMock()
            mock.type = "TestEvent"
            mock.data = json.dumps({"test": "data"}).encode("utf-8")
            mock.metadata = json.dumps({
                "event_id": str(uuid.uuid4()),
                "aggregate_type": "Interview",
                "aggregate_id": aggregate_id,
                "version": version,
                "occurred_at": datetime.now(timezone.utc).isoformat(),
                "schema_version": "1.0.0",
                "actor": None,
                "correlation_id": None,
                "causation_id": None,
                "source": "test",
                "trace_id": None,
                "project_id": "default",
                "tags": [],
            }).encode("utf-8")
            return mock

        with patch("src.events.store.EventStoreDBClient") as mock_esdb:
            mock_instance = MagicMock()
            # First call (max_count=1) returns one event
            # Second call (all events) returns 3 events
            mock_instance.read_stream.side_effect = [
                [create_mock_recorded(0)],  # First read
                [create_mock_recorded(0), create_mock_recorded(1), create_mock_recorded(2)],  # Second read
            ]
            mock_esdb.return_value = mock_instance

            await client.connect()
            version = await client.get_stream_version("Test-123")

            assert version == 2  # 3 events, 0-indexed = version 2

    async def test_get_stream_version_not_found(self):
        """Test getting version of non-existent stream returns None."""
        client = EventStoreClient()

        with patch.object(client, "read_stream", new_callable=AsyncMock) as mock_read:
            mock_read.side_effect = StreamNotFoundError("Not found")

            version = await client.get_stream_version("NonExistent-123")

            assert version is None


@pytest.mark.asyncio
class TestRecordedEventToEnvelope:
    """Test _recorded_event_to_envelope conversion."""

    def test_converts_recorded_event_correctly(self):
        """Test that recorded events are correctly converted to envelopes."""
        client = EventStoreClient()
        aggregate_id = str(uuid.uuid4())
        event_id = str(uuid.uuid4())
        correlation_id = str(uuid.uuid4())

        mock_recorded = MagicMock()
        mock_recorded.type = "TestEvent"
        mock_recorded.data = json.dumps({"key": "value"}).encode("utf-8")
        mock_recorded.metadata = json.dumps({
            "event_id": event_id,
            "aggregate_type": "Interview",
            "aggregate_id": aggregate_id,
            "version": 5,
            "occurred_at": "2024-01-15T10:30:00+00:00",
            "schema_version": "1.0.0",
            "actor": {"actor_type": "human", "user_id": "user-123"},
            "correlation_id": correlation_id,
            "causation_id": None,
            "source": "test_source",
            "trace_id": "trace-123",
            "project_id": "project-456",
            "tags": ["tag1", "tag2"],
        }).encode("utf-8")

        envelope = client._recorded_event_to_envelope(mock_recorded)

        assert envelope.event_id == event_id
        assert envelope.event_type == "TestEvent"
        assert envelope.aggregate_type == "Interview"
        assert envelope.aggregate_id == aggregate_id
        assert envelope.version == 5
        assert envelope.data == {"key": "value"}
        assert envelope.correlation_id == correlation_id
        assert envelope.source == "test_source"
        assert envelope.tags == ["tag1", "tag2"]


@pytest.mark.asyncio
class TestGlobalClient:
    """Test global client functions."""

    async def test_get_event_store_client_creates_singleton(self):
        """Test that get_event_store_client returns singleton."""
        # Reset global state
        import src.events.store as store_module
        store_module._global_client = None

        with patch.dict("os.environ", {"ESDB_CONNECTION_STRING": "esdb://test:2113?tls=false"}):
            client1 = get_event_store_client()
            client2 = get_event_store_client()

            assert client1 is client2

        # Cleanup
        store_module._global_client = None

    async def test_get_event_store_client_uses_env_var(self):
        """Test that get_event_store_client uses environment variable."""
        import src.events.store as store_module
        store_module._global_client = None

        with patch.dict("os.environ", {"ESDB_CONNECTION_STRING": "esdb://custom:9999?tls=true"}):
            client = get_event_store_client()
            assert client.connection_string == "esdb://custom:9999?tls=true"

        store_module._global_client = None

    async def test_get_event_store_client_uses_provided_string(self):
        """Test that provided connection string is used on first call."""
        import src.events.store as store_module
        store_module._global_client = None

        client = get_event_store_client(connection_string="esdb://provided:1234?tls=false")
        assert client.connection_string == "esdb://provided:1234?tls=false"

        store_module._global_client = None

    async def test_close_global_client(self):
        """Test that close_global_client disconnects and clears singleton."""
        import src.events.store as store_module
        store_module._global_client = None

        with patch("src.events.store.EventStoreDBClient") as mock_esdb:
            mock_instance = MagicMock()
            mock_esdb.return_value = mock_instance

            client = get_event_store_client()
            await client.connect()

            await close_global_client()

            assert store_module._global_client is None

    async def test_close_global_client_when_none(self):
        """Test that close_global_client is safe when no client exists."""
        import src.events.store as store_module
        store_module._global_client = None

        # Should not raise
        await close_global_client()

    async def test_get_event_store_client_host_environment_detection(self):
        """Test that get_event_store_client uses localhost for host environment."""
        import src.events.store as store_module
        store_module._global_client = None

        # Clear env var and mock host environment
        with patch.dict("os.environ", {}, clear=True):
            with patch("src.utils.environment.detect_environment", return_value="host"):
                client = get_event_store_client()
                assert client.connection_string == "esdb://localhost:2113?tls=false"

        store_module._global_client = None

    async def test_get_event_store_client_docker_environment_detection(self):
        """Test that get_event_store_client uses eventstore for docker environment."""
        import src.events.store as store_module
        store_module._global_client = None

        # Clear env var and mock docker environment
        with patch.dict("os.environ", {}, clear=True):
            with patch("src.utils.environment.detect_environment", return_value="docker"):
                client = get_event_store_client()
                assert client.connection_string == "esdb://eventstore:2113?tls=false"

        store_module._global_client = None

    async def test_get_event_store_client_ci_environment_detection(self):
        """Test that get_event_store_client uses eventstore for CI environment."""
        import src.events.store as store_module
        store_module._global_client = None

        # Clear env var and mock CI environment
        with patch.dict("os.environ", {}, clear=True):
            with patch("src.utils.environment.detect_environment", return_value="ci"):
                client = get_event_store_client()
                assert client.connection_string == "esdb://eventstore:2113?tls=false"

        store_module._global_client = None

    async def test_get_event_store_client_env_var_overrides_detection(self):
        """Test that environment variable takes precedence over environment detection."""
        import src.events.store as store_module
        store_module._global_client = None

        # Env var should override even when detection would return docker
        with patch.dict("os.environ", {"ESDB_CONNECTION_STRING": "esdb://override:9999?tls=true"}):
            with patch("src.utils.environment.detect_environment", return_value="docker"):
                client = get_event_store_client()
                assert client.connection_string == "esdb://override:9999?tls=true"

        store_module._global_client = None

    async def test_get_event_store_client_explicit_overrides_all(self):
        """Test that explicit connection_string parameter takes highest precedence."""
        import src.events.store as store_module
        store_module._global_client = None

        # Explicit should override env var and detection
        with patch.dict("os.environ", {"ESDB_CONNECTION_STRING": "esdb://envvar:8888?tls=true"}):
            with patch("src.utils.environment.detect_environment", return_value="docker"):
                client = get_event_store_client(connection_string="esdb://explicit:7777?tls=false")
                assert client.connection_string == "esdb://explicit:7777?tls=false"

        store_module._global_client = None


class TestExceptionClasses:
    """Test custom exception classes."""

    def test_event_store_error(self):
        """Test EventStoreError exception."""
        error = EventStoreError("Test error")
        assert str(error) == "Test error"

    def test_stream_not_found_error(self):
        """Test StreamNotFoundError exception."""
        error = StreamNotFoundError("Stream not found")
        assert str(error) == "Stream not found"
        assert isinstance(error, EventStoreError)

    def test_concurrency_error(self):
        """Test ConcurrencyError exception."""
        error = ConcurrencyError("Version mismatch", expected_version=5, actual_version=7)
        assert "Version mismatch" in str(error)
        assert error.expected_version == 5
        assert error.actual_version == 7
        assert isinstance(error, EventStoreError)
