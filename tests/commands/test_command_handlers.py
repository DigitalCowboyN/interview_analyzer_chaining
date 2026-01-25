"""
Tests for command handlers.

Validates that commands correctly create events and update aggregates.
"""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.commands.handlers import InterviewCommandHandler, SentenceCommandHandler
from src.commands.interview_commands import CreateInterviewCommand
from src.commands.sentence_commands import CreateSentenceCommand, EditSentenceCommand
from src.events.envelope import Actor, ActorType
from src.events.store import EventStoreClient


@pytest.mark.asyncio
class TestInterviewCommandHandler:
    """Test interview command handler."""

    async def test_create_interview_command(self):
        """Test creating an interview via command."""
        # Create mock event store
        mock_event_store = MagicMock(spec=EventStoreClient)
        mock_event_store.read_stream = AsyncMock(return_value=[])  # No existing events
        mock_event_store.append_events = AsyncMock(return_value=0)  # Return version 0

        # Create handler with mocked event store
        handler = InterviewCommandHandler(event_store=mock_event_store)

        # Create command
        interview_id = str(uuid.uuid4())
        project_id = str(uuid.uuid4())
        command = CreateInterviewCommand(
            interview_id=interview_id,
            project_id=project_id,
            title="Test Interview",
            source="test_file.txt",
            language="en",
            actor=Actor(user_id="test-user", actor_type=ActorType.SYSTEM, display="Test System"),
        )

        # Execute command
        result = await handler.handle(command)

        # Verify result
        assert result.success is True
        assert result.aggregate_id == interview_id
        assert result.version == 0  # Version returned from append_events
        assert result.event_count == 1
        assert "created successfully" in result.message.lower()

        # Verify event store was called
        mock_event_store.append_events.assert_called_once()


@pytest.mark.asyncio
class TestSentenceCommandHandler:
    """Test sentence command handler."""

    async def test_create_sentence_command(self):
        """Test creating a sentence via command."""
        # Create mock event store
        mock_event_store = MagicMock(spec=EventStoreClient)
        mock_event_store.read_stream = AsyncMock(return_value=[])  # No existing events
        mock_event_store.append_events = AsyncMock(return_value=0)  # Return version 0

        # Create handler with mocked event store
        handler = SentenceCommandHandler(event_store=mock_event_store)

        # Create command
        sentence_id = str(uuid.uuid4())
        interview_id = str(uuid.uuid4())
        command = CreateSentenceCommand(
            sentence_id=sentence_id,
            interview_id=interview_id,
            index=0,
            text="This is a test sentence.",
            actor=Actor(user_id="test-user", actor_type=ActorType.SYSTEM, display="Test System"),
        )

        # Execute command
        result = await handler.handle(command)

        # Verify result
        assert result.success is True
        assert result.aggregate_id == sentence_id
        assert result.version == 0  # Version returned from append_events
        assert result.event_count == 1

        # Verify event store was called
        mock_event_store.append_events.assert_called_once()

    async def test_edit_sentence_command(self):
        """Test editing a sentence via command."""
        from datetime import datetime, timezone

        # Create mock event store that simulates existing sentence
        mock_event_store = MagicMock(spec=EventStoreClient)

        # Track append calls to simulate version increments
        append_calls = []

        async def mock_append(*args, **kwargs):
            version = len(append_calls)
            append_calls.append(args)
            return version

        mock_event_store.append_events = AsyncMock(side_effect=mock_append)

        sentence_id = str(uuid.uuid4())
        interview_id = str(uuid.uuid4())
        correlation_id = str(uuid.uuid4())

        # Create a mock event for the existing sentence (matches EventEnvelope structure)
        existing_event = MagicMock()
        existing_event.event_type = "SentenceCreated"
        existing_event.version = 0
        existing_event.data = {
            "sentence_id": sentence_id,
            "interview_id": interview_id,
            "index": 0,
            "text": "Original text.",
            "speaker": None,
            "start_ms": None,
            "end_ms": None,
        }
        existing_event.actor = Actor(user_id="test-user", actor_type=ActorType.SYSTEM, display="Test System")
        existing_event.correlation_id = correlation_id
        existing_event.occurred_at = datetime.now(timezone.utc)

        # First call returns empty (create), second call returns the created event (edit)
        mock_event_store.read_stream = AsyncMock(side_effect=[[], [existing_event]])

        # Create handler with mocked event store
        handler = SentenceCommandHandler(event_store=mock_event_store)

        # First create a sentence
        create_command = CreateSentenceCommand(
            sentence_id=sentence_id,
            interview_id=interview_id,
            index=0,
            text="Original text.",
            actor=Actor(user_id="test-user", actor_type=ActorType.SYSTEM, display="Test System"),
        )
        await handler.handle(create_command)

        # Now edit it
        edit_command = EditSentenceCommand(
            sentence_id=sentence_id,
            interview_id=interview_id,
            new_text="Edited text.",
            editor_type="human",
            actor=Actor(user_id="test-user", actor_type=ActorType.HUMAN, display="Test User"),
        )
        result = await handler.handle(edit_command)

        # Verify result
        assert result.success is True
        assert result.aggregate_id == sentence_id
        assert result.version == 1  # Version returned from second append (0 was first, 1 is second)
        assert result.event_count == 1  # Only the edit event

        # Verify event store was called twice (create + edit)
        assert len(append_calls) == 2
