"""
Tests for command handlers.

Validates that commands correctly create events and update aggregates.
"""

import uuid

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
        # Create handler with in-memory event store (for testing)
        handler = InterviewCommandHandler()

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
        assert result.version == 0  # First event
        assert result.event_count == 1
        assert "created successfully" in result.message.lower()


@pytest.mark.asyncio
class TestSentenceCommandHandler:
    """Test sentence command handler."""

    async def test_create_sentence_command(self):
        """Test creating a sentence via command."""
        # Create handler
        handler = SentenceCommandHandler()

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
        assert result.version == 0  # First event
        assert result.event_count == 1

    async def test_edit_sentence_command(self):
        """Test editing a sentence via command."""
        # Create handler
        handler = SentenceCommandHandler()

        # First create a sentence
        sentence_id = str(uuid.uuid4())
        interview_id = str(uuid.uuid4())
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
        assert result.version == 1  # Second event
        assert result.event_count == 1  # Only the edit event
