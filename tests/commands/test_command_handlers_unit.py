"""
Unit tests for command handlers with mocked dependencies.

Tests command validation, aggregate loading/saving, and error handling
without requiring actual EventStoreDB.
"""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.commands.handlers import InterviewCommandHandler, SentenceCommandHandler
from src.commands.interview_commands import (
    ChangeInterviewStatusCommand,
    CreateInterviewCommand,
    UpdateInterviewCommand,
)
from src.commands.sentence_commands import (
    CreateSentenceCommand,
    EditSentenceCommand,
    GenerateAnalysisCommand,
    OverrideAnalysisCommand,
)
from src.events.aggregates import Interview, Sentence
from src.events.envelope import Actor, ActorType
from src.events.interview_events import InterviewStatus


@pytest.mark.asyncio
class TestInterviewCommandHandlerUnit:
    """Unit tests for InterviewCommandHandler with mocks."""

    async def test_create_interview_success(self):
        """Test successful interview creation."""
        # Setup
        interview_id = str(uuid.uuid4())
        project_id = str(uuid.uuid4())

        # Mock repository
        mock_repo = AsyncMock()
        mock_repo.load.return_value = None  # Interview doesn't exist yet
        mock_repo.save = AsyncMock()

        # Mock repository factory
        mock_factory = MagicMock()
        mock_factory.create_interview_repository.return_value = mock_repo

        # Create handler with mocked event store
        with patch("src.commands.handlers.RepositoryFactory", return_value=mock_factory):
            handler = InterviewCommandHandler()
            handler.repo_factory = mock_factory

            # Create command
            command = CreateInterviewCommand(
                interview_id=interview_id,
                project_id=project_id,
                title="Test Interview",
                source="test_file.txt",
                language="en",
                actor=Actor(user_id="test-user", actor_type=ActorType.SYSTEM, display="Test System"),
            )

            # Execute
            result = await handler.handle(command)

            # Verify
            assert result.success is True
            assert result.aggregate_id == interview_id
            assert result.version == 0
            assert result.event_count == 1
            assert "created successfully" in result.message.lower()

            # Verify repository interactions
            mock_repo.load.assert_called_once_with(interview_id)
            mock_repo.save.assert_called_once()

            # Verify the aggregate was created correctly
            saved_aggregate = mock_repo.save.call_args[0][0]
            assert isinstance(saved_aggregate, Interview)
            assert saved_aggregate.aggregate_id == interview_id
            assert saved_aggregate.title == "Test Interview"

    async def test_create_interview_already_exists(self):
        """Test creating an interview that already exists."""
        interview_id = str(uuid.uuid4())

        # Mock repository that returns existing interview
        existing_interview = Interview(interview_id)
        mock_repo = AsyncMock()
        mock_repo.load.return_value = existing_interview

        mock_factory = MagicMock()
        mock_factory.create_interview_repository.return_value = mock_repo

        with patch("src.commands.handlers.RepositoryFactory", return_value=mock_factory):
            handler = InterviewCommandHandler()
            handler.repo_factory = mock_factory

            command = CreateInterviewCommand(
                interview_id=interview_id,
                project_id=str(uuid.uuid4()),
                title="Test",
                source="test.txt",
            )

            # Should raise CommandValidationError
            from src.commands import CommandValidationError

            with pytest.raises(CommandValidationError) as exc_info:
                await handler.handle(command)

            assert "already exists" in str(exc_info.value)
            assert exc_info.value.field == "interview_id"

    async def test_update_interview_not_found(self):
        """Test updating an interview that doesn't exist."""
        interview_id = str(uuid.uuid4())

        # Mock repository that returns None (not found)
        mock_repo = AsyncMock()
        mock_repo.load.return_value = None

        mock_factory = MagicMock()
        mock_factory.create_interview_repository.return_value = mock_repo

        with patch("src.commands.handlers.RepositoryFactory", return_value=mock_factory):
            handler = InterviewCommandHandler()
            handler.repo_factory = mock_factory

            command = UpdateInterviewCommand(
                interview_id=interview_id,
                title="Updated Title",
            )

            # Should raise CommandValidationError
            from src.commands import CommandValidationError

            with pytest.raises(CommandValidationError) as exc_info:
                await handler.handle(command)

            assert "not found" in str(exc_info.value)

    async def test_change_status_invalid_status(self):
        """Test changing to an invalid status."""
        interview_id = str(uuid.uuid4())

        # Mock repository with existing interview
        existing_interview = Interview(interview_id)
        existing_interview.create(
            title="Test",
            source="test.txt",
            actor=Actor(user_id="test", actor_type=ActorType.SYSTEM, display="Test"),
        )
        mock_repo = AsyncMock()
        mock_repo.load.return_value = existing_interview

        mock_factory = MagicMock()
        mock_factory.create_interview_repository.return_value = mock_repo

        with patch("src.commands.handlers.RepositoryFactory", return_value=mock_factory):
            handler = InterviewCommandHandler()
            handler.repo_factory = mock_factory

            command = ChangeInterviewStatusCommand(
                interview_id=interview_id,
                new_status="INVALID_STATUS",
            )

            # Should raise CommandValidationError
            from src.commands import CommandValidationError

            with pytest.raises(CommandValidationError) as exc_info:
                await handler.handle(command)

            assert "Invalid status" in str(exc_info.value)


@pytest.mark.asyncio
class TestSentenceCommandHandlerUnit:
    """Unit tests for SentenceCommandHandler with mocks."""

    async def test_create_sentence_success(self):
        """Test successful sentence creation."""
        sentence_id = str(uuid.uuid4())
        interview_id = str(uuid.uuid4())

        # Mock repository
        mock_repo = AsyncMock()
        mock_repo.load.return_value = None  # Sentence doesn't exist yet
        mock_repo.save = AsyncMock()

        mock_factory = MagicMock()
        mock_factory.create_sentence_repository.return_value = mock_repo

        with patch("src.commands.handlers.RepositoryFactory", return_value=mock_factory):
            handler = SentenceCommandHandler()
            handler.repo_factory = mock_factory

            command = CreateSentenceCommand(
                sentence_id=sentence_id,
                interview_id=interview_id,
                index=0,
                text="This is a test sentence.",
                actor=Actor(user_id="test-user", actor_type=ActorType.SYSTEM, display="Test System"),
            )

            result = await handler.handle(command)

            assert result.success is True
            assert result.aggregate_id == sentence_id
            assert result.version == 0
            assert result.event_count == 1

            mock_repo.save.assert_called_once()
            saved_aggregate = mock_repo.save.call_args[0][0]
            assert isinstance(saved_aggregate, Sentence)
            assert saved_aggregate.text == "This is a test sentence."

    async def test_edit_sentence_success(self):
        """Test successful sentence edit."""
        sentence_id = str(uuid.uuid4())
        interview_id = str(uuid.uuid4())

        # Create existing sentence and mark events as committed (simulating loaded from repo)
        existing_sentence = Sentence(sentence_id)
        existing_sentence.create(
            interview_id=interview_id,
            index=0,
            text="Original text.",
            actor=Actor(user_id="test", actor_type=ActorType.SYSTEM, display="Test"),
        )
        # Mark events as committed to simulate that this aggregate was loaded from the event store
        # The version should be 0 (the create event)
        existing_sentence.mark_events_as_committed()
        # After marking as committed, version stays at 0 (last applied event)

        mock_repo = AsyncMock()
        mock_repo.load.return_value = existing_sentence
        mock_repo.save = AsyncMock()

        mock_factory = MagicMock()
        mock_factory.create_sentence_repository.return_value = mock_repo

        with patch("src.commands.handlers.RepositoryFactory", return_value=mock_factory):
            handler = SentenceCommandHandler()
            handler.repo_factory = mock_factory

            command = EditSentenceCommand(
                sentence_id=sentence_id,
                interview_id=interview_id,
                new_text="Edited text.",
                editor_type="human",
                actor=Actor(user_id="test-user", actor_type=ActorType.HUMAN, display="Test User"),
            )

            result = await handler.handle(command)

            assert result.success is True
            assert result.version == 1  # Second event
            # Only the edit event should be uncommitted
            assert result.event_count == 1

            mock_repo.save.assert_called_once()
            saved_aggregate = mock_repo.save.call_args[0][0]
            assert saved_aggregate.text == "Edited text."

    async def test_generate_analysis_success(self):
        """Test successful analysis generation."""
        sentence_id = str(uuid.uuid4())
        interview_id = str(uuid.uuid4())

        # Create existing sentence and mark events as committed
        existing_sentence = Sentence(sentence_id)
        existing_sentence.create(
            interview_id=interview_id,
            index=0,
            text="Test sentence.",
            actor=Actor(user_id="test", actor_type=ActorType.SYSTEM, display="Test"),
        )
        existing_sentence.mark_events_as_committed()

        mock_repo = AsyncMock()
        mock_repo.load.return_value = existing_sentence
        mock_repo.save = AsyncMock()

        mock_factory = MagicMock()
        mock_factory.create_sentence_repository.return_value = mock_repo

        with patch("src.commands.handlers.RepositoryFactory", return_value=mock_factory):
            handler = SentenceCommandHandler()
            handler.repo_factory = mock_factory

            command = GenerateAnalysisCommand(
                sentence_id=sentence_id,
                interview_id=interview_id,
                model="gpt-4",
                model_version="1.0",
                classification={"function_type": "question", "structure_type": "simple"},
                keywords=["test", "keyword"],
                topics=["testing"],
                domain_keywords=["qa"],
                confidence=0.95,
            )

            result = await handler.handle(command)

            assert result.success is True
            assert result.version == 1
            assert result.event_count == 1

            mock_repo.save.assert_called_once()

    async def test_edit_sentence_invalid_editor_type(self):
        """Test editing with invalid editor type."""
        sentence_id = str(uuid.uuid4())
        interview_id = str(uuid.uuid4())

        # Create existing sentence
        existing_sentence = Sentence(sentence_id)
        existing_sentence.create(
            interview_id=interview_id,
            index=0,
            text="Original text.",
            actor=Actor(user_id="test", actor_type=ActorType.SYSTEM, display="Test"),
        )

        mock_repo = AsyncMock()
        mock_repo.load.return_value = existing_sentence

        mock_factory = MagicMock()
        mock_factory.create_sentence_repository.return_value = mock_repo

        with patch("src.commands.handlers.RepositoryFactory", return_value=mock_factory):
            handler = SentenceCommandHandler()
            handler.repo_factory = mock_factory

            command = EditSentenceCommand(
                sentence_id=sentence_id,
                interview_id=interview_id,
                new_text="Edited text.",
                editor_type="INVALID_TYPE",
            )

            from src.commands import CommandValidationError

            with pytest.raises(CommandValidationError) as exc_info:
                await handler.handle(command)

            assert "Invalid editor type" in str(exc_info.value)
