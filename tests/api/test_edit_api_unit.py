"""
Unit tests for edit API endpoints (M2.6).

Tests the API layer with mocked dependencies to validate:
- Request/response handling
- Actor tracking
- Correlation ID propagation
- Error handling
- Validation logic
"""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from src.api.routers.edits import (
    EditResponse,
    EditSentenceRequest,
    OverrideAnalysisRequest,
    create_actor_from_request,
    edit_sentence,
    get_sentence_history,
    override_analysis,
)
from src.commands.handlers import CommandResult
from src.events.envelope import Actor, ActorType


@pytest.mark.asyncio
class TestEditSentenceEndpoint:
    """Test edit_sentence endpoint."""

    @pytest.fixture
    def mock_handler(self):
        """Mock SentenceCommandHandler."""
        handler = MagicMock()
        handler.handle_edit_sentence = AsyncMock()
        return handler

    @pytest.fixture
    def interview_id(self):
        """Test interview ID."""
        return str(uuid.uuid4())

    @pytest.fixture
    def sentence_index(self):
        """Test sentence index."""
        return 0

    async def test_edit_sentence_success(
        self, mock_handler, interview_id, sentence_index
    ):
        """Test successful sentence edit."""
        # Arrange
        request = EditSentenceRequest(
            text="Edited sentence text", editor_type="human", note="Fixed typo"
        )

        # Mock handler response
        sentence_id = str(
            uuid.uuid5(uuid.NAMESPACE_DNS, f"{interview_id}:{sentence_index}")
        )
        mock_handler.handle_edit_sentence.return_value = CommandResult(
            aggregate_id=sentence_id, version=1, event_count=1
        )

        # Act
        with patch(
            "src.api.routers.edits.get_sentence_command_handler",
            return_value=mock_handler,
        ):
            response = await edit_sentence(
                interview_id=interview_id,
                sentence_index=sentence_index,
                request=request,
                x_user_id="test-user",
                x_correlation_id="test-correlation-id",
            )

        # Assert
        assert response.status == "accepted"
        assert response.version == 1
        assert response.event_count == 1
        assert "accepted" in response.message.lower()

        # Verify handler was called with correct command
        mock_handler.handle_edit_sentence.assert_called_once()
        command = mock_handler.handle_edit_sentence.call_args[0][0]
        assert command.new_text == "Edited sentence text"
        assert command.editor_type == "human"
        assert command.actor.user_id == "test-user"
        assert command.correlation_id == "test-correlation-id"

    async def test_edit_sentence_generates_deterministic_uuid(
        self, mock_handler, interview_id, sentence_index
    ):
        """Test that sentence UUID is generated deterministically."""
        # Arrange
        request = EditSentenceRequest(text="New text")

        sentence_id = str(
            uuid.uuid5(uuid.NAMESPACE_DNS, f"{interview_id}:{sentence_index}")
        )
        mock_handler.handle_edit_sentence.return_value = CommandResult(
            aggregate_id=sentence_id, version=1, event_count=1
        )

        # Act
        with patch(
            "src.api.routers.edits.get_sentence_command_handler",
            return_value=mock_handler,
        ):
            response1 = await edit_sentence(
                interview_id=interview_id,
                sentence_index=sentence_index,
                request=request,
                x_user_id=None,
                x_correlation_id=None,
            )

            response2 = await edit_sentence(
                interview_id=interview_id,
                sentence_index=sentence_index,
                request=request,
                x_user_id=None,
                x_correlation_id=None,
            )

        # Assert - both calls should produce same sentence_id
        assert response1.sentence_id == response2.sentence_id

        # Verify expected UUID
        expected_id = str(
            uuid.uuid5(uuid.NAMESPACE_DNS, f"{interview_id}:{sentence_index}")
        )
        assert response1.sentence_id == expected_id

    async def test_edit_sentence_uses_anonymous_when_no_user_id(
        self, mock_handler, interview_id, sentence_index
    ):
        """Test that 'anonymous' is used when no user ID provided."""
        # Arrange
        request = EditSentenceRequest(text="New text")

        sentence_id = str(
            uuid.uuid5(uuid.NAMESPACE_DNS, f"{interview_id}:{sentence_index}")
        )
        mock_handler.handle_edit_sentence.return_value = CommandResult(
            aggregate_id=sentence_id, version=1, event_count=1
        )

        # Act
        with patch(
            "src.api.routers.edits.get_sentence_command_handler",
            return_value=mock_handler,
        ):
            await edit_sentence(
                interview_id=interview_id,
                sentence_index=sentence_index,
                request=request,
                x_user_id=None,  # No user ID
                x_correlation_id=None,
            )

        # Assert
        command = mock_handler.handle_edit_sentence.call_args[0][0]
        assert command.actor.user_id == "anonymous"

    async def test_edit_sentence_generates_correlation_id_if_not_provided(
        self, mock_handler, interview_id, sentence_index
    ):
        """Test that correlation ID is generated if not provided."""
        # Arrange
        request = EditSentenceRequest(text="New text")

        sentence_id = str(
            uuid.uuid5(uuid.NAMESPACE_DNS, f"{interview_id}:{sentence_index}")
        )
        mock_handler.handle_edit_sentence.return_value = CommandResult(
            aggregate_id=sentence_id, version=1, event_count=1
        )

        # Act
        with patch(
            "src.api.routers.edits.get_sentence_command_handler",
            return_value=mock_handler,
        ):
            await edit_sentence(
                interview_id=interview_id,
                sentence_index=sentence_index,
                request=request,
                x_user_id=None,
                x_correlation_id=None,  # No correlation ID
            )

        # Assert
        command = mock_handler.handle_edit_sentence.call_args[0][0]
        assert command.correlation_id is not None
        assert len(command.correlation_id) == 36  # UUID format

    async def test_edit_sentence_sentence_not_found(
        self, mock_handler, interview_id, sentence_index
    ):
        """Test error handling when sentence not found."""
        # Arrange
        request = EditSentenceRequest(text="New text")

        # Mock handler to raise ValueError (sentence not found)
        mock_handler.handle_edit_sentence.side_effect = ValueError(
            "Sentence not found"
        )

        # Act & Assert
        with patch(
            "src.api.routers.edits.get_sentence_command_handler",
            return_value=mock_handler,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await edit_sentence(
                    interview_id=interview_id,
                    sentence_index=sentence_index,
                    request=request,
                    x_user_id=None,
                    x_correlation_id=None,
                )

            assert exc_info.value.status_code == 404
            assert "not found" in str(exc_info.value.detail).lower()

    async def test_edit_sentence_internal_error(
        self, mock_handler, interview_id, sentence_index
    ):
        """Test error handling for internal errors."""
        # Arrange
        request = EditSentenceRequest(text="New text")

        # Mock handler to raise unexpected error
        mock_handler.handle_edit_sentence.side_effect = Exception("Database error")

        # Act & Assert
        with patch(
            "src.api.routers.edits.get_sentence_command_handler",
            return_value=mock_handler,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await edit_sentence(
                    interview_id=interview_id,
                    sentence_index=sentence_index,
                    request=request,
                    x_user_id=None,
                    x_correlation_id=None,
                )

            assert exc_info.value.status_code == 500
            assert "internal error" in str(exc_info.value.detail).lower()


@pytest.mark.asyncio
class TestOverrideAnalysisEndpoint:
    """Test override_analysis endpoint."""

    @pytest.fixture
    def mock_handler(self):
        """Mock SentenceCommandHandler."""
        handler = MagicMock()
        handler.handle_override_analysis = AsyncMock()
        return handler

    @pytest.fixture
    def interview_id(self):
        """Test interview ID."""
        return str(uuid.uuid4())

    @pytest.fixture
    def sentence_index(self):
        """Test sentence index."""
        return 0

    async def test_override_analysis_success(
        self, mock_handler, interview_id, sentence_index
    ):
        """Test successful analysis override."""
        # Arrange
        request = OverrideAnalysisRequest(
            function_type="question",
            structure_type="simple",
            purpose="inquiry",
            keywords=["test", "keyword"],
            topics=["testing"],
            domain_keywords=["python"],
            note="Human correction",
        )

        # Mock handler response
        sentence_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{interview_id}:{sentence_index}"))
        mock_handler.handle_override_analysis.return_value = CommandResult(aggregate_id=sentence_id, 
            version=2, event_count=1
        )

        # Act
        with patch(
            "src.api.routers.edits.get_sentence_command_handler",
            return_value=mock_handler,
        ):
            response = await override_analysis(
                interview_id=interview_id,
                sentence_index=sentence_index,
                request=request,
                x_user_id="test-user",
                x_correlation_id="test-correlation-id",
            )

        # Assert
        assert response.status == "accepted"
        assert response.version == 2
        assert response.event_count == 1
        assert "accepted" in response.message.lower()

        # Verify handler was called with correct command
        mock_handler.handle_override_analysis.assert_called_once()
        command = mock_handler.handle_override_analysis.call_args[0][0]
        assert command.fields_overridden["function_type"] == "question"
        assert command.fields_overridden["structure_type"] == "simple"
        assert command.fields_overridden["purpose"] == "inquiry"
        assert command.fields_overridden["keywords"] == ["test", "keyword"]
        assert command.note == "Human correction"
        assert command.actor.user_id == "test-user"

    async def test_override_analysis_partial_override(
        self, mock_handler, interview_id, sentence_index
    ):
        """Test partial override (only some fields provided)."""
        # Arrange
        request = OverrideAnalysisRequest(
            function_type="question",
            keywords=["corrected"],
            # Other fields not provided
        )

        sentence_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{interview_id}:{sentence_index}"))
        mock_handler.handle_override_analysis.return_value = CommandResult(aggregate_id=sentence_id, 
            version=2, event_count=1
        )

        # Act
        with patch(
            "src.api.routers.edits.get_sentence_command_handler",
            return_value=mock_handler,
        ):
            await override_analysis(
                interview_id=interview_id,
                sentence_index=sentence_index,
                request=request,
                x_user_id=None,
                x_correlation_id=None,
            )

        # Assert - only provided fields should be in fields_overridden
        command = mock_handler.handle_override_analysis.call_args[0][0]
        assert "function_type" in command.fields_overridden
        assert "keywords" in command.fields_overridden
        assert "structure_type" not in command.fields_overridden
        assert "purpose" not in command.fields_overridden
        assert len(command.fields_overridden) == 2

    async def test_override_analysis_no_fields_provided(
        self, mock_handler, interview_id, sentence_index
    ):
        """Test error when no override fields provided."""
        # Arrange
        request = OverrideAnalysisRequest(
            note="Just a note"  # No actual override fields
        )

        # Act & Assert
        with patch(
            "src.api.routers.edits.get_sentence_command_handler",
            return_value=mock_handler,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await override_analysis(
                    interview_id=interview_id,
                    sentence_index=sentence_index,
                    request=request,
                    x_user_id=None,
                    x_correlation_id=None,
                )

            assert exc_info.value.status_code == 400
            assert "at least one" in str(exc_info.value.detail).lower()

    async def test_override_analysis_sentence_not_found(
        self, mock_handler, interview_id, sentence_index
    ):
        """Test error handling when sentence not found."""
        # Arrange
        request = OverrideAnalysisRequest(function_type="question")

        # Mock handler to raise ValueError
        mock_handler.handle_override_analysis.side_effect = ValueError(
            "Sentence not found"
        )

        # Act & Assert
        with patch(
            "src.api.routers.edits.get_sentence_command_handler",
            return_value=mock_handler,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await override_analysis(
                    interview_id=interview_id,
                    sentence_index=sentence_index,
                    request=request,
                    x_user_id=None,
                    x_correlation_id=None,
                )

            assert exc_info.value.status_code == 404

    async def test_override_analysis_generates_deterministic_uuid(
        self, mock_handler, interview_id, sentence_index
    ):
        """Test that sentence UUID is generated deterministically."""
        # Arrange
        request = OverrideAnalysisRequest(function_type="question")

        sentence_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{interview_id}:{sentence_index}"))
        mock_handler.handle_override_analysis.return_value = CommandResult(aggregate_id=sentence_id, 
            version=2, event_count=1
        )

        # Act
        with patch(
            "src.api.routers.edits.get_sentence_command_handler",
            return_value=mock_handler,
        ):
            response1 = await override_analysis(
                interview_id=interview_id,
                sentence_index=sentence_index,
                request=request,
                x_user_id=None,
                x_correlation_id=None,
            )

            response2 = await override_analysis(
                interview_id=interview_id,
                sentence_index=sentence_index,
                request=request,
                x_user_id=None,
                x_correlation_id=None,
            )

        # Assert - both calls should produce same sentence_id
        assert response1.sentence_id == response2.sentence_id


@pytest.mark.asyncio
class TestGetSentenceHistoryEndpoint:
    """Test get_sentence_history endpoint."""

    @pytest.fixture
    def mock_repository(self):
        """Mock SentenceRepository."""
        repo = MagicMock()
        repo.load = AsyncMock()
        return repo

    @pytest.fixture
    def mock_event_store(self):
        """Mock EventStoreClient."""
        store = MagicMock()
        store.read_stream = AsyncMock()
        return store

    @pytest.fixture
    def interview_id(self):
        """Test interview ID."""
        return str(uuid.uuid4())

    @pytest.fixture
    def sentence_index(self):
        """Test sentence index."""
        return 0

    async def test_get_sentence_history_success(
        self, mock_repository, mock_event_store, interview_id, sentence_index
    ):
        """Test successful retrieval of sentence history."""
        # Arrange
        from src.events.aggregates import Sentence
        from src.events.envelope import EventEnvelope

        # Mock sentence aggregate
        sentence_id = str(
            uuid.uuid5(uuid.NAMESPACE_DNS, f"{interview_id}:{sentence_index}")
        )
        sentence = Sentence(aggregate_id=sentence_id)
        sentence.version = 2
        sentence.text = "Current text"
        mock_repository.load.return_value = sentence

        # Mock events
        from datetime import datetime, timezone
        
        event1 = MagicMock()
        event1.event_type = "SentenceCreated"
        event1.version = 0
        event1.occurred_at = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        event1.actor = MagicMock()
        event1.actor.actor_type = "system"
        event1.actor.user_id = "pipeline"
        event1.correlation_id = "corr-1"
        event1.data = {"text": "Original text"}

        event2 = MagicMock()
        event2.event_type = "SentenceEdited"
        event2.version = 1
        event2.occurred_at = datetime(2025, 1, 1, 0, 5, 0, tzinfo=timezone.utc)
        event2.actor = MagicMock()
        event2.actor.actor_type = "user"
        event2.actor.user_id = "test-user"
        event2.correlation_id = "corr-2"
        event2.data = {"text": "Edited text"}

        mock_event_store.read_stream.return_value = [event1, event2]

        # Act
        with patch(
            "src.api.routers.edits.get_sentence_repository", return_value=mock_repository
        ), patch("src.api.routers.edits.get_event_store", return_value=mock_event_store):
            history = await get_sentence_history(
                interview_id=interview_id, sentence_index=sentence_index
            )

        # Assert
        assert history["sentence_id"] == sentence.aggregate_id
        assert history["interview_id"] == interview_id
        assert history["sentence_index"] == sentence_index
        assert history["current_version"] == 2
        assert history["current_text"] == "Current text"
        assert history["event_count"] == 2
        assert len(history["events"]) == 2

        # Verify event details
        assert history["events"][0]["event_type"] == "SentenceCreated"
        assert history["events"][0]["version"] == 0
        assert history["events"][0]["actor"]["user_id"] == "pipeline"

        assert history["events"][1]["event_type"] == "SentenceEdited"
        assert history["events"][1]["version"] == 1
        assert history["events"][1]["actor"]["user_id"] == "test-user"

    async def test_get_sentence_history_sentence_not_found(
        self, mock_repository, interview_id, sentence_index
    ):
        """Test error when sentence not found."""
        # Arrange
        mock_repository.load.return_value = None

        # Act & Assert
        with patch(
            "src.api.routers.edits.get_sentence_repository", return_value=mock_repository
        ):
            with pytest.raises(HTTPException) as exc_info:
                await get_sentence_history(
                    interview_id=interview_id, sentence_index=sentence_index
                )

            assert exc_info.value.status_code == 404
            assert "not found" in str(exc_info.value.detail).lower()


class TestCreateActorFromRequest:
    """Test create_actor_from_request helper function."""

    def test_uses_provided_user_id(self):
        """Test that provided user_id takes precedence."""
        actor = create_actor_from_request(user_id="explicit-user", x_user_id="header-user")

        assert actor.actor_type == ActorType.HUMAN
        assert actor.user_id == "explicit-user"

    def test_uses_header_user_id(self):
        """Test that header user_id is used when no explicit user_id."""
        actor = create_actor_from_request(user_id=None, x_user_id="header-user")

        assert actor.actor_type == ActorType.HUMAN
        assert actor.user_id == "header-user"

    def test_uses_anonymous_when_no_user_id(self):
        """Test that 'anonymous' is used when no user_id provided."""
        actor = create_actor_from_request(user_id=None, x_user_id=None)

        assert actor.actor_type == ActorType.HUMAN
        assert actor.user_id == "anonymous"
