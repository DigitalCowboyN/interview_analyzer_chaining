"""
Unit tests for Repository pattern (src/events/repository.py).

Tests aggregate loading, saving, concurrency handling, and factory functions.
"""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.events.aggregates import Interview, Sentence
from src.events.envelope import Actor, ActorType, EventEnvelope
from src.events.interview_events import InterviewStatus
from src.events.repository import (
    InterviewRepository,
    Repository,
    RepositoryFactory,
    SentenceRepository,
    get_interview_repository,
    get_repository_factory,
    get_sentence_repository,
)
from src.events.sentence_events import SentenceStatus
from src.events.store import ConcurrencyError, StreamNotFoundError


def create_interview_created_envelope(aggregate_id: str, version: int = 0) -> EventEnvelope:
    """Helper to create InterviewCreated event envelopes."""
    return EventEnvelope(
        event_type="InterviewCreated",
        aggregate_type="Interview",
        aggregate_id=aggregate_id,
        version=version,
        data={
            "title": "Test Interview",
            "source": "test.txt",
            "language": "en",
            "status": "created",
        },
        actor=Actor(actor_type=ActorType.SYSTEM),
    )


def create_sentence_created_envelope(
    aggregate_id: str, interview_id: str, index: int = 0, version: int = 0
) -> EventEnvelope:
    """Helper to create SentenceCreated event envelopes."""
    return EventEnvelope(
        event_type="SentenceCreated",
        aggregate_type="Sentence",
        aggregate_id=aggregate_id,
        version=version,
        data={
            "interview_id": interview_id,
            "index": index,
            "text": "Test sentence.",
            "speaker": "Speaker1",
            "status": "created",
        },
        actor=Actor(actor_type=ActorType.SYSTEM),
    )


class TestInterviewRepositoryBasics:
    """Test basic InterviewRepository functionality."""

    def test_stream_naming_convention(self):
        """Test that stream name follows Interview-{id} convention."""
        mock_store = MagicMock()
        repo = InterviewRepository(mock_store)

        aggregate_id = str(uuid.uuid4())
        stream_name = repo._get_stream_name(aggregate_id)

        assert stream_name == f"Interview-{aggregate_id}"

    def test_create_aggregate_returns_interview(self):
        """Test that _create_aggregate returns an Interview instance."""
        mock_store = MagicMock()
        repo = InterviewRepository(mock_store)

        aggregate_id = str(uuid.uuid4())
        aggregate = repo._create_aggregate(aggregate_id)

        assert isinstance(aggregate, Interview)
        assert aggregate.aggregate_id == aggregate_id


class TestSentenceRepositoryBasics:
    """Test basic SentenceRepository functionality."""

    def test_stream_naming_convention(self):
        """Test that stream name follows Sentence-{id} convention."""
        mock_store = MagicMock()
        repo = SentenceRepository(mock_store)

        aggregate_id = str(uuid.uuid4())
        stream_name = repo._get_stream_name(aggregate_id)

        assert stream_name == f"Sentence-{aggregate_id}"

    def test_create_aggregate_returns_sentence(self):
        """Test that _create_aggregate returns a Sentence instance."""
        mock_store = MagicMock()
        repo = SentenceRepository(mock_store)

        aggregate_id = str(uuid.uuid4())
        aggregate = repo._create_aggregate(aggregate_id)

        assert isinstance(aggregate, Sentence)
        assert aggregate.aggregate_id == aggregate_id


@pytest.mark.asyncio
class TestRepositoryLoad:
    """Test Repository.load() method."""

    async def test_load_existing_aggregate(self):
        """Test loading an existing aggregate from stream."""
        mock_store = AsyncMock()
        repo = InterviewRepository(mock_store)

        aggregate_id = str(uuid.uuid4())
        mock_events = [create_interview_created_envelope(aggregate_id)]
        mock_store.read_stream.return_value = mock_events

        interview = await repo.load(aggregate_id)

        assert interview is not None
        assert interview.aggregate_id == aggregate_id
        assert interview.title == "Test Interview"
        assert interview.source == "test.txt"
        mock_store.read_stream.assert_called_once_with(f"Interview-{aggregate_id}")

    async def test_load_nonexistent_aggregate_returns_none(self):
        """Test loading non-existent aggregate returns None."""
        mock_store = AsyncMock()
        repo = InterviewRepository(mock_store)

        aggregate_id = str(uuid.uuid4())
        mock_store.read_stream.side_effect = StreamNotFoundError("Not found")

        interview = await repo.load(aggregate_id)

        assert interview is None

    async def test_load_empty_stream_returns_none(self):
        """Test loading from empty stream returns None."""
        mock_store = AsyncMock()
        repo = InterviewRepository(mock_store)

        aggregate_id = str(uuid.uuid4())
        mock_store.read_stream.return_value = []

        interview = await repo.load(aggregate_id)

        assert interview is None

    async def test_load_multiple_events_reconstructs_state(self):
        """Test loading aggregate with multiple events reconstructs full state."""
        mock_store = AsyncMock()
        repo = InterviewRepository(mock_store)

        aggregate_id = str(uuid.uuid4())
        created_event = create_interview_created_envelope(aggregate_id, version=0)
        status_changed_event = EventEnvelope(
            event_type="StatusChanged",
            aggregate_type="Interview",
            aggregate_id=aggregate_id,
            version=1,
            data={
                "from_status": "created",
                "to_status": "processing",
                "reason": "Starting analysis",
            },
            actor=Actor(actor_type=ActorType.SYSTEM),
        )
        mock_store.read_stream.return_value = [created_event, status_changed_event]

        interview = await repo.load(aggregate_id)

        assert interview is not None
        assert interview.version == 1
        assert interview.status == InterviewStatus.PROCESSING


@pytest.mark.asyncio
class TestRepositorySave:
    """Test Repository.save() method."""

    async def test_save_new_aggregate(self):
        """Test saving a new aggregate."""
        mock_store = AsyncMock()
        mock_store.append_events.return_value = 0
        repo = InterviewRepository(mock_store)

        aggregate_id = str(uuid.uuid4())
        interview = Interview(aggregate_id)
        interview.create(title="New Interview", source="new.txt")

        await repo.save(interview)

        mock_store.append_events.assert_called_once()
        call_args = mock_store.append_events.call_args
        assert call_args.kwargs["stream_name"] == f"Interview-{aggregate_id}"
        assert call_args.kwargs["expected_version"] == -1  # New stream

    async def test_save_existing_aggregate(self):
        """Test saving an existing aggregate with new events."""
        mock_store = AsyncMock()
        mock_store.append_events.return_value = 2
        repo = InterviewRepository(mock_store)

        aggregate_id = str(uuid.uuid4())
        interview = Interview(aggregate_id)
        # Simulate loading from version 1
        interview.version = 1
        interview._uncommitted_events = []  # Clear any uncommitted

        # Add new event
        interview.change_status(new_status=InterviewStatus.COMPLETED, reason="Done")

        await repo.save(interview)

        mock_store.append_events.assert_called_once()
        call_args = mock_store.append_events.call_args
        assert call_args.kwargs["expected_version"] == 1  # Previous version

    async def test_save_no_uncommitted_events_does_nothing(self):
        """Test saving aggregate with no uncommitted events doesn't call append."""
        mock_store = AsyncMock()
        repo = InterviewRepository(mock_store)

        aggregate_id = str(uuid.uuid4())
        interview = Interview(aggregate_id)
        # No events created

        await repo.save(interview)

        mock_store.append_events.assert_not_called()

    async def test_save_marks_events_as_committed(self):
        """Test that save clears uncommitted events after successful append."""
        mock_store = AsyncMock()
        mock_store.append_events.return_value = 0
        repo = InterviewRepository(mock_store)

        aggregate_id = str(uuid.uuid4())
        interview = Interview(aggregate_id)
        interview.create(title="Test", source="test.txt")

        assert len(interview.get_uncommitted_events()) == 1

        await repo.save(interview)

        assert len(interview.get_uncommitted_events()) == 0

    async def test_save_retries_on_concurrency_error(self):
        """Test that save retries when concurrency error occurs."""
        mock_store = AsyncMock()
        # First call fails with concurrency error, second succeeds
        mock_store.append_events.side_effect = [
            ConcurrencyError("Conflict", expected_version=0, actual_version=1),
            1,  # Success on retry
        ]
        mock_store.read_stream.return_value = [
            create_interview_created_envelope(str(uuid.uuid4()), version=0)
        ]

        repo = InterviewRepository(mock_store)

        aggregate_id = str(uuid.uuid4())
        interview = Interview(aggregate_id)
        interview.create(title="Test", source="test.txt")

        await repo.save(interview, max_retries=1, retry_delay=0.01)

        assert mock_store.append_events.call_count == 2

    async def test_save_raises_after_exhausting_retries(self):
        """Test that save raises ConcurrencyError after max retries."""
        mock_store = AsyncMock()
        mock_store.append_events.side_effect = ConcurrencyError(
            "Conflict", expected_version=0, actual_version=1
        )
        mock_store.read_stream.return_value = [
            create_interview_created_envelope(str(uuid.uuid4()), version=0)
        ]

        repo = InterviewRepository(mock_store)

        aggregate_id = str(uuid.uuid4())
        interview = Interview(aggregate_id)
        interview.create(title="Test", source="test.txt")

        with pytest.raises(ConcurrencyError):
            await repo.save(interview, max_retries=2, retry_delay=0.01)

        # Initial + 2 retries = 3 attempts
        assert mock_store.append_events.call_count == 3


@pytest.mark.asyncio
class TestRepositoryExists:
    """Test Repository.exists() method."""

    async def test_exists_returns_true_when_found(self):
        """Test exists returns True when aggregate exists."""
        mock_store = AsyncMock()
        mock_store.get_stream_version.return_value = 5
        repo = InterviewRepository(mock_store)

        aggregate_id = str(uuid.uuid4())
        exists = await repo.exists(aggregate_id)

        assert exists is True
        mock_store.get_stream_version.assert_called_once_with(f"Interview-{aggregate_id}")

    async def test_exists_returns_false_when_not_found(self):
        """Test exists returns False when aggregate doesn't exist."""
        mock_store = AsyncMock()
        mock_store.get_stream_version.return_value = None
        repo = InterviewRepository(mock_store)

        aggregate_id = str(uuid.uuid4())
        exists = await repo.exists(aggregate_id)

        assert exists is False


@pytest.mark.asyncio
class TestRepositoryGetVersion:
    """Test Repository.get_version() method."""

    async def test_get_version_returns_current_version(self):
        """Test get_version returns the current stream version."""
        mock_store = AsyncMock()
        mock_store.get_stream_version.return_value = 10
        repo = InterviewRepository(mock_store)

        aggregate_id = str(uuid.uuid4())
        version = await repo.get_version(aggregate_id)

        assert version == 10
        mock_store.get_stream_version.assert_called_once_with(f"Interview-{aggregate_id}")

    async def test_get_version_returns_none_when_not_found(self):
        """Test get_version returns None when aggregate doesn't exist."""
        mock_store = AsyncMock()
        mock_store.get_stream_version.return_value = None
        repo = InterviewRepository(mock_store)

        aggregate_id = str(uuid.uuid4())
        version = await repo.get_version(aggregate_id)

        assert version is None


class TestRepositoryFactory:
    """Test RepositoryFactory class."""

    def test_factory_init(self):
        """Test factory initialization with event store."""
        mock_store = MagicMock()
        factory = RepositoryFactory(mock_store)

        assert factory.event_store is mock_store

    def test_create_interview_repository(self):
        """Test creating interview repository from factory."""
        mock_store = MagicMock()
        factory = RepositoryFactory(mock_store)

        repo = factory.create_interview_repository()

        assert isinstance(repo, InterviewRepository)
        assert repo.event_store is mock_store

    def test_create_sentence_repository(self):
        """Test creating sentence repository from factory."""
        mock_store = MagicMock()
        factory = RepositoryFactory(mock_store)

        repo = factory.create_sentence_repository()

        assert isinstance(repo, SentenceRepository)
        assert repo.event_store is mock_store


class TestGlobalRepositoryFunctions:
    """Test global repository factory functions."""

    def test_get_repository_factory_creates_singleton(self):
        """Test get_repository_factory returns singleton."""
        import src.events.repository as repo_module

        repo_module._global_factory = None

        mock_store = MagicMock()
        factory1 = get_repository_factory(mock_store)
        factory2 = get_repository_factory()

        assert factory1 is factory2

        # Cleanup
        repo_module._global_factory = None

    def test_get_repository_factory_uses_global_event_store(self):
        """Test get_repository_factory uses global event store if none provided."""
        import src.events.repository as repo_module

        repo_module._global_factory = None

        # Patch at the source module where it's imported from
        with patch("src.events.store.get_event_store_client") as mock_get_client:
            mock_store = MagicMock()
            mock_get_client.return_value = mock_store

            factory = get_repository_factory()

            mock_get_client.assert_called_once()
            assert factory.event_store is mock_store

        repo_module._global_factory = None

    def test_get_interview_repository(self):
        """Test get_interview_repository returns InterviewRepository."""
        import src.events.repository as repo_module

        repo_module._global_factory = None

        mock_store = MagicMock()
        with patch("src.events.store.get_event_store_client", return_value=mock_store):
            repo = get_interview_repository()

            assert isinstance(repo, InterviewRepository)

        repo_module._global_factory = None

    def test_get_sentence_repository(self):
        """Test get_sentence_repository returns SentenceRepository."""
        import src.events.repository as repo_module

        repo_module._global_factory = None

        mock_store = MagicMock()
        with patch("src.events.store.get_event_store_client", return_value=mock_store):
            repo = get_sentence_repository()

            assert isinstance(repo, SentenceRepository)

        repo_module._global_factory = None


@pytest.mark.asyncio
class TestSentenceRepositorySpecific:
    """Test SentenceRepository-specific functionality."""

    async def test_load_sentence_aggregate(self):
        """Test loading a sentence aggregate."""
        mock_store = AsyncMock()
        repo = SentenceRepository(mock_store)

        aggregate_id = str(uuid.uuid4())
        interview_id = str(uuid.uuid4())
        mock_events = [create_sentence_created_envelope(aggregate_id, interview_id)]
        mock_store.read_stream.return_value = mock_events

        sentence = await repo.load(aggregate_id)

        assert sentence is not None
        assert sentence.aggregate_id == aggregate_id
        assert sentence.interview_id == interview_id
        assert sentence.text == "Test sentence."
        mock_store.read_stream.assert_called_once_with(f"Sentence-{aggregate_id}")

    async def test_save_sentence_aggregate(self):
        """Test saving a sentence aggregate."""
        mock_store = AsyncMock()
        mock_store.append_events.return_value = 0
        repo = SentenceRepository(mock_store)

        aggregate_id = str(uuid.uuid4())
        interview_id = str(uuid.uuid4())
        sentence = Sentence(aggregate_id)
        sentence.create(interview_id=interview_id, index=0, text="New sentence.")

        await repo.save(sentence)

        mock_store.append_events.assert_called_once()
        call_args = mock_store.append_events.call_args
        assert call_args.kwargs["stream_name"] == f"Sentence-{aggregate_id}"
