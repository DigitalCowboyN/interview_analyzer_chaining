"""
Repository pattern implementation for event-sourced aggregates.

Provides high-level repository interfaces for loading and saving aggregates
with optimistic concurrency control and retry logic. Handles stream naming
conventions and aggregate reconstruction from events.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Generic, Optional, TypeVar

from .aggregates import AggregateRoot, Interview, Sentence
from .store import ConcurrencyError, EventStoreClient, StreamNotFoundError

logger = logging.getLogger(__name__)

# Type variable for aggregate types
T = TypeVar("T", bound=AggregateRoot)


class Repository(ABC, Generic[T]):
    """
    Abstract base repository for event-sourced aggregates.

    Provides common functionality for loading and saving aggregates
    with proper error handling and retry logic.
    """

    def __init__(self, event_store: EventStoreClient):
        """
        Initialize the repository.

        Args:
            event_store: EventStore client for persistence
        """
        self.event_store = event_store

    @abstractmethod
    def _create_aggregate(self, aggregate_id: str) -> T:
        """Create a new instance of the aggregate."""
        pass

    @abstractmethod
    def _get_stream_name(self, aggregate_id: str) -> str:
        """Get the stream name for the aggregate."""
        pass

    async def load(self, aggregate_id: str) -> Optional[T]:
        """
        Load an aggregate from the event store.

        Args:
            aggregate_id: UUID of the aggregate to load

        Returns:
            T: The loaded aggregate, or None if not found
        """
        stream_name = self._get_stream_name(aggregate_id)

        try:
            events = await self.event_store.read_stream(stream_name)
            if not events:
                return None

            aggregate = self._create_aggregate(aggregate_id)
            aggregate.load_from_history(events)

            logger.debug(
                f"Loaded aggregate {aggregate_id} from stream '{stream_name}' "
                f"with {len(events)} events, version: {aggregate.version}"
            )
            return aggregate

        except StreamNotFoundError:
            logger.debug(f"Stream '{stream_name}' not found for aggregate {aggregate_id}")
            return None

    async def save(
        self, aggregate: T, expected_version: Optional[int] = None, max_retries: int = 3, retry_delay: float = 0.1
    ) -> None:
        """
        Save an aggregate to the event store with optimistic concurrency control.

        Args:
            aggregate: The aggregate to save
            expected_version: Expected current version (None to use aggregate version)
            max_retries: Maximum number of retry attempts for concurrency conflicts
            retry_delay: Delay between retry attempts in seconds

        Raises:
            ConcurrencyError: If concurrency conflict cannot be resolved after retries
            ValueError: If aggregate has no uncommitted events
        """
        uncommitted_events = aggregate.get_uncommitted_events()
        if not uncommitted_events:
            logger.debug(f"No uncommitted events for aggregate {aggregate.aggregate_id}")
            return

        stream_name = self._get_stream_name(aggregate.aggregate_id)
        # For new aggregates, the stream doesn't exist yet
        # EventStoreDB requires expected_version=-1 for new streams
        if expected_version is not None:
            current_expected_version = expected_version
        else:
            # Calculate the version before uncommitted events were added
            # If version before was -1, this is a new stream
            version_before_uncommitted = aggregate.version - len(uncommitted_events)
            if version_before_uncommitted < 0:
                # New stream - use -1 to indicate stream doesn't exist
                current_expected_version = -1
            else:
                # Existing stream - use current version minus uncommitted events
                current_expected_version = version_before_uncommitted

        for attempt in range(max_retries + 1):
            try:
                new_version = await self.event_store.append_events(
                    stream_name=stream_name, events=uncommitted_events, expected_version=current_expected_version
                )

                # Mark events as committed and update aggregate version
                aggregate.mark_events_as_committed()
                aggregate.version = new_version

                logger.debug(
                    f"Saved aggregate {aggregate.aggregate_id} to stream '{stream_name}' "
                    f"with {len(uncommitted_events)} events, new version: {new_version}"
                )
                return

            except ConcurrencyError as e:
                if attempt < max_retries:
                    logger.warning(
                        f"Concurrency conflict on attempt {attempt + 1} for aggregate "
                        f"{aggregate.aggregate_id}: {e}. Retrying with reload..."
                    )

                    # Reload the aggregate to get the latest state
                    latest_aggregate = await self.load(aggregate.aggregate_id)
                    if latest_aggregate is None:
                        # Aggregate was deleted, treat as conflict
                        raise ConcurrencyError(
                            f"Aggregate {aggregate.aggregate_id} was deleted during save", current_expected_version, -1
                        )

                    # Update expected version for retry
                    current_expected_version = latest_aggregate.version

                    # Apply business logic to determine if retry is valid
                    # For now, we'll just retry with the new expected version
                    # In a more sophisticated system, you might want to:
                    # 1. Merge changes if possible
                    # 2. Validate that the retry is still valid
                    # 3. Re-execute business logic with updated state

                    await asyncio.sleep(retry_delay)
                    continue
                else:
                    logger.error(
                        f"Failed to save aggregate {aggregate.aggregate_id} after "
                        f"{max_retries} retries due to concurrency conflicts"
                    )
                    raise

        raise ConcurrencyError(
            f"Failed to save aggregate {aggregate.aggregate_id} after {max_retries} retries",
            current_expected_version,
            -1,
        )

    async def exists(self, aggregate_id: str) -> bool:
        """
        Check if an aggregate exists in the event store.

        Args:
            aggregate_id: UUID of the aggregate to check

        Returns:
            bool: True if the aggregate exists, False otherwise
        """
        stream_name = self._get_stream_name(aggregate_id)
        version = await self.event_store.get_stream_version(stream_name)
        return version is not None

    async def get_version(self, aggregate_id: str) -> Optional[int]:
        """
        Get the current version of an aggregate.

        Args:
            aggregate_id: UUID of the aggregate

        Returns:
            int: Current version of the aggregate, or None if not found
        """
        stream_name = self._get_stream_name(aggregate_id)
        return await self.event_store.get_stream_version(stream_name)


class InterviewRepository(Repository[Interview]):
    """Repository for Interview aggregates."""

    def _create_aggregate(self, aggregate_id: str) -> Interview:
        """Create a new Interview instance."""
        return Interview(aggregate_id)

    def _get_stream_name(self, aggregate_id: str) -> str:
        """Get the stream name for an Interview aggregate."""
        return f"Interview-{aggregate_id}"


class SentenceRepository(Repository[Sentence]):
    """Repository for Sentence aggregates."""

    def _create_aggregate(self, aggregate_id: str) -> Sentence:
        """Create a new Sentence instance."""
        return Sentence(aggregate_id)

    def _get_stream_name(self, aggregate_id: str) -> str:
        """Get the stream name for a Sentence aggregate."""
        return f"Sentence-{aggregate_id}"


class RepositoryFactory:
    """
    Factory for creating repository instances.

    Provides a centralized way to create repositories with proper
    EventStore client injection and configuration.
    """

    def __init__(self, event_store: EventStoreClient):
        """
        Initialize the repository factory.

        Args:
            event_store: EventStore client for all repositories
        """
        self.event_store = event_store

    def create_interview_repository(self) -> InterviewRepository:
        """
        Create an InterviewRepository instance.

        Returns:
            InterviewRepository: Configured repository instance
        """
        return InterviewRepository(self.event_store)

    def create_sentence_repository(self) -> SentenceRepository:
        """
        Create a SentenceRepository instance.

        Returns:
            SentenceRepository: Configured repository instance
        """
        return SentenceRepository(self.event_store)


# Global repository factory instance
_global_factory: Optional[RepositoryFactory] = None


def get_repository_factory(event_store: Optional[EventStoreClient] = None) -> RepositoryFactory:
    """
    Get the global repository factory instance.

    Args:
        event_store: EventStore client (only used on first call)

    Returns:
        RepositoryFactory: Global factory instance
    """
    global _global_factory

    if _global_factory is None:
        if event_store is None:
            from .store import get_event_store_client

            event_store = get_event_store_client()

        _global_factory = RepositoryFactory(event_store)

    return _global_factory


def get_interview_repository() -> InterviewRepository:
    """
    Get an InterviewRepository instance using the global factory.

    Returns:
        InterviewRepository: Configured repository instance
    """
    factory = get_repository_factory()
    return factory.create_interview_repository()


def get_sentence_repository() -> SentenceRepository:
    """
    Get a SentenceRepository instance using the global factory.

    Returns:
        SentenceRepository: Configured repository instance
    """
    factory = get_repository_factory()
    return factory.create_sentence_repository()
