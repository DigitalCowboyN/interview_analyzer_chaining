"""
Base handler with version checking and retry logic.

All projection handlers inherit from this base to ensure idempotency
and consistent error handling.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Optional

from src.events.envelope import EventEnvelope
from src.utils.neo4j_driver import Neo4jConnectionManager

from ..config import (
    RETRY_EXPONENTIAL_BASE,
    RETRY_INITIAL_DELAY,
    RETRY_MAX_ATTEMPTS,
    RETRY_MAX_DELAY,
)
from ..parked_events import ParkedEventsManager

logger = logging.getLogger(__name__)


class BaseProjectionHandler(ABC):
    """
    Base class for all projection handlers.

    Provides version checking, retry logic, and Neo4j session management.
    """

    def __init__(self, parked_events_manager: Optional[ParkedEventsManager] = None):
        """
        Initialize the handler.

        Args:
            parked_events_manager: Manager for parking failed events
        """
        self.neo4j_manager = Neo4jConnectionManager()
        self.parked_events_manager = parked_events_manager or ParkedEventsManager()

    async def handle_with_retry(self, event: EventEnvelope, lane_id: int):
        """
        Handle an event with retry logic.

        Args:
            event: Event to handle
            lane_id: Lane processing this event

        Raises:
            Exception: If event fails after all retries (event will be parked)
        """
        for attempt in range(RETRY_MAX_ATTEMPTS):
            try:
                await self.handle(event)
                return  # Success

            except Exception as e:
                if attempt < RETRY_MAX_ATTEMPTS - 1:
                    delay = self._calculate_backoff(attempt)
                    logger.warning(
                        f"Retry {attempt + 1}/{RETRY_MAX_ATTEMPTS} for event {event.event_id} "
                        f"after {delay}s. Error: {e}"
                    )
                    await asyncio.sleep(delay)
                else:
                    # Final attempt failed, park the event
                    logger.error(
                        f"Event {event.event_id} failed after {RETRY_MAX_ATTEMPTS} attempts. "
                        f"Parking event. Error: {e}"
                    )
                    await self.parked_events_manager.park_event(
                        event=event,
                        error=e,
                        retry_count=RETRY_MAX_ATTEMPTS,
                        lane_id=lane_id,
                    )
                    raise

    def _calculate_backoff(self, attempt: int) -> float:
        """
        Calculate exponential backoff delay.

        Args:
            attempt: Retry attempt number (0-based)

        Returns:
            float: Delay in seconds
        """
        delay = RETRY_INITIAL_DELAY * (RETRY_EXPONENTIAL_BASE ** attempt)
        return min(delay, RETRY_MAX_DELAY)

    async def handle(self, event: EventEnvelope):
        """
        Handle an event with version checking.

        Args:
            event: Event to handle
        """
        async with self.neo4j_manager.get_session() as session:
            # Check version (idempotency guard)
            current_version = await self._get_version(session, event)
            if current_version is not None and event.version <= current_version:
                logger.debug(
                    f"Skipping event {event.event_id} (version {event.version}) - "
                    f"already applied (current version: {current_version})"
                )
                return

            # Apply event in transaction
            async with session.begin_transaction() as tx:
                try:
                    await self.apply(tx, event)
                    await self._set_version(tx, event)
                    await tx.commit()

                    logger.debug(
                        f"Applied event {event.event_id} (type: {event.event_type}, "
                        f"version: {event.version}) to {event.aggregate_id}"
                    )

                except Exception as e:
                    await tx.rollback()
                    logger.error(
                        f"Failed to apply event {event.event_id}: {e}",
                        exc_info=True
                    )
                    raise

    async def _get_version(self, session, event: EventEnvelope) -> Optional[int]:
        """
        Get the current event version for an aggregate from Neo4j.

        Args:
            session: Neo4j session
            event: Event to check version for

        Returns:
            int: Current version, or None if aggregate doesn't exist
        """
        node_label = event.aggregate_type
        query = f"""
        MATCH (n:{node_label} {{aggregate_id: $aggregate_id}})
        RETURN n.event_version as version
        """

        result = await session.run(
            query,
            aggregate_id=event.aggregate_id
        )

        record = await result.single()
        if record:
            return record["version"]
        return None

    async def _set_version(self, tx, event: EventEnvelope):
        """
        Set the event version for an aggregate in Neo4j.

        Args:
            tx: Neo4j transaction
            event: Event being applied
        """
        node_label = event.aggregate_type
        query = f"""
        MATCH (n:{node_label} {{aggregate_id: $aggregate_id}})
        SET n.event_version = $version
        """

        await tx.run(
            query,
            aggregate_id=event.aggregate_id,
            version=event.version
        )

    @abstractmethod
    async def apply(self, tx, event: EventEnvelope):
        """
        Apply the event to Neo4j.

        This method must be implemented by subclasses to define
        the specific Neo4j updates for each event type.

        Args:
            tx: Neo4j transaction
            event: Event to apply
        """
        pass
