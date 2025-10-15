"""
Lane manager for partitioned event processing.

Events are partitioned by interview_id into N lanes, ensuring in-order
processing within each lane while allowing parallel processing across lanes.
"""

import asyncio
import hashlib
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

from src.events.envelope import EventEnvelope

from .config import LANE_COUNT, QUEUE_DEPTH_ALERT_THRESHOLD

logger = logging.getLogger(__name__)


class Lane:
    """
    A single processing lane for events.

    Each lane maintains its own queue and processes events sequentially
    to ensure ordering guarantees for events from the same interview.
    """

    def __init__(self, lane_id: int, handler_registry):
        """
        Initialize a lane.

        Args:
            lane_id: Unique identifier for this lane
            handler_registry: Registry for looking up event handlers
        """
        self.lane_id = lane_id
        self.handler_registry = handler_registry
        self.queue: asyncio.Queue = asyncio.Queue()
        self.is_running = False
        self.last_processed_at: Optional[datetime] = None
        self.events_processed = 0
        self.events_failed = 0
        self._task: Optional[asyncio.Task] = None

    async def start(self):
        """Start the lane processing task."""
        if self.is_running:
            logger.warning(f"Lane {self.lane_id} is already running")
            return

        self.is_running = True
        self._task = asyncio.create_task(self._process_loop())
        logger.info(f"Lane {self.lane_id} started")

    async def stop(self):
        """Stop the lane processing task."""
        if not self.is_running:
            return

        self.is_running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info(f"Lane {self.lane_id} stopped")

    async def enqueue(self, event: EventEnvelope, checkpoint_callback):
        """
        Add an event to this lane's queue.

        Args:
            event: Event to process
            checkpoint_callback: Callback to invoke after successful processing
        """
        queue_depth = self.queue.qsize()
        if queue_depth > QUEUE_DEPTH_ALERT_THRESHOLD:
            logger.warning(
                f"Lane {self.lane_id} queue depth ({queue_depth}) exceeds threshold "
                f"({QUEUE_DEPTH_ALERT_THRESHOLD})"
            )

        await self.queue.put((event, checkpoint_callback))

    async def _process_loop(self):
        """Main processing loop for this lane."""
        logger.info(f"Lane {self.lane_id} processing loop started")

        while self.is_running:
            try:
                # Get next event from queue (with timeout to allow graceful shutdown)
                try:
                    event, checkpoint_callback = await asyncio.wait_for(
                        self.queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue

                # Process the event
                await self._process_event(event, checkpoint_callback)

            except asyncio.CancelledError:
                logger.info(f"Lane {self.lane_id} processing loop cancelled")
                break
            except Exception as e:
                logger.error(
                    f"Unexpected error in lane {self.lane_id} processing loop: {e}",
                    exc_info=True
                )
                # Continue processing despite error
                await asyncio.sleep(1.0)

        logger.info(f"Lane {self.lane_id} processing loop stopped")

    async def _process_event(self, event: EventEnvelope, checkpoint_callback):
        """
        Process a single event.

        Args:
            event: Event to process
            checkpoint_callback: Callback to invoke after successful processing
        """
        try:
            # Get handler for this event type
            handler = self.handler_registry.get_handler(event.event_type)
            if handler is None:
                logger.warning(
                    f"No handler found for event type {event.event_type}, skipping"
                )
                await checkpoint_callback()
                return

            # Process event with retry logic (handler includes retry-to-park)
            await handler.handle_with_retry(event, self.lane_id)

            # Update metrics
            self.events_processed += 1
            self.last_processed_at = datetime.now(timezone.utc)

            # Checkpoint after successful processing
            await checkpoint_callback()

            logger.debug(
                f"Lane {self.lane_id} processed event {event.event_id} "
                f"(type: {event.event_type}, version: {event.version})"
            )

        except Exception as e:
            # Event was parked or failed permanently
            self.events_failed += 1
            logger.error(
                f"Lane {self.lane_id} failed to process event {event.event_id}: {e}",
                exc_info=True
            )
            # Still checkpoint to move past this event
            await checkpoint_callback()

    def get_status(self) -> Dict:
        """Get current status of this lane."""
        return {
            "id": self.lane_id,
            "is_running": self.is_running,
            "queue_depth": self.queue.qsize(),
            "events_processed": self.events_processed,
            "events_failed": self.events_failed,
            "last_processed_at": self.last_processed_at.isoformat() if self.last_processed_at else None,
        }


class LaneManager:
    """
    Manages multiple processing lanes for partitioned event processing.

    Routes events to lanes based on interview_id hash to ensure
    in-order processing per interview while allowing parallelism.
    """

    def __init__(self, handler_registry, lane_count: int = LANE_COUNT):
        """
        Initialize the lane manager.

        Args:
            handler_registry: Registry for looking up event handlers
            lane_count: Number of lanes to create
        """
        self.lane_count = lane_count
        self.handler_registry = handler_registry
        self.lanes: List[Lane] = [
            Lane(lane_id=i, handler_registry=handler_registry)
            for i in range(lane_count)
        ]
        logger.info(f"LaneManager initialized with {lane_count} lanes")

    async def start(self):
        """Start all lanes."""
        logger.info("Starting all lanes...")
        for lane in self.lanes:
            await lane.start()
        logger.info(f"All {self.lane_count} lanes started")

    async def stop(self):
        """Stop all lanes."""
        logger.info("Stopping all lanes...")
        for lane in self.lanes:
            await lane.stop()
        logger.info("All lanes stopped")

    def get_lane_for_interview(self, interview_id: str) -> Lane:
        """
        Get the lane that should process events for a given interview.

        Uses consistent hashing to ensure all events for the same interview
        go to the same lane.

        Args:
            interview_id: Interview identifier

        Returns:
            Lane: The lane assigned to this interview
        """
        # Hash the interview_id and mod by lane count
        hash_value = int(hashlib.md5(interview_id.encode()).hexdigest(), 16)
        lane_index = hash_value % self.lane_count
        return self.lanes[lane_index]

    async def route_event(self, event: EventEnvelope, checkpoint_callback):
        """
        Route an event to the appropriate lane.

        Args:
            event: Event to route
            checkpoint_callback: Callback to invoke after successful processing
        """
        # Extract interview_id from event data
        interview_id = self._extract_interview_id(event)
        if not interview_id:
            logger.error(
                f"Could not extract interview_id from event {event.event_id}, "
                f"type: {event.event_type}"
            )
            # Checkpoint anyway to avoid blocking
            await checkpoint_callback()
            return

        # Get the lane for this interview
        lane = self.get_lane_for_interview(interview_id)

        # Enqueue the event
        await lane.enqueue(event, checkpoint_callback)

    def _extract_interview_id(self, event: EventEnvelope) -> Optional[str]:
        """
        Extract interview_id from event data.

        Args:
            event: Event to extract from

        Returns:
            str: Interview ID, or None if not found
        """
        # For Interview events, the aggregate_id is the interview_id
        if event.aggregate_type == "Interview":
            return event.aggregate_id

        # For Sentence events, interview_id is in the data
        if event.aggregate_type == "Sentence":
            return event.data.get("interview_id")

        logger.warning(
            f"Unknown aggregate type {event.aggregate_type} for event {event.event_id}"
        )
        return None

    def get_status(self) -> Dict:
        """Get status of all lanes."""
        return {
            "lane_count": self.lane_count,
            "lanes": [lane.get_status() for lane in self.lanes],
            "total_events_processed": sum(lane.events_processed for lane in self.lanes),
            "total_events_failed": sum(lane.events_failed for lane in self.lanes),
        }
