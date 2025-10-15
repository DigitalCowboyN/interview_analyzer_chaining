"""
Main projection service orchestrator.

Coordinates lane manager, subscription manager, and handler registry
to maintain Neo4j projections from EventStoreDB events.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Optional

from src.events.store import EventStoreClient, get_event_store_client

from .config import ENABLE_PROJECTION_SERVICE
from .lane_manager import LaneManager
from .parked_events import ParkedEventsManager
from .subscription_manager import SubscriptionManager

logger = logging.getLogger(__name__)


class ProjectionService:
    """
    Main projection service that orchestrates event consumption and Neo4j updates.

    Manages the lifecycle of subscriptions, lanes, and handlers to ensure
    reliable, ordered processing of events.
    """

    def __init__(
        self,
        event_store: Optional[EventStoreClient] = None,
        handler_registry=None,
        lane_count: Optional[int] = None,
    ):
        """
        Initialize the projection service.

        Args:
            event_store: EventStore client (uses global if not provided)
            handler_registry: Handler registry for event processing
            lane_count: Number of processing lanes (uses config default if not provided)
        """
        self.event_store = event_store or get_event_store_client()
        self.handler_registry = handler_registry
        self.parked_events_manager = ParkedEventsManager(self.event_store)

        # Initialize lane manager
        self.lane_manager = LaneManager(
            handler_registry=self.handler_registry,
            lane_count=lane_count if lane_count is not None else 12,
        )

        # Initialize subscription manager
        self.subscription_manager = SubscriptionManager(
            event_store=self.event_store,
            lane_manager=self.lane_manager,
        )

        self.is_running = False
        self.started_at: Optional[datetime] = None

    async def start(self):
        """Start the projection service."""
        if not ENABLE_PROJECTION_SERVICE:
            logger.warning("Projection service is disabled via configuration")
            return

        if self.is_running:
            logger.warning("Projection service is already running")
            return

        logger.info("Starting projection service...")
        self.is_running = True
        self.started_at = datetime.now(timezone.utc)

        try:
            # Start lanes first
            await self.lane_manager.start()

            # Then start subscriptions (which will route to lanes)
            await self.subscription_manager.start()

            logger.info("Projection service started successfully")

        except Exception as e:
            logger.error(f"Failed to start projection service: {e}", exc_info=True)
            await self.stop()
            raise

    async def stop(self):
        """Stop the projection service."""
        if not self.is_running:
            return

        logger.info("Stopping projection service...")
        self.is_running = False

        try:
            # Stop subscriptions first (stop receiving new events)
            await self.subscription_manager.stop()

            # Then stop lanes (finish processing queued events)
            await self.lane_manager.stop()

            logger.info("Projection service stopped successfully")

        except Exception as e:
            logger.error(f"Error stopping projection service: {e}", exc_info=True)
            raise

    async def run(self):
        """
        Run the projection service until interrupted.

        This is a convenience method for running the service as a standalone process.
        """
        await self.start()

        try:
            # Run until interrupted
            while self.is_running:
                await asyncio.sleep(1.0)
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        finally:
            await self.stop()

    def get_health_status(self) -> Dict:
        """
        Get health status of the projection service.

        Returns:
            Dict: Health status including lane and subscription information
        """
        lane_status = self.lane_manager.get_status()
        subscription_status = self.subscription_manager.get_status()

        # Calculate uptime
        uptime_seconds = None
        if self.started_at:
            uptime_seconds = (datetime.now(timezone.utc) - self.started_at).total_seconds()

        # Determine overall health
        all_lanes_running = all(
            lane["is_running"] for lane in lane_status["lanes"]
        )
        all_subscriptions_running = all(
            sub["running"] for sub in subscription_status["subscriptions"].values()
        )

        is_healthy = (
            self.is_running
            and all_lanes_running
            and all_subscriptions_running
        )

        return {
            "status": "healthy" if is_healthy else "unhealthy",
            "is_running": self.is_running,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "uptime_seconds": uptime_seconds,
            "lanes": lane_status,
            "subscriptions": subscription_status,
            "parked_events": {
                "interview": 0,  # TODO: Implement count retrieval
                "sentence": 0,
            },
        }


# Global projection service instance
_projection_service: Optional[ProjectionService] = None


def get_projection_service(
    handler_registry=None,
    lane_count: Optional[int] = None,
) -> ProjectionService:
    """
    Get the global projection service instance.

    Args:
        handler_registry: Handler registry (only used on first call)
        lane_count: Number of lanes (only used on first call)

    Returns:
        ProjectionService: The global instance
    """
    global _projection_service
    if _projection_service is None:
        _projection_service = ProjectionService(
            handler_registry=handler_registry,
            lane_count=lane_count,
        )
    return _projection_service
