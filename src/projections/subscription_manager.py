"""
Subscription manager for EventStoreDB persistent subscriptions.

Manages persistent subscriptions to category streams, ensuring reliable
event delivery and automatic checkpoint management.
"""

import asyncio
import logging
from typing import Callable, Dict, Optional

from esdbclient import EventStoreDBClient
from esdbclient.exceptions import NotFound

from src.events.envelope import EventEnvelope
from src.events.store import EventStoreClient, get_event_store_client

from .config import SUBSCRIPTION_CONFIG, is_event_allowed

logger = logging.getLogger(__name__)


class SubscriptionManager:
    """
    Manages persistent subscriptions to EventStoreDB.

    Creates and maintains subscriptions to category streams, filtering
    events by type and routing them to the lane manager.
    """

    def __init__(
        self,
        event_store: Optional[EventStoreClient] = None,
        lane_manager=None,
    ):
        """
        Initialize the subscription manager.

        Args:
            event_store: EventStore client (uses global if not provided)
            lane_manager: Lane manager for routing events
        """
        self.event_store = event_store or get_event_store_client()
        self.lane_manager = lane_manager
        self.subscriptions: Dict[str, asyncio.Task] = {}
        self.is_running = False

    async def start(self):
        """Start all subscriptions."""
        if self.is_running:
            logger.warning("SubscriptionManager is already running")
            return

        self.is_running = True
        logger.info("Starting subscriptions...")

        for sub_name, config in SUBSCRIPTION_CONFIG.items():
            task = asyncio.create_task(
                self._run_subscription(sub_name, config)
            )
            self.subscriptions[sub_name] = task
            logger.info(
                f"Started subscription '{sub_name}' to stream '{config['stream']}' "
                f"with group '{config['group']}'"
            )

        logger.info(f"All {len(self.subscriptions)} subscriptions started")

    async def stop(self):
        """Stop all subscriptions."""
        if not self.is_running:
            return

        self.is_running = False
        logger.info("Stopping subscriptions...")

        for sub_name, task in self.subscriptions.items():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            logger.info(f"Stopped subscription '{sub_name}'")

        self.subscriptions.clear()
        logger.info("All subscriptions stopped")

    async def _run_subscription(self, sub_name: str, config: Dict):
        """
        Run a single persistent subscription.

        Args:
            sub_name: Name of the subscription
            config: Subscription configuration
        """
        stream_name = config["stream"]
        group_name = config["group"]

        while self.is_running:
            try:
                logger.info(
                    f"Connecting to subscription '{sub_name}' "
                    f"(stream: {stream_name}, group: {group_name})"
                )

                # Ensure subscription exists (create if not)
                await self._ensure_subscription_exists(stream_name, group_name)

                # Read from persistent subscription
                async with self.event_store.get_client() as client:
                    subscription = client.read_subscription_to_stream(
                        group_name=group_name,
                        stream_name=stream_name,
                    )

                    for event in subscription:
                        if not self.is_running:
                            break

                        # Convert to EventEnvelope
                        envelope = self.event_store._recorded_event_to_envelope(event)

                        # Filter by allowlist
                        if not is_event_allowed(sub_name, envelope.event_type):
                            logger.debug(
                                f"Skipping event {envelope.event_id} "
                                f"(type: {envelope.event_type}) - not in allowlist"
                            )
                            subscription.ack(event.id)
                            continue

                        # Route to lane manager
                        if self.lane_manager:
                            await self.lane_manager.route_event(
                                envelope,
                                checkpoint_callback=lambda: subscription.ack(event.id)
                            )
                        else:
                            # No lane manager, just ack
                            subscription.ack(event.id)

            except asyncio.CancelledError:
                logger.info(f"Subscription '{sub_name}' cancelled")
                break
            except Exception as e:
                logger.error(
                    f"Error in subscription '{sub_name}': {e}",
                    exc_info=True
                )
                if self.is_running:
                    logger.info(f"Reconnecting subscription '{sub_name}' in 5 seconds...")
                    await asyncio.sleep(5.0)
                else:
                    break

    async def _ensure_subscription_exists(self, stream_name: str, group_name: str):
        """
        Ensure a persistent subscription exists, creating it if necessary.

        Args:
            stream_name: Stream to subscribe to
            group_name: Consumer group name
        """
        try:
            async with self.event_store.get_client() as client:
                # Try to get subscription info (this will raise if it doesn't exist)
                try:
                    client.get_subscription_info(
                        group_name=group_name,
                        stream_name=stream_name,
                    )
                    logger.debug(
                        f"Subscription '{group_name}' to '{stream_name}' already exists"
                    )
                except NotFound:
                    # Subscription doesn't exist, create it
                    logger.info(
                        f"Creating subscription '{group_name}' to '{stream_name}'"
                    )
                    client.create_subscription_to_stream(
                        group_name=group_name,
                        stream_name=stream_name,
                        from_end=False,  # Start from beginning
                    )
                    logger.info(
                        f"Created subscription '{group_name}' to '{stream_name}'"
                    )
        except Exception as e:
            logger.error(
                f"Failed to ensure subscription '{group_name}' exists: {e}",
                exc_info=True
            )
            raise

    def get_status(self) -> Dict:
        """Get status of all subscriptions."""
        return {
            "is_running": self.is_running,
            "subscription_count": len(self.subscriptions),
            "subscriptions": {
                name: {
                    "running": not task.done(),
                    "cancelled": task.cancelled() if task.done() else False,
                }
                for name, task in self.subscriptions.items()
            },
        }
