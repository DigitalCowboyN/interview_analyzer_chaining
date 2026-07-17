"""
Subscription manager for EventStoreDB persistent subscriptions.

Manages persistent subscriptions to category streams, ensuring reliable
event delivery and automatic checkpoint management.
"""

import asyncio
import logging
from typing import Any, Callable, Dict, Optional

from esdbclient import EventStoreDBClient
from esdbclient.exceptions import NotFound

from src.events.envelope import EventEnvelope
from src.events.store import EventStoreClient, get_event_store_client

from .config import SUBSCRIPTION_CONFIG, is_event_allowed

logger = logging.getLogger(__name__)

# Sentinel returned by asyncio.to_thread(next, iterator, _SUBSCRIPTION_ENDED)
# when the underlying (blocking) subscription iterator is exhausted, so we
# can distinguish "no more events" from a real event without next() raising.
_SUBSCRIPTION_ENDED = object()


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
        # Active esdbclient PersistentSubscription object per subscription
        # name, for as long as that subscription's outer loop iteration is
        # connected. Held so stop() can call .stop() on it directly -- see
        # stop()'s docstring for why this is required for a clean shutdown.
        self._active_subscriptions: Dict[str, Any] = {}
        self.is_running = False

    async def start(self):
        """Start all subscriptions."""
        if self.is_running:
            logger.warning("SubscriptionManager is already running")
            return

        self.is_running = True
        logger.info("Starting subscriptions...")

        for sub_name, config in SUBSCRIPTION_CONFIG.items():
            task = asyncio.create_task(self._run_subscription(sub_name, config))
            self.subscriptions[sub_name] = task
            logger.info(
                f"Started subscription '{sub_name}' to stream '{config['stream']}' " f"with group '{config['group']}'"
            )

        logger.info(f"All {len(self.subscriptions)} subscriptions started")

    async def stop(self):
        """
        Stop all subscriptions.

        Each subscription task spends most of its life blocked inside
        `asyncio.to_thread(next, iterator, ...)`, waiting on the esdbclient
        PersistentSubscription's blocking network read. Cancelling the
        asyncio.Task alone does not unblock that worker thread -- the
        cancellation is only delivered the next time the coroutine resumes
        on the event loop, which won't happen until the blocking next()
        call itself returns. So before cancelling/awaiting each task, call
        `.stop()` on its active esdbclient subscription object directly:
        that cancels the subscription's underlying gRPC response stream,
        which unblocks the worker thread's next() call promptly instead of
        leaving it hung until the network call times out on its own.
        """
        if not self.is_running:
            return

        self.is_running = False
        logger.info("Stopping subscriptions...")

        for sub_name in self.subscriptions:
            subscription = self._active_subscriptions.get(sub_name)
            if subscription is not None:
                subscription.stop()

        for sub_name, task in self.subscriptions.items():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            logger.info(f"Stopped subscription '{sub_name}'")

        self.subscriptions.clear()
        self._active_subscriptions.clear()
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
                logger.info(f"Connecting to subscription '{sub_name}' " f"(stream: {stream_name}, group: {group_name})")

                # Ensure subscription exists (create if not)
                await self._ensure_subscription_exists(stream_name, group_name)

                # Read from persistent subscription
                async with self.event_store.get_client() as client:
                    subscription = client.read_subscription_to_stream(
                        group_name=group_name,
                        stream_name=stream_name,
                    )
                    # Keep a reference so stop() (and the reconnect path
                    # below, on the next exception) can call .stop() on
                    # this exact object to unblock a worker thread that's
                    # parked in its blocking next() call.
                    self._active_subscriptions[sub_name] = subscription

                    # `subscription` is a synchronous (blocking) esdbclient
                    # generator. Iterating it directly with `for event in
                    # subscription:` inside this coroutine would block the
                    # event loop indefinitely on network I/O, starving the
                    # other subscription tasks (they'd never get a turn to
                    # run) -- this was the defect that kept the deployed
                    # service from ever creating the 'sentence'/'project'
                    # persistent subscriptions. Offload each blocking next()
                    # call to a worker thread via asyncio.to_thread so the
                    # loop stays free; one worker thread is held per
                    # subscription for as long as it's waiting on the next
                    # event, which is fine for our fixed, small subscription
                    # count.
                    iterator = iter(subscription)
                    while self.is_running:
                        event = await asyncio.to_thread(next, iterator, _SUBSCRIPTION_ENDED)
                        if event is _SUBSCRIPTION_ENDED:
                            break

                        # Convert to EventEnvelope
                        envelope = self.event_store._recorded_event_to_envelope(event)

                        # Filter by allowlist
                        if not is_event_allowed(sub_name, envelope.event_type):
                            logger.debug(
                                f"Skipping event {envelope.event_id} "
                                f"(type: {envelope.event_type}) - not in allowlist"
                            )
                            # Ack by the LINK's id (event.ack_id), not the
                            # resolved event's id (event.id). On $ce- category
                            # streams read with resolve_links=True, the server
                            # tracks in-flight messages by the link's id
                            # (esdbclient RecordedEvent.ack_id: link.id if
                            # resolved, else id); acking a bare event.id is a
                            # no-op ack for a different (nonexistent) message
                            # id, so the real in-flight message is never
                            # cleared server-side and gets redelivered until
                            # parked.
                            subscription.ack(event.ack_id)
                            continue

                        # Route to lane manager
                        if self.lane_manager:
                            # Bind event.ack_id as a default arg so the
                            # callback captures *this* event's ack id at
                            # creation time. LaneManager.route_event enqueues
                            # (event, checkpoint_callback) and invokes the
                            # callback later, from the lane's processing loop
                            # -- a plain `lambda: subscription.ack(event.ack_id)`
                            # would close over the loop variable `event` by
                            # reference, so by the time the callback ran it
                            # could have already been rebound to a later
                            # iteration's event, acking the wrong id.
                            await self.lane_manager.route_event(
                                envelope,
                                checkpoint_callback=lambda ack_id=event.ack_id: subscription.ack(ack_id),
                            )
                        else:
                            # No lane manager, just ack (by link id, see above)
                            subscription.ack(event.ack_id)

            except asyncio.CancelledError:
                logger.info(f"Subscription '{sub_name}' cancelled")
                break
            except Exception as e:
                logger.error(f"Error in subscription '{sub_name}': {e}", exc_info=True)
                if self.is_running:
                    logger.info(f"Reconnecting subscription '{sub_name}' in 5 seconds...")
                    await asyncio.sleep(5.0)
                else:
                    break
            finally:
                # This subscription's outer loop iteration has ended (clean
                # exhaustion, cancellation, or an exception handled above) --
                # stop the subscription object (if still active) so its
                # underlying gRPC response stream doesn't leak into the next
                # reconnect attempt, then drop the reference so stop()
                # doesn't try to re-stop a subscription that's no longer
                # active for this iteration.
                stale_subscription = self._active_subscriptions.pop(sub_name, None)
                if stale_subscription is not None:
                    stale_subscription.stop()

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
                    logger.debug(f"Subscription '{group_name}' to '{stream_name}' already exists")
                except NotFound:
                    # Subscription doesn't exist, create it
                    logger.info(f"Creating subscription '{group_name}' to '{stream_name}'")
                    client.create_subscription_to_stream(
                        group_name=group_name,
                        stream_name=stream_name,
                        from_end=False,  # Start from beginning
                        resolve_links=True,  # $ce- streams deliver $> links; resolve to real events
                    )
                    logger.info(f"Created subscription '{group_name}' to '{stream_name}'")
        except Exception as e:
            logger.error(f"Failed to ensure subscription '{group_name}' exists: {e}", exc_info=True)
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
