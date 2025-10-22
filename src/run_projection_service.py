"""
Standalone runner for the projection service.

Starts the projection service to consume events from EventStoreDB
and update Neo4j projections.
"""

import argparse
import asyncio
import logging
import signal
import sys

from src.events.store import get_event_store_client
from src.projections.bootstrap import create_handler_registry
from src.projections.config import ESDB_CONNECTION_STRING
from src.projections.projection_service import ProjectionService

logger = logging.getLogger(__name__)


async def main(lane_count: int = 12, log_level: str = "INFO"):
    """
    Main entry point for projection service.

    Args:
        lane_count: Number of processing lanes
        log_level: Logging level
    """
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s | %(levelname)-8s | %(name)s:%(filename)s:%(lineno)d | %(message)s',
        datefmt='%H:%M:%S'
    )

    logger.info("=" * 60)
    logger.info("Starting Interview Analyzer Projection Service")
    logger.info("=" * 60)

    # Create handler registry
    logger.info("Initializing handler registry...")
    handler_registry = create_handler_registry()

    # Create EventStore client with connection string from config
    logger.info(f"Connecting to EventStore at {ESDB_CONNECTION_STRING}...")
    event_store = get_event_store_client(connection_string=ESDB_CONNECTION_STRING)

    # Create projection service
    logger.info(f"Creating projection service with {lane_count} lanes...")
    service = ProjectionService(
        event_store=event_store,
        handler_registry=handler_registry,
        lane_count=lane_count,
    )

    # Setup signal handlers for graceful shutdown
    shutdown_event = asyncio.Event()

    def signal_handler(sig, frame):
        logger.info(f"Received signal {sig}, initiating graceful shutdown...")
        shutdown_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Start the service
        await service.start()

        logger.info("Projection service is running. Press Ctrl+C to stop.")

        # Wait for shutdown signal
        await shutdown_event.wait()

    except Exception as e:
        logger.error(f"Projection service failed: {e}", exc_info=True)
        sys.exit(1)
    finally:
        # Stop the service
        logger.info("Stopping projection service...")
        await service.stop()
        logger.info("Projection service stopped cleanly")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the projection service")
    parser.add_argument(
        "--lane-count",
        type=int,
        default=12,
        help="Number of processing lanes (default: 12)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    # Run the async main function
    asyncio.run(main(lane_count=args.lane_count, log_level=args.log_level))
