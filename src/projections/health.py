"""
Health check endpoint for projection service.

Provides detailed status of lanes, subscriptions, and parked events.
"""

import logging
from datetime import datetime, timezone
from typing import Dict

from .projection_service import get_projection_service

logger = logging.getLogger(__name__)


async def get_health_status() -> Dict:
    """
    Get comprehensive health status of the projection service.

    Returns:
        Dict: Health status including all components
    """
    try:
        service = get_projection_service()
        return service.get_health_status()
    except Exception as e:
        logger.error(f"Failed to get health status: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


def format_health_status(status: Dict) -> str:
    """
    Format health status as a human-readable string.

    Args:
        status: Health status dictionary

    Returns:
        str: Formatted status
    """
    lines = []
    lines.append(f"Status: {status.get('status', 'unknown').upper()}")
    lines.append(f"Running: {status.get('is_running', False)}")

    if status.get('started_at'):
        lines.append(f"Started: {status['started_at']}")

    if status.get('uptime_seconds'):
        uptime = status['uptime_seconds']
        hours = int(uptime // 3600)
        minutes = int((uptime % 3600) // 60)
        seconds = int(uptime % 60)
        lines.append(f"Uptime: {hours}h {minutes}m {seconds}s")

    # Lane status
    lanes = status.get('lanes', {})
    if lanes:
        lines.append(f"\nLanes: {lanes.get('lane_count', 0)}")
        lines.append(f"  Total Events Processed: {lanes.get('total_events_processed', 0)}")
        lines.append(f"  Total Events Failed: {lanes.get('total_events_failed', 0)}")

        # Show individual lane status
        for lane in lanes.get('lanes', []):
            lines.append(
                f"  Lane {lane['id']}: "
                f"Queue={lane['queue_depth']}, "
                f"Processed={lane['events_processed']}, "
                f"Failed={lane['events_failed']}"
            )

    # Subscription status
    subscriptions = status.get('subscriptions', {})
    if subscriptions:
        lines.append(f"\nSubscriptions: {subscriptions.get('subscription_count', 0)}")
        for name, sub_status in subscriptions.get('subscriptions', {}).items():
            running = "Running" if sub_status.get('running') else "Stopped"
            lines.append(f"  {name}: {running}")

    # Parked events
    parked = status.get('parked_events', {})
    if parked:
        lines.append("\nParked Events:")
        for agg_type, count in parked.items():
            lines.append(f"  {agg_type}: {count}")

    return "\n".join(lines)
