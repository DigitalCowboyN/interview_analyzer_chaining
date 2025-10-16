"""
Handler registry for routing events to appropriate handlers.

Maintains a mapping of event types to handler instances.
"""

import logging
from typing import Dict, Optional

from src.events.envelope import EventEnvelope

from .base_handler import BaseProjectionHandler

logger = logging.getLogger(__name__)


class HandlerRegistry:
    """
    Registry for event handlers.

    Maps event types to handler instances for routing.
    """

    def __init__(self):
        """Initialize the registry."""
        self._handlers: Dict[str, BaseProjectionHandler] = {}

    def register(self, event_type: str, handler: BaseProjectionHandler):
        """
        Register a handler for an event type.

        Args:
            event_type: Type of event to handle
            handler: Handler instance
        """
        if event_type in self._handlers:
            logger.warning(f"Overwriting existing handler for event type '{event_type}'")

        self._handlers[event_type] = handler
        logger.debug(f"Registered handler for event type '{event_type}'")

    def get_handler(self, event_type: str) -> Optional[BaseProjectionHandler]:
        """
        Get the handler for an event type.

        Args:
            event_type: Type of event

        Returns:
            BaseProjectionHandler: Handler instance, or None if not found
        """
        return self._handlers.get(event_type)

    def has_handler(self, event_type: str) -> bool:
        """
        Check if a handler is registered for an event type.

        Args:
            event_type: Type of event

        Returns:
            bool: True if handler exists
        """
        return event_type in self._handlers

    def get_registered_types(self) -> list:
        """Get all registered event types."""
        return list(self._handlers.keys())


# Global registry instance
_registry: Optional[HandlerRegistry] = None


def get_handler_registry() -> HandlerRegistry:
    """Get the global handler registry instance."""
    global _registry
    if _registry is None:
        _registry = HandlerRegistry()
    return _registry
