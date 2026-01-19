"""
Unit tests for the projection handler registry.

Tests the HandlerRegistry class which routes events to appropriate handlers.
"""

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.projections.handlers.registry import (
    HandlerRegistry,
    get_handler_registry,
    _registry,
)
from src.projections.handlers.base_handler import BaseProjectionHandler
from src.events.envelope import EventEnvelope


class MockProjectionHandler(BaseProjectionHandler):
    """Mock handler for testing."""

    def __init__(self):
        # Don't call super().__init__() to avoid ParkedEventsManager dependency
        self.parked_events_manager = None

    async def apply(self, tx, event: EventEnvelope):
        """Mock apply method."""
        pass


class TestHandlerRegistry:
    """Tests for HandlerRegistry class."""

    def test_registry_initializes_with_empty_handlers(self):
        """Registry should start with no handlers registered."""
        registry = HandlerRegistry()

        assert registry.get_registered_types() == []

    def test_register_handler_adds_handler_to_registry(self):
        """Registering a handler should make it retrievable."""
        registry = HandlerRegistry()
        handler = MockProjectionHandler()

        registry.register("TestEvent", handler)

        assert registry.get_handler("TestEvent") is handler

    def test_register_multiple_handlers_for_different_event_types(self):
        """Registry should support multiple handlers for different event types."""
        registry = HandlerRegistry()
        handler1 = MockProjectionHandler()
        handler2 = MockProjectionHandler()

        registry.register("EventTypeA", handler1)
        registry.register("EventTypeB", handler2)

        assert registry.get_handler("EventTypeA") is handler1
        assert registry.get_handler("EventTypeB") is handler2
        assert len(registry.get_registered_types()) == 2

    def test_register_duplicate_event_type_logs_warning(self, caplog):
        """Registering a handler for an existing event type should log a warning."""
        registry = HandlerRegistry()
        handler1 = MockProjectionHandler()
        handler2 = MockProjectionHandler()

        registry.register("TestEvent", handler1)

        with caplog.at_level(logging.WARNING):
            registry.register("TestEvent", handler2)

        assert "Overwriting existing handler" in caplog.text
        assert "TestEvent" in caplog.text

    def test_register_duplicate_event_type_overwrites_handler(self):
        """Registering for an existing event type should replace the handler."""
        registry = HandlerRegistry()
        handler1 = MockProjectionHandler()
        handler2 = MockProjectionHandler()

        registry.register("TestEvent", handler1)
        registry.register("TestEvent", handler2)

        assert registry.get_handler("TestEvent") is handler2
        assert registry.get_handler("TestEvent") is not handler1

    def test_get_handler_returns_none_for_unregistered_event_type(self):
        """Getting a handler for an unregistered event type should return None."""
        registry = HandlerRegistry()

        result = registry.get_handler("NonExistentEvent")

        assert result is None

    def test_get_handler_returns_correct_handler_among_many(self):
        """Getting a handler should return the correct one when multiple are registered."""
        registry = HandlerRegistry()
        handlers = {f"Event{i}": MockProjectionHandler() for i in range(5)}

        for event_type, handler in handlers.items():
            registry.register(event_type, handler)

        # Verify each handler is returned correctly
        for event_type, expected_handler in handlers.items():
            assert registry.get_handler(event_type) is expected_handler

    def test_has_handler_returns_true_for_registered_event_type(self):
        """has_handler should return True for registered event types."""
        registry = HandlerRegistry()
        handler = MockProjectionHandler()

        registry.register("TestEvent", handler)

        assert registry.has_handler("TestEvent") is True

    def test_has_handler_returns_false_for_unregistered_event_type(self):
        """has_handler should return False for unregistered event types."""
        registry = HandlerRegistry()

        assert registry.has_handler("NonExistentEvent") is False

    def test_get_registered_types_returns_all_registered_event_types(self):
        """get_registered_types should return a list of all registered event types."""
        registry = HandlerRegistry()
        event_types = ["EventA", "EventB", "EventC"]

        for event_type in event_types:
            registry.register(event_type, MockProjectionHandler())

        registered_types = registry.get_registered_types()

        assert set(registered_types) == set(event_types)
        assert len(registered_types) == len(event_types)

    def test_get_registered_types_returns_empty_list_when_no_handlers(self):
        """get_registered_types should return empty list when no handlers registered."""
        registry = HandlerRegistry()

        assert registry.get_registered_types() == []

    def test_get_registered_types_returns_copy_not_reference(self):
        """get_registered_types should return a copy, not internal reference."""
        registry = HandlerRegistry()
        registry.register("TestEvent", MockProjectionHandler())

        types_list = registry.get_registered_types()
        types_list.append("FakeEvent")

        # Modifying returned list should not affect internal state
        assert "FakeEvent" not in registry.get_registered_types()


class TestGetHandlerRegistry:
    """Tests for the global registry singleton accessor."""

    def test_get_handler_registry_returns_handler_registry_instance(self):
        """get_handler_registry should return a HandlerRegistry instance."""
        # Reset global state for test isolation
        import src.projections.handlers.registry as registry_module
        registry_module._registry = None

        registry = get_handler_registry()

        assert isinstance(registry, HandlerRegistry)

    def test_get_handler_registry_returns_same_instance_on_multiple_calls(self):
        """get_handler_registry should return the same singleton instance."""
        # Reset global state for test isolation
        import src.projections.handlers.registry as registry_module
        registry_module._registry = None

        registry1 = get_handler_registry()
        registry2 = get_handler_registry()

        assert registry1 is registry2

    def test_get_handler_registry_creates_instance_only_once(self):
        """get_handler_registry should only create one instance."""
        # Reset global state for test isolation
        import src.projections.handlers.registry as registry_module
        registry_module._registry = None

        registry1 = get_handler_registry()
        registry1.register("TestEvent", MockProjectionHandler())

        registry2 = get_handler_registry()

        # Second call should return same instance with the registered handler
        assert registry2.has_handler("TestEvent")

    def test_get_handler_registry_state_persists_across_calls(self):
        """Handlers registered via one call should be visible in subsequent calls."""
        # Reset global state for test isolation
        import src.projections.handlers.registry as registry_module
        registry_module._registry = None

        handler = MockProjectionHandler()
        registry1 = get_handler_registry()
        registry1.register("PersistentEvent", handler)

        registry2 = get_handler_registry()

        assert registry2.get_handler("PersistentEvent") is handler
