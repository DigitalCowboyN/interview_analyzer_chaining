"""
Unit tests for the projection bootstrap module.

Tests the create_handler_registry function which initializes all projection handlers.
"""

import logging
from unittest.mock import MagicMock, patch

import pytest

from src.projections.bootstrap import create_handler_registry
from src.projections.handlers.registry import HandlerRegistry
from src.projections.handlers.interview_handlers import (
    InterviewCreatedHandler,
    InterviewMetadataUpdatedHandler,
    InterviewStatusChangedHandler,
)
from src.projections.handlers.sentence_handlers import (
    AnalysisGeneratedHandler,
    AnalysisOverriddenHandler,
    SentenceCreatedHandler,
    SentenceEditedHandler,
)
from src.projections.parked_events import ParkedEventsManager


class TestCreateHandlerRegistry:
    """Tests for create_handler_registry function."""

    @patch("src.projections.bootstrap.get_event_store_client")
    def test_create_handler_registry_returns_handler_registry_instance(
        self, mock_get_client
    ):
        """create_handler_registry should return a HandlerRegistry instance."""
        mock_get_client.return_value = MagicMock()

        registry = create_handler_registry()

        assert isinstance(registry, HandlerRegistry)

    @patch("src.projections.bootstrap.get_event_store_client")
    def test_create_handler_registry_registers_all_interview_handlers(
        self, mock_get_client
    ):
        """Registry should include all interview event handlers."""
        mock_get_client.return_value = MagicMock()

        registry = create_handler_registry()

        # Verify Interview handlers are registered
        assert registry.has_handler("InterviewCreated")
        assert registry.has_handler("InterviewUpdated")
        assert registry.has_handler("StatusChanged")

    @patch("src.projections.bootstrap.get_event_store_client")
    def test_create_handler_registry_registers_all_sentence_handlers(
        self, mock_get_client
    ):
        """Registry should include all sentence event handlers."""
        mock_get_client.return_value = MagicMock()

        registry = create_handler_registry()

        # Verify Sentence handlers are registered
        assert registry.has_handler("SentenceCreated")
        assert registry.has_handler("SentenceEdited")
        assert registry.has_handler("AnalysisGenerated")
        assert registry.has_handler("AnalysisOverridden")

    @patch("src.projections.bootstrap.get_event_store_client")
    def test_create_handler_registry_registers_correct_handler_types(
        self, mock_get_client
    ):
        """Each event type should be mapped to the correct handler class."""
        mock_get_client.return_value = MagicMock()

        registry = create_handler_registry()

        # Verify correct handler types
        assert isinstance(
            registry.get_handler("InterviewCreated"), InterviewCreatedHandler
        )
        assert isinstance(
            registry.get_handler("InterviewUpdated"), InterviewMetadataUpdatedHandler
        )
        assert isinstance(
            registry.get_handler("StatusChanged"), InterviewStatusChangedHandler
        )
        assert isinstance(
            registry.get_handler("SentenceCreated"), SentenceCreatedHandler
        )
        assert isinstance(
            registry.get_handler("SentenceEdited"), SentenceEditedHandler
        )
        assert isinstance(
            registry.get_handler("AnalysisGenerated"), AnalysisGeneratedHandler
        )
        assert isinstance(
            registry.get_handler("AnalysisOverridden"), AnalysisOverriddenHandler
        )

    @patch("src.projections.bootstrap.get_event_store_client")
    def test_create_handler_registry_registers_exactly_seven_handlers(
        self, mock_get_client
    ):
        """Registry should have exactly 7 handlers registered."""
        mock_get_client.return_value = MagicMock()

        registry = create_handler_registry()

        registered_types = registry.get_registered_types()
        assert len(registered_types) == 7

    @patch("src.projections.bootstrap.get_event_store_client")
    def test_create_handler_registry_uses_provided_parked_events_manager(
        self, mock_get_client
    ):
        """When parked_events_manager is provided, it should be used for all handlers."""
        mock_get_client.return_value = MagicMock()
        custom_manager = MagicMock(spec=ParkedEventsManager)

        registry = create_handler_registry(parked_events_manager=custom_manager)

        # Verify handlers received the custom manager
        handler = registry.get_handler("InterviewCreated")
        assert handler.parked_events_manager is custom_manager

        handler = registry.get_handler("SentenceCreated")
        assert handler.parked_events_manager is custom_manager

    @patch("src.projections.bootstrap.get_event_store_client")
    def test_create_handler_registry_creates_parked_events_manager_when_not_provided(
        self, mock_get_client
    ):
        """When no parked_events_manager is provided, one should be created."""
        mock_event_store = MagicMock()
        mock_get_client.return_value = mock_event_store

        registry = create_handler_registry()

        # Verify a ParkedEventsManager was created and used
        handler = registry.get_handler("InterviewCreated")
        assert handler.parked_events_manager is not None
        assert isinstance(handler.parked_events_manager, ParkedEventsManager)

    @patch("src.projections.bootstrap.get_event_store_client")
    def test_create_handler_registry_shares_parked_events_manager_across_handlers(
        self, mock_get_client
    ):
        """All handlers should share the same parked_events_manager instance."""
        mock_get_client.return_value = MagicMock()

        registry = create_handler_registry()

        # Get parked_events_manager from multiple handlers
        interview_handler = registry.get_handler("InterviewCreated")
        sentence_handler = registry.get_handler("SentenceCreated")
        analysis_handler = registry.get_handler("AnalysisGenerated")

        # Verify they all share the same manager
        assert interview_handler.parked_events_manager is sentence_handler.parked_events_manager
        assert sentence_handler.parked_events_manager is analysis_handler.parked_events_manager

    @patch("src.projections.bootstrap.get_event_store_client")
    def test_create_handler_registry_logs_initialization_message(
        self, mock_get_client, caplog
    ):
        """Registry initialization should log the number and types of handlers."""
        mock_get_client.return_value = MagicMock()

        with caplog.at_level(logging.INFO):
            create_handler_registry()

        assert "Handler registry initialized" in caplog.text
        assert "7 handlers" in caplog.text

    @patch("src.projections.bootstrap.get_event_store_client")
    def test_create_handler_registry_returns_new_instance_each_call(
        self, mock_get_client
    ):
        """Each call to create_handler_registry should return a new registry instance."""
        mock_get_client.return_value = MagicMock()

        registry1 = create_handler_registry()
        registry2 = create_handler_registry()

        assert registry1 is not registry2

    @patch("src.projections.bootstrap.get_event_store_client")
    def test_create_handler_registry_event_types_match_expected_names(
        self, mock_get_client
    ):
        """Registered event types should match the exact expected names."""
        mock_get_client.return_value = MagicMock()

        registry = create_handler_registry()

        expected_types = {
            "InterviewCreated",
            "InterviewUpdated",
            "StatusChanged",
            "SentenceCreated",
            "SentenceEdited",
            "AnalysisGenerated",
            "AnalysisOverridden",
        }
        registered_types = set(registry.get_registered_types())

        assert registered_types == expected_types
