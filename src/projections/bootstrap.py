"""Bootstrap handler registry with all projection handlers."""

import logging

from src.events.store import get_event_store_client
from src.projections.handlers.interview_handlers import (
    InterviewCreatedHandler,
    InterviewMetadataUpdatedHandler,
    InterviewStatusChangedHandler,
)
from src.projections.handlers.registry import HandlerRegistry
from src.projections.handlers.sentence_handlers import (
    AnalysisGeneratedHandler,
    AnalysisOverriddenHandler,
    SentenceCreatedHandler,
    SentenceEditedHandler,
)
from src.projections.parked_events import ParkedEventsManager

logger = logging.getLogger(__name__)


def create_handler_registry(
    parked_events_manager: ParkedEventsManager = None,
) -> HandlerRegistry:
    """
    Create and populate a handler registry with all projection handlers.

    Args:
        parked_events_manager: Shared parked events manager (created if not provided)

    Returns:
        HandlerRegistry: Fully populated registry
    """
    registry = HandlerRegistry()

    # Create shared parked events manager
    if parked_events_manager is None:
        parked_events_manager = ParkedEventsManager(get_event_store_client())

    # Register Interview handlers
    registry.register("InterviewCreated", InterviewCreatedHandler(parked_events_manager))
    registry.register("InterviewUpdated", InterviewMetadataUpdatedHandler(parked_events_manager))
    registry.register("StatusChanged", InterviewStatusChangedHandler(parked_events_manager))

    # Register Sentence handlers
    registry.register("SentenceCreated", SentenceCreatedHandler(parked_events_manager))
    registry.register("SentenceEdited", SentenceEditedHandler(parked_events_manager))
    registry.register("AnalysisGenerated", AnalysisGeneratedHandler(parked_events_manager))
    registry.register("AnalysisOverridden", AnalysisOverriddenHandler(parked_events_manager))

    registered_types = registry.get_registered_types()
    logger.info(f"Handler registry initialized with {len(registered_types)} handlers: {registered_types}")

    return registry
