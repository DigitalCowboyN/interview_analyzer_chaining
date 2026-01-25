"""
src/api/routers/edits.py

API router for user edit operations (sentence edits, analysis overrides).
Implements command-based user corrections with event sourcing.
"""

import uuid
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel, Field

from src.commands.handlers import SentenceCommandHandler
from src.commands.sentence_commands import EditSentenceCommand, OverrideAnalysisCommand
from src.events.envelope import Actor, ActorType
from src.events.repository import SentenceRepository
from src.events.store import EventStoreClient
from src.utils.logger import get_logger

router = APIRouter(prefix="/edits", tags=["Edits"])
logger = get_logger()


# --- Request/Response Models ---

class EditSentenceRequest(BaseModel):
    """Request model for editing sentence text."""

    text: str = Field(..., description="New sentence text", min_length=1)
    editor_type: str = Field(
        default="human",
        description="Type of editor making the change",
    )
    note: Optional[str] = Field(None, description="Optional edit note/reason")


class OverrideAnalysisRequest(BaseModel):
    """Request model for overriding analysis results."""

    function_type: Optional[str] = Field(None, description="Override function type")
    structure_type: Optional[str] = Field(None, description="Override structure type")
    purpose: Optional[str] = Field(None, description="Override purpose")
    keywords: Optional[list[str]] = Field(None, description="Override keywords")
    topics: Optional[list[str]] = Field(None, description="Override topics")
    domain_keywords: Optional[list[str]] = Field(
        None, description="Override domain keywords"
    )
    note: Optional[str] = Field(None, description="Optional override note/reason")


class EditResponse(BaseModel):
    """Response model for edit operations."""

    status: str = Field(..., description="Operation status (accepted, error)")
    sentence_id: str = Field(..., description="Sentence UUID")
    version: int = Field(..., description="New version number after edit")
    event_count: int = Field(..., description="Number of events generated")
    message: str = Field(..., description="Human-readable message")


# --- Helper Functions ---

def get_event_store() -> EventStoreClient:
    """Get EventStoreDB client instance with environment-aware defaults."""
    import os

    from src.config import config
    from src.utils.environment import detect_environment

    # Priority: 1) config file, 2) env var, 3) environment-aware default
    connection_string = config.get("event_sourcing", {}).get("connection_string")
    if not connection_string:
        connection_string = os.getenv("ESDB_CONNECTION_STRING")
    if not connection_string:
        env = detect_environment()
        if env in ("docker", "ci"):
            connection_string = "esdb://eventstore:2113?tls=false"
        else:
            connection_string = "esdb://localhost:2113?tls=false"
    return EventStoreClient(connection_string)


def get_sentence_command_handler() -> SentenceCommandHandler:
    """Get SentenceCommandHandler instance."""
    event_store = get_event_store()
    return SentenceCommandHandler(event_store)


def get_sentence_repository() -> SentenceRepository:
    """Get SentenceRepository instance for loading sentences."""
    event_store = get_event_store()
    return SentenceRepository(event_store)


def create_actor_from_request(
    user_id: Optional[str] = None,
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
) -> Actor:
    """
    Create Actor from request headers.

    Args:
        user_id: User ID from path/query (takes precedence)
        x_user_id: User ID from X-User-ID header

    Returns:
        Actor instance with HUMAN actor type
    """
    # Use provided user_id first, then header, then default
    actual_user_id = user_id or x_user_id or "anonymous"

    return Actor(
        actor_type=ActorType.HUMAN,
        user_id=actual_user_id,
    )


# --- API Endpoints ---

@router.post(
    "/sentences/{interview_id}/{sentence_index}/edit",
    response_model=EditResponse,
    status_code=202,
    summary="Edit Sentence Text",
    description="Edit the text of a sentence. Generates SentenceEdited event.",
)
async def edit_sentence(
    interview_id: str,
    sentence_index: int,
    request: EditSentenceRequest,
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
    x_correlation_id: Optional[str] = Header(None, alias="X-Correlation-ID"),
) -> EditResponse:
    """
    Edit sentence text.

    Args:
        interview_id: Interview UUID
        sentence_index: Sentence index (0-based)
        request: Edit request with new text
        x_user_id: Optional user ID header
        x_correlation_id: Optional correlation ID for tracing

    Returns:
        EditResponse with accepted status and new version

    Raises:
        HTTPException: If sentence not found or validation fails
    """
    try:
        # Generate sentence UUID (deterministic from interview_id:index)
        sentence_id = str(
            uuid.uuid5(uuid.NAMESPACE_DNS, f"{interview_id}:{sentence_index}")
        )

        # Create actor from request
        actor = create_actor_from_request(user_id=None, x_user_id=x_user_id)

        # Generate correlation ID if not provided
        correlation_id = x_correlation_id or str(uuid.uuid4())

        # Create command
        command = EditSentenceCommand(
            sentence_id=sentence_id,
            interview_id=interview_id,
            new_text=request.text,
            editor_type=request.editor_type,
            actor=actor,
            correlation_id=correlation_id,
        )

        # Execute command
        handler = get_sentence_command_handler()
        result = await handler.handle(command)

        logger.info(
            f"Sentence {sentence_id} edited by {actor.user_id} "
            f"(version {result.version}, {result.event_count} events)"
        )

        return EditResponse(
            status="accepted",
            sentence_id=sentence_id,
            version=result.version,
            event_count=result.event_count,
            message=f"Sentence edit accepted. {result.event_count} event(s) generated.",
        )

    except ValueError as e:
        # Sentence not found or validation error
        logger.warning(f"Edit sentence validation error: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error editing sentence: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@router.post(
    "/sentences/{interview_id}/{sentence_index}/analysis/override",
    response_model=EditResponse,
    status_code=202,
    summary="Override Analysis Results",
    description="Override AI analysis results with human corrections. Generates AnalysisOverridden event.",
)
async def override_analysis(
    interview_id: str,
    sentence_index: int,
    request: OverrideAnalysisRequest,
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
    x_correlation_id: Optional[str] = Header(None, alias="X-Correlation-ID"),
) -> EditResponse:
    """
    Override analysis results with human corrections.

    Args:
        interview_id: Interview UUID
        sentence_index: Sentence index (0-based)
        request: Override request with corrected analysis fields
        x_user_id: Optional user ID header
        x_correlation_id: Optional correlation ID for tracing

    Returns:
        EditResponse with accepted status and new version

    Raises:
        HTTPException: If sentence not found or validation fails
    """
    try:
        # Generate sentence UUID (deterministic from interview_id:index)
        sentence_id = str(
            uuid.uuid5(uuid.NAMESPACE_DNS, f"{interview_id}:{sentence_index}")
        )

        # Create actor from request
        actor = create_actor_from_request(user_id=None, x_user_id=x_user_id)

        # Generate correlation ID if not provided
        correlation_id = x_correlation_id or str(uuid.uuid4())

        # Build overrides dictionary (only include provided fields)
        overrides: Dict[str, Any] = {}
        if request.function_type is not None:
            overrides["function_type"] = request.function_type
        if request.structure_type is not None:
            overrides["structure_type"] = request.structure_type
        if request.purpose is not None:
            overrides["purpose"] = request.purpose
        if request.keywords is not None:
            overrides["keywords"] = request.keywords
        if request.topics is not None:
            overrides["topics"] = request.topics
        if request.domain_keywords is not None:
            overrides["domain_keywords"] = request.domain_keywords

        if not overrides:
            raise HTTPException(
                status_code=400,
                detail="At least one analysis field must be provided for override",
            )

        # Create command
        command = OverrideAnalysisCommand(
            sentence_id=sentence_id,
            interview_id=interview_id,
            fields_overridden=overrides,
            note=request.note,
            actor=actor,
            correlation_id=correlation_id,
        )

        # Execute command
        handler = get_sentence_command_handler()
        result = await handler.handle(command)

        logger.info(
            f"Analysis for sentence {sentence_id} overridden by {actor.user_id} "
            f"(version {result.version}, {result.event_count} events, "
            f"{len(overrides)} fields)"
        )

        return EditResponse(
            status="accepted",
            sentence_id=sentence_id,
            version=result.version,
            event_count=result.event_count,
            message=f"Analysis override accepted. {result.event_count} event(s) generated.",
        )

    except HTTPException:
        # Re-raise HTTPExceptions (e.g., 400 for validation)
        raise
    except ValueError as e:
        # Sentence not found or validation error
        logger.warning(f"Override analysis validation error: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error overriding analysis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@router.get(
    "/sentences/{interview_id}/{sentence_index}/history",
    summary="Get Sentence Edit History",
    description="Get complete edit history for a sentence (all events).",
)
async def get_sentence_history(
    interview_id: str,
    sentence_index: int,
) -> Dict[str, Any]:
    """
    Get complete edit history for a sentence.

    Args:
        interview_id: Interview UUID
        sentence_index: Sentence index (0-based)

    Returns:
        Dictionary with sentence history (events)

    Raises:
        HTTPException: If sentence not found
    """
    try:
        # Generate sentence UUID (deterministic from interview_id:index)
        sentence_id = str(
            uuid.uuid5(uuid.NAMESPACE_DNS, f"{interview_id}:{sentence_index}")
        )

        # Load sentence from repository (which loads all events)
        repo = get_sentence_repository()
        sentence = await repo.load(sentence_id)

        if sentence is None:
            raise HTTPException(
                status_code=404,
                detail=f"Sentence not found: {sentence_id}",
            )

        # Get event history from EventStoreDB
        event_store = get_event_store()
        stream_name = f"Sentence-{sentence_id}"
        events = await event_store.read_stream(stream_name)

        # Build history response
        history = {
            "sentence_id": sentence_id,
            "interview_id": interview_id,
            "sentence_index": sentence_index,
            "current_version": sentence.version,
            "current_text": sentence.text,
            "is_edited": sentence.text != getattr(sentence, "original_text", sentence.text),
            "event_count": len(events),
            "events": [
                {
                    "event_type": event.event_type,
                    "version": event.version,
                    "occurred_at": event.occurred_at.isoformat(),
                    "actor": {
                        "actor_type": event.actor.actor_type,
                        "user_id": event.actor.user_id,
                    },
                    "correlation_id": event.correlation_id,
                    "data": event.data,
                }
                for event in events
            ],
        }

        return history

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting sentence history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
