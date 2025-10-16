"""
Command handler implementations.

Handlers validate commands, load aggregates, execute business logic,
and persist events to the event store.
"""

import logging
from typing import Optional

from src.events.aggregates import Interview, Sentence
from src.events.envelope import Actor, generate_correlation_id
from src.events.interview_events import InterviewStatus
from src.events.repository import RepositoryFactory
from src.events.sentence_events import EditorType
from src.events.store import EventStoreClient, get_event_store_client

from . import (
    CommandExecutionError,
    CommandHandler,
    CommandResult,
    CommandValidationError,
)
from .interview_commands import (
    ChangeInterviewStatusCommand,
    CreateInterviewCommand,
    UpdateInterviewCommand,
)
from .sentence_commands import (
    CreateSentenceCommand,
    EditSentenceCommand,
    GenerateAnalysisCommand,
    OverrideAnalysisCommand,
)

logger = logging.getLogger(__name__)


class InterviewCommandHandler(CommandHandler):
    """Handles interview-related commands."""

    def __init__(self, event_store: Optional[EventStoreClient] = None):
        """
        Initialize the handler.

        Args:
            event_store: EventStore client (uses global if not provided)
        """
        self.event_store = event_store or get_event_store_client()
        self.repo_factory = RepositoryFactory(self.event_store)

    async def handle(self, command) -> CommandResult:
        """
        Handle an interview command.

        Args:
            command: Interview command to handle

        Returns:
            CommandResult: Result of command execution
        """
        if isinstance(command, CreateInterviewCommand):
            return await self._handle_create(command)
        elif isinstance(command, UpdateInterviewCommand):
            return await self._handle_update(command)
        elif isinstance(command, ChangeInterviewStatusCommand):
            return await self._handle_status_change(command)
        else:
            raise CommandValidationError(f"Unknown command type: {type(command)}")

    async def _handle_create(self, command: CreateInterviewCommand) -> CommandResult:
        """Handle CreateInterviewCommand."""
        try:
            repo = self.repo_factory.create_interview_repository()

            # Check if interview already exists
            existing = await repo.load(command.interview_id)
            if existing is not None:
                raise CommandValidationError(f"Interview {command.interview_id} already exists", field="interview_id")

            # Create new interview aggregate
            interview = Interview(command.interview_id)

            # Execute command
            interview.create(
                title=command.title,
                source=command.source,
                language=command.language,
                started_at=command.started_at,
                metadata=command.metadata,
                actor=command.actor,
                correlation_id=command.correlation_id or generate_correlation_id(),
            )

            # Persist events
            await repo.save(interview)

            logger.info(
                f"Created interview {command.interview_id} with {len(interview.get_uncommitted_events())} events"
            )

            return CommandResult(
                aggregate_id=command.interview_id,
                version=interview.version,
                event_count=len(interview.get_uncommitted_events()),
                message=f"Interview '{command.title}' created successfully",
            )

        except CommandValidationError:
            raise
        except Exception as e:
            logger.error(f"Failed to create interview {command.interview_id}: {e}", exc_info=True)
            raise CommandExecutionError(f"Failed to create interview: {e}", original_error=e)

    async def _handle_update(self, command: UpdateInterviewCommand) -> CommandResult:
        """Handle UpdateInterviewCommand."""
        try:
            repo = self.repo_factory.create_interview_repository()

            # Load existing interview
            interview = await repo.load(command.interview_id)
            if interview is None:
                raise CommandValidationError(f"Interview {command.interview_id} not found", field="interview_id")

            # Execute command
            interview.update(
                title=command.title,
                language=command.language,
                metadata_diff=command.metadata_diff,
                actor=command.actor,
                correlation_id=command.correlation_id or generate_correlation_id(),
            )

            # Persist events
            await repo.save(interview)

            logger.info(f"Updated interview {command.interview_id}")

            return CommandResult(
                aggregate_id=command.interview_id,
                version=interview.version,
                event_count=len(interview.get_uncommitted_events()),
                message="Interview updated successfully",
            )

        except CommandValidationError:
            raise
        except Exception as e:
            logger.error(f"Failed to update interview {command.interview_id}: {e}", exc_info=True)
            raise CommandExecutionError(f"Failed to update interview: {e}", original_error=e)

    async def _handle_status_change(self, command: ChangeInterviewStatusCommand) -> CommandResult:
        """Handle ChangeInterviewStatusCommand."""
        try:
            repo = self.repo_factory.create_interview_repository()

            # Load existing interview
            interview = await repo.load(command.interview_id)
            if interview is None:
                raise CommandValidationError(f"Interview {command.interview_id} not found", field="interview_id")

            # Validate status
            try:
                new_status = InterviewStatus(command.new_status)
            except ValueError:
                raise CommandValidationError(f"Invalid status: {command.new_status}", field="new_status")

            # Execute command
            interview.change_status(
                new_status=new_status,
                reason=command.reason,
                actor=command.actor,
                correlation_id=command.correlation_id or generate_correlation_id(),
            )

            # Persist events
            await repo.save(interview)

            logger.info(f"Changed interview {command.interview_id} status to {new_status}")

            return CommandResult(
                aggregate_id=command.interview_id,
                version=interview.version,
                event_count=len(interview.get_uncommitted_events()),
                message=f"Interview status changed to {new_status.value}",
            )

        except CommandValidationError:
            raise
        except Exception as e:
            logger.error(f"Failed to change interview status {command.interview_id}: {e}", exc_info=True)
            raise CommandExecutionError(f"Failed to change interview status: {e}", original_error=e)


class SentenceCommandHandler(CommandHandler):
    """Handles sentence-related commands."""

    def __init__(self, event_store: Optional[EventStoreClient] = None):
        """
        Initialize the handler.

        Args:
            event_store: EventStore client (uses global if not provided)
        """
        self.event_store = event_store or get_event_store_client()
        self.repo_factory = RepositoryFactory(self.event_store)

    async def handle(self, command) -> CommandResult:
        """
        Handle a sentence command.

        Args:
            command: Sentence command to handle

        Returns:
            CommandResult: Result of command execution
        """
        if isinstance(command, CreateSentenceCommand):
            return await self._handle_create(command)
        elif isinstance(command, EditSentenceCommand):
            return await self._handle_edit(command)
        elif isinstance(command, GenerateAnalysisCommand):
            return await self._handle_generate_analysis(command)
        elif isinstance(command, OverrideAnalysisCommand):
            return await self._handle_override_analysis(command)
        else:
            raise CommandValidationError(f"Unknown command type: {type(command)}")

    async def _handle_create(self, command: CreateSentenceCommand) -> CommandResult:
        """Handle CreateSentenceCommand."""
        try:
            repo = self.repo_factory.create_sentence_repository()

            # Check if sentence already exists
            existing = await repo.load(command.sentence_id)
            if existing is not None:
                raise CommandValidationError(f"Sentence {command.sentence_id} already exists", field="sentence_id")

            # Create new sentence aggregate
            sentence = Sentence(command.sentence_id)

            # Execute command
            sentence.create(
                interview_id=command.interview_id,
                index=command.index,
                text=command.text,
                speaker=command.speaker,
                start_ms=command.start_ms,
                end_ms=command.end_ms,
                actor=command.actor,
                correlation_id=command.correlation_id or generate_correlation_id(),
            )

            # Persist events
            await repo.save(sentence)

            logger.debug(f"Created sentence {command.sentence_id} for interview {command.interview_id}")

            return CommandResult(
                aggregate_id=command.sentence_id,
                version=sentence.version,
                event_count=len(sentence.get_uncommitted_events()),
                message="Sentence created successfully",
            )

        except CommandValidationError:
            raise
        except Exception as e:
            logger.error(f"Failed to create sentence {command.sentence_id}: {e}", exc_info=True)
            raise CommandExecutionError(f"Failed to create sentence: {e}", original_error=e)

    async def _handle_edit(self, command: EditSentenceCommand) -> CommandResult:
        """Handle EditSentenceCommand."""
        try:
            repo = self.repo_factory.create_sentence_repository()

            # Load existing sentence
            sentence = await repo.load(command.sentence_id)
            if sentence is None:
                raise CommandValidationError(f"Sentence {command.sentence_id} not found", field="sentence_id")

            # Validate editor type
            try:
                editor_type = EditorType(command.editor_type.lower())
            except ValueError:
                raise CommandValidationError(f"Invalid editor type: {command.editor_type}", field="editor_type")

            # Execute command
            sentence.edit(
                new_text=command.new_text,
                editor_type=editor_type,
                actor=command.actor,
                correlation_id=command.correlation_id or generate_correlation_id(),
            )

            # Persist events
            await repo.save(sentence)

            logger.info(f"Edited sentence {command.sentence_id}")

            return CommandResult(
                aggregate_id=command.sentence_id,
                version=sentence.version,
                event_count=len(sentence.get_uncommitted_events()),
                message="Sentence edited successfully",
            )

        except CommandValidationError:
            raise
        except Exception as e:
            logger.error(f"Failed to edit sentence {command.sentence_id}: {e}", exc_info=True)
            raise CommandExecutionError(f"Failed to edit sentence: {e}", original_error=e)

    async def _handle_generate_analysis(self, command: GenerateAnalysisCommand) -> CommandResult:
        """Handle GenerateAnalysisCommand."""
        try:
            repo = self.repo_factory.create_sentence_repository()

            # Load existing sentence
            sentence = await repo.load(command.sentence_id)
            if sentence is None:
                raise CommandValidationError(f"Sentence {command.sentence_id} not found", field="sentence_id")

            # Execute command
            sentence.generate_analysis(
                model=command.model,
                model_version=command.model_version,
                classification=command.classification,
                keywords=command.keywords,
                topics=command.topics,
                domain_keywords=command.domain_keywords,
                confidence=command.confidence,
                raw_ref=command.raw_ref,
                actor=command.actor,
                correlation_id=command.correlation_id or generate_correlation_id(),
            )

            # Persist events
            await repo.save(sentence)

            logger.debug(f"Generated analysis for sentence {command.sentence_id}")

            return CommandResult(
                aggregate_id=command.sentence_id,
                version=sentence.version,
                event_count=len(sentence.get_uncommitted_events()),
                message="Analysis generated successfully",
            )

        except CommandValidationError:
            raise
        except Exception as e:
            logger.error(f"Failed to generate analysis for sentence {command.sentence_id}: {e}", exc_info=True)
            raise CommandExecutionError(f"Failed to generate analysis: {e}", original_error=e)

    async def _handle_override_analysis(self, command: OverrideAnalysisCommand) -> CommandResult:
        """Handle OverrideAnalysisCommand."""
        try:
            repo = self.repo_factory.create_sentence_repository()

            # Load existing sentence
            sentence = await repo.load(command.sentence_id)
            if sentence is None:
                raise CommandValidationError(f"Sentence {command.sentence_id} not found", field="sentence_id")

            # Execute command
            sentence.override_analysis(
                fields_overridden=command.fields_overridden,
                note=command.note,
                actor=command.actor,
                correlation_id=command.correlation_id or generate_correlation_id(),
            )

            # Persist events
            await repo.save(sentence)

            logger.info(f"Overrode analysis for sentence {command.sentence_id}")

            return CommandResult(
                aggregate_id=command.sentence_id,
                version=sentence.version,
                event_count=len(sentence.get_uncommitted_events()),
                message="Analysis overridden successfully",
            )

        except CommandValidationError:
            raise
        except Exception as e:
            logger.error(f"Failed to override analysis for sentence {command.sentence_id}: {e}", exc_info=True)
            raise CommandExecutionError(f"Failed to override analysis: {e}", original_error=e)


# Global handler instances
_interview_handler: Optional[InterviewCommandHandler] = None
_sentence_handler: Optional[SentenceCommandHandler] = None


def get_interview_command_handler() -> InterviewCommandHandler:
    """Get the global interview command handler instance."""
    global _interview_handler
    if _interview_handler is None:
        _interview_handler = InterviewCommandHandler()
    return _interview_handler


def get_sentence_command_handler() -> SentenceCommandHandler:
    """Get the global sentence command handler instance."""
    global _sentence_handler
    if _sentence_handler is None:
        _sentence_handler = SentenceCommandHandler()
    return _sentence_handler
