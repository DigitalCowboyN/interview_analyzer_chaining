"""
Command infrastructure for event-sourced architecture.

Commands represent user intent and are dispatched to command handlers
which validate business rules and generate domain events.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from src.events.envelope import Actor


class Command(BaseModel):
    """Base class for all commands."""

    correlation_id: Optional[str] = Field(None, description="Correlation ID for tracking related operations")
    actor: Optional[Actor] = Field(None, description="Who/what initiated this command")

    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True


class CommandResult(BaseModel):
    """Result of command execution."""

    aggregate_id: str = Field(..., description="ID of the aggregate that was modified")
    version: int = Field(..., description="New version of the aggregate after command execution")
    event_count: int = Field(..., description="Number of events generated")
    success: bool = Field(default=True, description="Whether command executed successfully")
    message: Optional[str] = Field(None, description="Optional message about the result")


class CommandHandler(ABC):
    """Base class for command handlers."""

    @abstractmethod
    async def handle(self, command: Command) -> CommandResult:
        """
        Handle a command and return the result.

        Args:
            command: The command to handle

        Returns:
            CommandResult: Result of command execution

        Raises:
            ValueError: If command validation fails
            Exception: If command execution fails
        """
        pass


class CommandValidationError(Exception):
    """Raised when command validation fails."""

    def __init__(self, message: str, field: Optional[str] = None):
        super().__init__(message)
        self.field = field
        self.message = message


class CommandExecutionError(Exception):
    """Raised when command execution fails."""

    def __init__(self, message: str, original_error: Optional[Exception] = None):
        super().__init__(message)
        self.message = message
        self.original_error = original_error
