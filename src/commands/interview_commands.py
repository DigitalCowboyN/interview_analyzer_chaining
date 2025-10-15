"""
Interview-related commands.

Commands for creating and updating interviews.
"""

from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import Field

from . import Command


class CreateInterviewCommand(Command):
    """Command to create a new interview."""

    interview_id: str = Field(..., description="Unique identifier for the interview")
    project_id: str = Field(..., description="Project this interview belongs to")
    title: str = Field(..., description="Title of the interview")
    source: str = Field(..., description="Source file or identifier")
    language: Optional[str] = Field(None, description="Language of the interview content")
    started_at: Optional[datetime] = Field(None, description="When the interview was conducted")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class UpdateInterviewCommand(Command):
    """Command to update an existing interview."""

    interview_id: str = Field(..., description="Unique identifier for the interview")
    title: Optional[str] = Field(None, description="Updated title")
    language: Optional[str] = Field(None, description="Updated language")
    metadata_diff: Dict[str, Any] = Field(default_factory=dict, description="Metadata changes")


class ChangeInterviewStatusCommand(Command):
    """Command to change interview status."""

    interview_id: str = Field(..., description="Unique identifier for the interview")
    new_status: str = Field(..., description="New status to transition to")
    reason: Optional[str] = Field(None, description="Reason for status change")
