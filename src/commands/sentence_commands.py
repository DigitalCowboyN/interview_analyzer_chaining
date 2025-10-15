"""
Sentence-related commands.

Commands for creating, editing, and analyzing sentences.
"""

from typing import Any, Dict, List, Optional

from pydantic import Field

from . import Command


class CreateSentenceCommand(Command):
    """Command to create a new sentence."""

    sentence_id: str = Field(..., description="Unique identifier for the sentence")
    interview_id: str = Field(..., description="Interview this sentence belongs to")
    index: int = Field(..., ge=0, description="Sequence order within the interview")
    text: str = Field(..., description="The sentence text content")
    speaker: Optional[str] = Field(None, description="Speaker identifier if available")
    start_ms: Optional[int] = Field(None, ge=0, description="Start time in milliseconds")
    end_ms: Optional[int] = Field(None, ge=0, description="End time in milliseconds")


class EditSentenceCommand(Command):
    """Command to edit an existing sentence."""

    sentence_id: str = Field(..., description="Unique identifier for the sentence")
    interview_id: str = Field(..., description="Interview this sentence belongs to")
    new_text: str = Field(..., description="Updated sentence text")
    editor_type: str = Field(default="human", description="Type of editor (human/ai)")


class GenerateAnalysisCommand(Command):
    """Command to generate AI analysis for a sentence."""

    sentence_id: str = Field(..., description="Unique identifier for the sentence")
    interview_id: str = Field(..., description="Interview this sentence belongs to")
    model: str = Field(..., description="AI model used for analysis")
    model_version: str = Field(..., description="Model version")
    classification: Dict[str, Any] = Field(..., description="Classification results")
    keywords: List[str] = Field(default_factory=list, description="Overall keywords")
    topics: List[str] = Field(default_factory=list, description="Topics identified")
    domain_keywords: List[str] = Field(default_factory=list, description="Domain-specific keywords")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confidence score")
    raw_ref: Optional[str] = Field(None, description="Reference to raw analysis data")


class OverrideAnalysisCommand(Command):
    """Command to manually override analysis results."""

    sentence_id: str = Field(..., description="Unique identifier for the sentence")
    interview_id: str = Field(..., description="Interview this sentence belongs to")
    fields_overridden: Dict[str, Any] = Field(..., description="Fields that were manually overridden")
    note: Optional[str] = Field(None, description="Note explaining the override")


class RegenerateAnalysisCommand(Command):
    """Command to regenerate analysis for a sentence."""

    sentence_id: str = Field(..., description="Unique identifier for the sentence")
    interview_id: str = Field(..., description="Interview this sentence belongs to")
    model: str = Field(..., description="AI model used for re-analysis")
    reason: str = Field(..., description="Reason for regeneration")
