"""
src/api/schemas.py

Defines Pydantic models used for API request validation and response serialization.
"""

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class FileListResponse(BaseModel):
    """Response model for listing available analysis filenames."""

    filenames: List[str] = Field(
        ..., description="List of filenames for completed analyses."
    )


# === Models for File Content Endpoint ===


class AnalysisResult(BaseModel):
    """Represents the analysis result for a single sentence."""

    # Allow for potential error fields or other dynamic fields from analysis
    model_config = ConfigDict(extra="allow")

    sentence_id: int = Field(
        ..., description="Unique identifier for the sentence within the file."
    )
    sequence_order: int = Field(
        ..., description="Original sequence order of the sentence."
    )
    sentence: str = Field(..., description="The original sentence text.")

    # Include known analysis fields (optional makes them robust to missing keys)
    function_type: Optional[str] = None
    structure_type: Optional[str] = None
    purpose: Optional[str] = None
    topic_level_1: Optional[str] = None
    topic_level_3: Optional[str] = None
    overall_keywords: Optional[List[str]] = None
    domain_keywords: Optional[List[str]] = None


class FileContentResponse(BaseModel):
    """Response model for returning the content of an analysis file."""

    filename: str = Field(..., description="The name of the analysis file.")
    results: List[AnalysisResult] = Field(
        ..., description="List of analysis results for each sentence."
    )


# --- Schemas for Analysis Triggering ---


class AnalysisRequest(BaseModel):
    """Request model for triggering analysis on a specific input file."""

    input_filename: str


class AnalysisResponse(BaseModel):
    """Response model after successfully triggering an analysis."""

    message: str
    input_filename: str
    # Potentially add job_id here later if using background tasks


# === Models for Analysis Trigger Endpoint ===


class AnalysisTriggerRequest(BaseModel):
    """Request model for triggering analysis on a specific input file."""

    input_filename: str = Field(
        ...,
        description="Name of the input file (e.g., 'interview1.txt') located in the configured input directory.",
    )


class AnalysisTriggerResponse(BaseModel):
    """Response model after successfully triggering an analysis task."""

    message: str = Field(..., description="Status message indicating analysis started.")
    input_filename: str = Field(
        ..., description="The input filename for which analysis was triggered."
    )
    task_id: str = Field(
        ..., description="Unique identifier assigned to this analysis task."
    )
    # job_id: Optional[str] = None # Consider adding later for tracking background tasks
