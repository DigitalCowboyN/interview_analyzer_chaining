"""Pydantic models validating LLM responses for Layer 1 ingestion passes."""

from typing import List

from pydantic import BaseModel, Field


class SpeakerAssignment(BaseModel):
    """One fragment's proposed speaker assignment."""

    index: int = Field(..., ge=0, description="Fragment index within the window")
    speaker: str = Field(..., description="Provisional handle, e.g. 'S1'")
    confidence: float = Field(..., ge=0.0, le=1.0)


class SpeakerWindowResponse(BaseModel):
    """Response for one speaker-inference window."""

    assignments: List[SpeakerAssignment]


class UtteranceProposal(BaseModel):
    """A proposed stitched utterance (possibly spanning non-adjacent fragments)."""

    speaker: str = Field(..., description="Speaker handle")
    fragment_indices: List[int] = Field(..., min_length=1, description="Ordered fragment indices")
    confidence: float = Field(..., ge=0.0, le=1.0)


class InterruptionProposal(BaseModel):
    """A proposed interruption between two utterances in this window."""

    interrupting: int = Field(..., ge=0, description="Index into the utterances list")
    interrupted: int = Field(..., ge=0, description="Index into the utterances list")
    at_index: int = Field(..., ge=0, description="Fragment index where the interruption begins")


class StitchWindowResponse(BaseModel):
    """Response for one stitching window."""

    utterances: List[UtteranceProposal]
    interruptions: List[InterruptionProposal] = Field(default_factory=list)
