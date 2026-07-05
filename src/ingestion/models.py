"""Data models for Layer 1 ingestion (format detection and normalization)."""

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, model_validator


class TranscriptFormat(str, Enum):
    """Detected input transcript format."""

    LABELED = "labeled"  # speaker-prefixed lines, e.g. "Alice: ..."
    FLAT = "flat"  # undifferentiated prose, no speaker markers


class RawFragment(BaseModel):
    """A contiguous run of speech, verbatim, grounded in the source text."""

    text: str = Field(..., description="Verbatim fragment text (stripped)")
    start_char: int = Field(..., ge=0, description="Offset into source text")
    end_char: int = Field(..., gt=0, description="End offset into source text")
    sequence_order: int = Field(..., ge=0, description="As-spoken order, immutable")
    speaker_label: Optional[str] = Field(
        None, description="Speaker label parsed from the source, if the format had one"
    )

    @model_validator(mode="after")
    def _end_after_start(self) -> "RawFragment":
        if self.end_char <= self.start_char:
            raise ValueError("end_char must be > start_char")
        return self


class NormalizedTranscript(BaseModel):
    """Result of normalizing a raw input into offset-grounded fragments."""

    content_hash: str = Field(..., description="sha256 hex digest of the source text")
    format: TranscriptFormat
    fragments: List[RawFragment]
    speaker_labels: List[str] = Field(
        default_factory=list, description="Distinct parsed labels, in order of first appearance"
    )
