"""Response models for registry extractors (numeric confidence throughout).

All models forbid extra keys: Pydantic then emits `additionalProperties: false`
in model_json_schema(), which OpenAI's strict json_schema mode REQUIRES on
every object — without it the fallback provider 400s on every structured call.
"""

from typing import List, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictResult(BaseModel):
    """Base for extractor responses: closed schemas for strict-mode providers."""

    model_config = ConfigDict(extra="forbid")


class FunctionTypeResult(StrictResult):
    function_type: str
    confidence: float = Field(..., ge=0.0, le=1.0)


class StructureTypeResult(StrictResult):
    structure_type: str
    confidence: float = Field(..., ge=0.0, le=1.0)


class PurposeResult(StrictResult):
    purpose: str
    confidence: float = Field(..., ge=0.0, le=1.0)


class TopicLevel1Result(StrictResult):
    topic_level_1: str
    confidence: float = Field(..., ge=0.0, le=1.0)


class TopicLevel3Result(StrictResult):
    topic_level_3: str
    confidence: float = Field(..., ge=0.0, le=1.0)


class OverallKeywordsResult(StrictResult):
    overall_keywords: List[str]


class DomainKeywordsResult(StrictResult):
    domain_keywords: List[str]


class EntityMention(StrictResult):
    text: str
    entity_type: str = Field(..., description="person | organization | product | tool | other")
    start: int = Field(..., ge=0, description="Offset within the fragment text")
    end: int = Field(..., gt=0)
    confidence: float = Field(..., ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _span_valid(self) -> "EntityMention":
        if self.end <= self.start:
            raise ValueError("end must be > start")
        return self


class EntityMentionsResult(StrictResult):
    entities: List[EntityMention]


class ClaimItem(StrictResult):
    text: str
    kind: Literal["assertion", "commitment", "request"]
    confidence: float = Field(..., ge=0.0, le=1.0)


class ClaimsResult(StrictResult):
    claims: List[ClaimItem]
