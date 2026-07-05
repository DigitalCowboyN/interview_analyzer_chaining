"""Response models for registry extractors (numeric confidence throughout)."""

from typing import List, Literal

from pydantic import BaseModel, Field, model_validator


class FunctionTypeResult(BaseModel):
    function_type: str
    confidence: float = Field(..., ge=0.0, le=1.0)


class StructureTypeResult(BaseModel):
    structure_type: str
    confidence: float = Field(..., ge=0.0, le=1.0)


class PurposeResult(BaseModel):
    purpose: str
    confidence: float = Field(..., ge=0.0, le=1.0)


class TopicLevel1Result(BaseModel):
    topic_level_1: str
    confidence: float = Field(..., ge=0.0, le=1.0)


class TopicLevel3Result(BaseModel):
    topic_level_3: str
    confidence: float = Field(..., ge=0.0, le=1.0)


class OverallKeywordsResult(BaseModel):
    overall_keywords: List[str]


class DomainKeywordsResult(BaseModel):
    domain_keywords: List[str]


class EntityMention(BaseModel):
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


class EntityMentionsResult(BaseModel):
    entities: List[EntityMention]


class ClaimItem(BaseModel):
    text: str
    kind: Literal["assertion", "commitment", "request"]
    confidence: float = Field(..., ge=0.0, le=1.0)


class ClaimsResult(BaseModel):
    claims: List[ClaimItem]
