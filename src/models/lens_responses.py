"""Response models for lens extractors (meeting_minutes lens).

Same strict-mode contract as extractor_responses: closed schemas, numeric
confidence. Optional fields are declared WITHOUT defaults (required, nullable)
so every property stays in `required` — OpenAI strict json_schema mode rejects
schemas with optional properties.
"""

from typing import List, Optional

from pydantic import Field

from src.models.extractor_responses import StrictResult


class ObjectiveItem(StrictResult):
    text: str
    confidence: float = Field(..., ge=0.0, le=1.0)


class ObjectivesResult(StrictResult):
    objectives: List[ObjectiveItem]


class DecisionItem(StrictResult):
    text: str
    made_by: Optional[str]
    confidence: float = Field(..., ge=0.0, le=1.0)


class DecisionsResult(StrictResult):
    decisions: List[DecisionItem]


class ActionItem(StrictResult):
    text: str
    owner: Optional[str]
    due: Optional[str]
    confidence: float = Field(..., ge=0.0, le=1.0)


class ActionItemsResult(StrictResult):
    action_items: List[ActionItem]


class FollowupItem(StrictResult):
    text: str
    confidence: float = Field(..., ge=0.0, le=1.0)


class FollowupsResult(StrictResult):
    followups: List[FollowupItem]
