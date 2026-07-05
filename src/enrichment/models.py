"""Enrichment data models."""

import importlib
from typing import List, Literal

from pydantic import BaseModel, Field


class ExtractorSpec(BaseModel):
    """One declared, focused extractor (one LLM call per unit of its scope)."""

    name: str
    prompt_key: str
    response_model: str
    context_needs: List[str] = Field(default_factory=list)
    scope: Literal["fragment", "utterance", "document"] = "fragment"
    enabled: bool = True

    def resolve_model(self) -> type:
        """Resolve the response_model name against src.models.extractor_responses."""
        module = importlib.import_module("src.models.extractor_responses")
        model = getattr(module, self.response_model, None)
        if model is None:
            raise ValueError(f"Unknown response model: {self.response_model}")
        return model
