"""Enrichment data models."""

import importlib
from typing import List, Literal

from pydantic import BaseModel, Field


class ExtractorSpec(BaseModel):
    """One declared, focused extractor (one LLM call per unit of its scope).

    Prompt-placeholder contract: the executor ALWAYS supplies `{sentence}`
    (the unit's text — fragment text or utterance text depending on scope).
    `context_needs` names the single context key whose value is supplied as
    `{context}`; empty means the prompt uses no `{context}` placeholder.
    `domain_keywords` extractors additionally receive `{domain_keywords}`.
    """

    name: str
    prompt_key: str
    response_model: str
    response_module: str = "src.models.extractor_responses"
    context_needs: List[str] = Field(default_factory=list)
    scope: Literal["fragment", "utterance", "document"] = "fragment"
    enabled: bool = True

    def resolve_model(self) -> type:
        """Resolve the response_model name against response_module."""
        module = importlib.import_module(self.response_module)
        model = getattr(module, self.response_model, None)
        if model is None:
            raise ValueError(f"Unknown response model: {self.response_model}")
        return model
