"""Lens profile model: one YAML under lenses/ fully describes a lens.

A lens declares its extractors (ordinary ExtractorSpecs run by the M4.2
executor) and how their items project into the graph (node labels, optional
speaker links). Adding a lens requires only a YAML file and a prompts YAML —
no new Python, event types, or handlers.
"""

import re
from pathlib import Path
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

from src.enrichment.models import ExtractorSpec
from src.utils.helpers import load_yaml

# Node labels are interpolated into Cypher; they must look like identifiers.
_LABEL_RE = re.compile(r"^[A-Z][A-Za-z0-9]*$")


class LensNodeMapping(BaseModel):
    """How one node_type projects into the graph."""

    speaker_link: Optional[Dict[str, str]] = None


class LensExtractorDecl(BaseModel):
    """One focused lens extractor, declared in the lens YAML."""

    name: str
    prompt_key: str
    response_model: str
    scope: Literal["fragment", "utterance", "document"]
    node_type: str
    items_field: str

    def to_extractor_spec(self) -> ExtractorSpec:
        return ExtractorSpec(
            name=self.name,
            prompt_key=self.prompt_key,
            response_model=self.response_model,
            response_module="src.models.lens_responses",
            scope=self.scope,
        )


class LensSpec(BaseModel):
    name: str
    version: int = Field(..., ge=1)
    prompts_file: str
    extractors: List[LensExtractorDecl]
    projects_to: Dict[str, LensNodeMapping]

    @model_validator(mode="after")
    def _validate(self) -> "LensSpec":
        for label in self.projects_to:
            if not _LABEL_RE.match(label):
                raise ValueError(f"Invalid node label: {label!r}")
        for decl in self.extractors:
            if decl.node_type not in self.projects_to:
                raise ValueError(
                    f"Extractor {decl.name!r} references undeclared node_type {decl.node_type!r}"
                )
        return self


def load_lens(name: str, lenses_dir: str = "lenses") -> LensSpec:
    """Load lenses/<name>.yaml; ValueError on unknown name or invalid spec."""
    path = Path(lenses_dir) / f"{name}.yaml"
    try:
        raw = load_yaml(path)
    except FileNotFoundError:
        raise ValueError(f"Unknown lens: {name}")
    return LensSpec.model_validate(raw)
