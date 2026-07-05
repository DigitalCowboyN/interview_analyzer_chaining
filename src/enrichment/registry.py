"""Loads the declared extractor set from YAML config."""

from typing import List

from src.utils.helpers import load_yaml

from .models import ExtractorSpec


class ExtractorRegistry:
    """Reads config/extractors.yaml into ordered, enabled ExtractorSpecs."""

    @staticmethod
    def load(path: str = "config/extractors.yaml") -> List[ExtractorSpec]:
        doc = load_yaml(path)
        specs = [ExtractorSpec.model_validate(item) for item in doc["extractors"]]
        return [s for s in specs if s.enabled]
