"""Tolerant OKF-compatible front-matter parsing for incoming transcripts.

Malformed or non-mapping YAML degrades to "no front matter" with a warning;
ingest never fails on front matter.
"""

import re
from typing import Any, Dict, Optional, Tuple

import yaml

from src.utils.logger import get_logger

logger = get_logger()

_FRONT_MATTER_RE = re.compile(r"\A---\r?\n(.*?)\r?\n---\r?\n", re.DOTALL)


def parse_front_matter(text: str) -> Tuple[Optional[Dict[str, Any]], int]:
    """Return (front_matter, body_start). (None, 0) when absent or invalid."""
    m = _FRONT_MATTER_RE.match(text)
    if not m:
        return None, 0
    try:
        data = yaml.safe_load(m.group(1))
    except yaml.YAMLError as exc:
        logger.warning(f"Malformed front matter ignored ({exc}); treating as body")
        return None, 0
    if not isinstance(data, dict):
        logger.warning("Front matter is not a mapping; treating as body")
        return None, 0
    return data, m.end()
