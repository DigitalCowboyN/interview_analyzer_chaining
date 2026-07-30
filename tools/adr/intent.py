from __future__ import annotations

import re

_KEYWORDS = (
    "architect", "design", "decision", "trade-off", "tradeoff", "should we",
    "brainstorm", "spec", "approach", "refactor", "adr", "alternative",
)
_PATTERN = re.compile("|".join(re.escape(k) for k in _KEYWORDS), re.IGNORECASE)


def is_architectural(prompt: str) -> bool:
    return bool(_PATTERN.search(prompt or ""))
