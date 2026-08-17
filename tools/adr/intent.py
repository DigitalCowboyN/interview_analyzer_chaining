"""Keyword heuristic deciding whether a user prompt reads as an architectural decision in
progress, used by the `context` UserPromptSubmit hook (`tools.adr.__main__`) to gate its
ADR-index nudge to prompts that plausibly need it."""
from __future__ import annotations

import re

_KEYWORDS = (
    "architect", "design", "decision", "trade-off", "tradeoff", "should we",
    "brainstorm", "spec", "approach", "refactor", "adr", "alternative",
)
_PATTERN = re.compile("|".join(re.escape(k) for k in _KEYWORDS), re.IGNORECASE)


def is_architectural(prompt: str) -> bool:
    return bool(_PATTERN.search(prompt or ""))
