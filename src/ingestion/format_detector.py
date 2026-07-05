"""Detects whether a transcript has speaker-labeled lines or is flat prose."""

import re

from .models import TranscriptFormat

# A speaker line: short name-like prefix followed by a colon and speech.
SPEAKER_LINE_RE = re.compile(r"^([A-Z][\w .'\-]{0,40}?):\s+(.*)$")

# Minimum fraction of non-empty lines that must match, and minimum distinct labels.
_MIN_MATCH_RATIO = 0.3
_MIN_DISTINCT_LABELS = 2


def detect_format(text: str) -> TranscriptFormat:
    """Classify input as LABELED (speaker-prefixed lines) or FLAT prose."""
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return TranscriptFormat.FLAT

    labels = set()
    matches = 0
    for line in lines:
        m = SPEAKER_LINE_RE.match(line.strip())
        if m:
            matches += 1
            labels.add(m.group(1))

    if matches / len(lines) >= _MIN_MATCH_RATIO and len(labels) >= _MIN_DISTINCT_LABELS:
        return TranscriptFormat.LABELED
    return TranscriptFormat.FLAT
