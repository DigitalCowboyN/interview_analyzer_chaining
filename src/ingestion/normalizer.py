"""Normalizes raw transcript text into offset-grounded fragments.

The source text is never modified; every fragment carries offsets such that
source[start_char:end_char] == fragment.text.
"""

import hashlib
from typing import List, Optional, Tuple

from src.utils.text_processing import segment_text_with_offsets

from .format_detector import SPEAKER_LINE_RE, detect_format
from .models import NormalizedTranscript, RawFragment, TranscriptFormat


def normalize(text: str) -> NormalizedTranscript:
    """Normalize raw transcript text into a NormalizedTranscript."""
    content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    fmt = detect_format(text)

    if fmt == TranscriptFormat.LABELED:
        fragments, labels = _parse_labeled(text)
    else:
        fragments = [
            RawFragment(text=t, start_char=s, end_char=e, sequence_order=i)
            for i, (t, s, e) in enumerate(segment_text_with_offsets(text))
        ]
        labels = []

    return NormalizedTranscript(
        content_hash=content_hash, format=fmt, fragments=fragments, speaker_labels=labels
    )


def _parse_labeled(text: str) -> Tuple[List[RawFragment], List[str]]:
    """Parse speaker-prefixed lines; segment each speech block with offsets."""
    fragments: List[RawFragment] = []
    labels: List[str] = []
    order = 0
    offset = 0  # running offset of the current line within the source text

    current_label: Optional[str] = None
    for line in text.splitlines(keepends=True):
        # Match against the line itself (leading whitespace removed, offset
        # tracked) so match spans give exact source positions. A secondary
        # substring search would mis-ground speech that repeats the speaker
        # prefix (e.g. "A: A").
        lead = len(line) - len(line.lstrip())
        content = line.lstrip().rstrip("\n\r")
        m = SPEAKER_LINE_RE.match(content) if content.strip() else None
        if m:
            current_label = m.group(1)
            if current_label not in labels:
                labels.append(current_label)
            speech_offset = offset + lead + m.start(2)
            _append_segments(fragments, m.group(2), speech_offset, current_label, order)
        elif content.strip():
            # Continuation line of the current speaker's speech
            _append_segments(fragments, content, offset + lead, current_label, order)
        order = len(fragments)
        offset += len(line)

    return fragments, labels


def _append_segments(
    fragments: List[RawFragment],
    speech: str,
    base_offset: int,
    label: Optional[str],
    start_order: int,
) -> None:
    for i, (t, s, e) in enumerate(segment_text_with_offsets(speech)):
        fragments.append(
            RawFragment(
                text=t,
                start_char=base_offset + s,
                end_char=base_offset + e,
                sequence_order=start_order + i,
                speaker_label=label,
            )
        )
