import pytest

from src.ingestion.front_matter import parse_front_matter
from src.ingestion.normalizer import normalize

FM_TEXT = """---
title: Q3 Vendor Selection
project: telemetry
date: 2026-07-01
participants: [Alice Johnson, Bob Reyes]
---
Alice: We will go with vendor X.
Bob: Sounds good to me.
"""


def test_parse_extracts_mapping_and_body_offset():
    fm, body_start = parse_front_matter(FM_TEXT)
    assert fm["title"] == "Q3 Vendor Selection"
    assert fm["participants"] == ["Alice Johnson", "Bob Reyes"]
    assert FM_TEXT[body_start:].startswith("Alice: We will go")


def test_no_front_matter_returns_none_and_zero():
    assert parse_front_matter("Alice: Hello there everyone.") == (None, 0)


def test_malformed_yaml_degrades_to_body():
    text = "---\n: not : valid : yaml [\n---\nAlice: Hi there everyone.\n"
    fm, body_start = parse_front_matter(text)
    assert fm is None
    assert body_start == 0  # whole text treated as body


def test_non_mapping_yaml_degrades():
    text = "---\n- just\n- a list\n---\nAlice: Hi there everyone.\n"
    assert parse_front_matter(text) == (None, 0)


def test_normalize_preserves_offsets_invariant_with_front_matter():
    transcript = normalize(FM_TEXT)
    assert transcript.front_matter["project"] == "telemetry"
    assert len(transcript.fragments) >= 2
    for frag in transcript.fragments:
        assert FM_TEXT[frag.start_char:frag.end_char] == frag.text  # THE invariant


def test_normalize_without_front_matter_unchanged():
    text = "Alice: We will go with vendor X.\nBob: Sounds good to me.\n"
    transcript = normalize(text)
    assert transcript.front_matter is None
    for frag in transcript.fragments:
        assert text[frag.start_char:frag.end_char] == frag.text
