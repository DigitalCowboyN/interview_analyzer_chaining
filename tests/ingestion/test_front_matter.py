import json

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

FM_FLAT_TEXT = """---
title: Flat Notes
participants: [Alice Johnson]
---
This is plain prose without speaker labels. It has several sentences.
Nobody is labeled here at all. The segmenter must still ground offsets.
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


@pytest.mark.parametrize("text", [FM_TEXT, FM_FLAT_TEXT], ids=["labeled", "flat"])
def test_normalize_preserves_offsets_invariant_with_front_matter(text):
    transcript = normalize(text)
    assert transcript.front_matter is not None
    assert len(transcript.fragments) >= 2
    for frag in transcript.fragments:
        assert text[frag.start_char:frag.end_char] == frag.text


def test_normalize_without_front_matter_unchanged():
    text = "Alice: We will go with vendor X.\nBob: Sounds good to me.\n"
    transcript = normalize(text)
    assert transcript.front_matter is None
    for frag in transcript.fragments:
        assert text[frag.start_char:frag.end_char] == frag.text


def test_unquoted_dates_are_normalized_to_json_safe_strings():
    text = (
        "---\n"
        "title: Q3 Vendor Selection\n"
        "date: 2026-07-01\n"
        "milestones:\n"
        "  - name: kickoff\n"
        "    when: 2026-06-01\n"
        "nested:\n"
        "  reviewed: 2026-07-05\n"
        "---\n"
        "Alice: We will go with vendor X.\n"
    )
    fm, _ = parse_front_matter(text)
    json.dumps(fm)  # must not raise TypeError
    assert fm["date"] == "2026-07-01"
    assert fm["milestones"][0]["when"] == "2026-06-01"
    assert fm["nested"]["reviewed"] == "2026-07-05"
