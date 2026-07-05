import hashlib

from src.ingestion.models import TranscriptFormat
from src.ingestion.normalizer import normalize

LABELED = """Alice: Hi, thanks for joining today. I have a few questions.
Bob: Happy to be here.
Alice: Let's get started.
"""

FLAT = "Well, hey, how are you doing? Are you able to hear me? Yep."


def test_flat_text_fragments_are_offset_grounded():
    result = normalize(FLAT)
    assert result.format == TranscriptFormat.FLAT
    assert len(result.fragments) == 3
    for frag in result.fragments:
        assert FLAT[frag.start_char:frag.end_char] == frag.text
        assert frag.speaker_label is None


def test_labeled_text_parses_speakers_and_grounds_offsets():
    result = normalize(LABELED)
    assert result.format == TranscriptFormat.LABELED
    assert result.speaker_labels == ["Alice", "Bob"]
    # Alice's first line contains two sentences -> two fragments
    alice_frags = [f for f in result.fragments if f.speaker_label == "Alice"]
    assert len(alice_frags) == 3
    for frag in result.fragments:
        assert LABELED[frag.start_char:frag.end_char] == frag.text


def test_sequence_order_is_contiguous_from_zero():
    result = normalize(LABELED)
    assert [f.sequence_order for f in result.fragments] == list(range(len(result.fragments)))


def test_content_hash_is_sha256_of_source():
    result = normalize(FLAT)
    assert result.content_hash == hashlib.sha256(FLAT.encode("utf-8")).hexdigest()


def test_speech_repeating_speaker_prefix_is_grounded_at_speech_position():
    # "A" appears both as the label and as the speech; the span must point at
    # the speech occurrence, not the label.
    text = "A: A.\nB: Bye now.\n"
    result = normalize(text)
    first = result.fragments[0]
    assert first.speaker_label == "A"
    assert first.text == "A."
    assert first.start_char == 3
    assert text[first.start_char:first.end_char] == "A."


def test_continuation_lines_inherit_speaker_and_ground_offsets():
    text = "Alice: Hello there.\nThis continues her thought.\nBob: Sure.\n"
    result = normalize(text)
    continuation = [f for f in result.fragments if f.text == "This continues her thought."]
    assert len(continuation) == 1
    assert continuation[0].speaker_label == "Alice"
    assert text[continuation[0].start_char:continuation[0].end_char] == continuation[0].text
