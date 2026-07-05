from src.ingestion.format_detector import detect_format
from src.ingestion.models import TranscriptFormat

LABELED = """Alice: Hi, thanks for joining today.
Bob: Happy to be here.
Alice: Let's get started with the first question.
Bob: Sure thing.
"""

FLAT = (
    "Well, hey, how are you doing? Are you able to hear me? Yep. Hello? "
    "I can hear you. Can you hear me? Oh, I can hear you. Yes. OK, awesome."
)


def test_labeled_transcript_detected():
    assert detect_format(LABELED) == TranscriptFormat.LABELED


def test_flat_transcript_detected():
    assert detect_format(FLAT) == TranscriptFormat.FLAT


def test_single_colon_line_is_not_labeled():
    text = "Note: this is just a note.\n" + FLAT
    assert detect_format(text) == TranscriptFormat.FLAT


def test_empty_text_is_flat():
    assert detect_format("") == TranscriptFormat.FLAT
