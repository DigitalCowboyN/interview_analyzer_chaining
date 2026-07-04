import pytest
from pydantic import ValidationError

from src.ingestion.models import RawFragment


def test_inverted_offsets_rejected():
    with pytest.raises(ValidationError, match="end_char must be > start_char"):
        RawFragment(text="x", start_char=100, end_char=1, sequence_order=0)


def test_zero_length_span_rejected():
    with pytest.raises(ValidationError, match="end_char must be > start_char"):
        RawFragment(text="", start_char=5, end_char=5, sequence_order=0)


def test_valid_offsets_accepted():
    frag = RawFragment(text="Hi.", start_char=0, end_char=3, sequence_order=0)
    assert frag.end_char > frag.start_char
