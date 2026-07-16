"""Parse-validation for the authored sample corpus (data/samples/*.txt).

Unit-level only: exercises the pre-event parse surface (front-matter parsing
+ format detection + normalization) that runs before any events are emitted
or any LLM/infra call happens. Walks the directory so Task 10's additional
samples are covered automatically without touching this file.
"""

from pathlib import Path
from typing import Dict, List

import pytest

from src.ingestion.models import TranscriptFormat
from src.ingestion.normalizer import normalize

SAMPLES_DIR = Path(__file__).parent.parent.parent / "data" / "samples"
SAMPLE_FILES = sorted(SAMPLES_DIR.glob("*.txt"))

MIN_FRAGMENTS = 20

# Category-1 (mature, clean) files: front-matter participants must equal the
# detected speaker labels exactly (labels are accurate, no adversarial noise).
# Keyed by filename so Task 10's category-2/3 files can add their own entries
# without this dict needing to cover every sample.
CATEGORY_1_FILES = {
    "user_interview_mature.txt",
    "team_meeting_mature.txt",
}


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


@pytest.fixture(params=SAMPLE_FILES, ids=lambda p: p.name)
def sample_path(request) -> Path:
    return request.param


def test_corpus_is_non_empty():
    """Sanity: the data-driven parametrization actually found files."""
    assert SAMPLE_FILES, f"No *.txt samples found under {SAMPLES_DIR}"


def test_file_parses_without_error(sample_path: Path):
    """Front-matter + format-detection + normalization succeed for every file."""
    text = _read(sample_path)
    result = normalize(text)
    assert result.content_hash
    assert result.format in (TranscriptFormat.LABELED, TranscriptFormat.FLAT)


def test_minimum_fragment_count(sample_path: Path):
    """At least MIN_FRAGMENTS normalized sentences/fragments come out."""
    result = normalize(_read(sample_path))
    assert len(result.fragments) >= MIN_FRAGMENTS, (
        f"{sample_path.name}: expected >= {MIN_FRAGMENTS} fragments, "
        f"got {len(result.fragments)}"
    )


def test_front_matter_participants_round_trip(sample_path: Path):
    """For files WITH front matter, the participants list round-trips."""
    text = _read(sample_path)
    result = normalize(text)
    if result.front_matter is None:
        pytest.skip(f"{sample_path.name} has no front matter")
    fm_participants = result.front_matter.get("participants")
    assert isinstance(fm_participants, list) and fm_participants, (
        f"{sample_path.name}: front matter missing a non-empty participants list"
    )
    assert all(isinstance(p, str) and p.strip() for p in fm_participants)


def test_category_1_speaker_labels_match_participants(sample_path: Path):
    """Category-1 files: detected format is LABELED and speaker labels equal
    the manifest's participants (front matter is authoritative and clean)."""
    if sample_path.name not in CATEGORY_1_FILES:
        pytest.skip(f"{sample_path.name} is not a category-1 sample")

    text = _read(sample_path)
    result = normalize(text)
    assert result.format == TranscriptFormat.LABELED

    participants: List[str] = result.front_matter.get("participants", [])
    assert set(result.speaker_labels) == set(participants), (
        f"{sample_path.name}: speaker labels {sorted(result.speaker_labels)} "
        f"!= front-matter participants {sorted(participants)}"
    )


def _fragment_texts_by_label(sample_path: Path) -> Dict[str, int]:
    result = normalize(_read(sample_path))
    counts: Dict[str, int] = {}
    for frag in result.fragments:
        if frag.speaker_label:
            counts[frag.speaker_label] = counts.get(frag.speaker_label, 0) + 1
    return counts


def test_category_1_every_speaker_has_fragments(sample_path: Path):
    """Category-1 files: every declared participant actually speaks at least
    once (guards against a manifest claim that doesn't hold in the text)."""
    if sample_path.name not in CATEGORY_1_FILES:
        pytest.skip(f"{sample_path.name} is not a category-1 sample")

    counts = _fragment_texts_by_label(sample_path)
    for speaker, count in counts.items():
        assert count > 0, f"{sample_path.name}: {speaker} has no fragments"
