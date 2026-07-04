# Layer 1: Ingestion, Map, Speaker Genesis & Stitching — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ingest raw, possibly speaker-unlabeled transcripts into the event-sourced system with character-offset grounding, inferred (correctable) speakers, and utterance stitching that overlays — never rewrites — the verbatim fragment sequence.

**Architecture:** A new `src/ingestion/` package normalizes any input into offset-grounded fragments, runs LLM speaker inference and stitching passes, and emits domain events through the existing repository/aggregate path (Interview aggregate owns speakers and utterances; Sentence aggregate owns fragments and their attribution). New projection handlers materialize Speaker and Utterance nodes in Neo4j. A new API router exposes user corrections, which follow the existing edit-protection pattern (human events lock fields against system regeneration).

**Tech Stack:** Python 3.10, Pydantic v2, spaCy (`en_core_web_sm`), EventStoreDB via existing `EventStoreClient`/repositories, Neo4j via existing projection service, FastAPI, pytest (asyncio).

**Spec:** `docs/superpowers/specs/2026-07-04-mine-layers-design.md` (Layer 1 section).

## Global Constraints

- The verbatim fragment sequence is immutable: no task may reorder, merge, or rewrite fragment text. Stitching emits relationship data only.
- Every fragment carries `start_char`/`end_char` offsets into the immutable source text; `source_text[start_char:end_char] == fragment_text` must always hold.
- All inference results carry numeric confidence in `[0.0, 1.0]` and `actor_type=SYSTEM`; all user corrections carry `actor_type=HUMAN` and survive regeneration (system passes must not overwrite human-corrected fields).
- Low-confidence results are committed with their confidence values, never parked or dropped.
- Deterministic IDs: fragment/sentence UUID = `uuid5(NAMESPACE_DNS, f"{interview_id}:{index}")` (existing convention); speaker UUID = `uuid5(NAMESPACE_DNS, f"{interview_id}:speaker:{handle}")`; utterance UUID = `uuid5(NAMESPACE_DNS, f"{interview_id}:utterance:{ordinal}")`.
- Follow existing patterns exactly: Pydantic `*Data` payload models + aggregate command methods (`src/events/`), `BaseProjectionHandler` subclasses registered in `src/projections/bootstrap.py`, prompts in YAML under `prompts/`, response models validated with Pydantic.
- Unit tests must not require API keys or infrastructure: mock the `agent` singleton. Integration tests get `@pytest.mark.integration`.
- Run unit tests with: `python -m pytest tests -m "not integration" -q`. Style: `black`, `flake8` (line length per existing config).

---

### Task 1: Offset-preserving segmentation

**Files:**
- Modify: `src/utils/text_processing.py`
- Test: `tests/utils/test_text_processing.py`

**Interfaces:**
- Consumes: existing module-level `nlp` spaCy model in `text_processing.py`.
- Produces: `segment_text_with_offsets(text: str) -> List[Tuple[str, int, int]]` — list of `(fragment_text, start_char, end_char)` where `text[start_char:end_char] == fragment_text`. Later tasks (normalizer) depend on this exact signature.

- [ ] **Step 1: Write the failing tests** (append to `tests/utils/test_text_processing.py`)

```python
from src.utils.text_processing import segment_text_with_offsets


class TestSegmentTextWithOffsets:
    def test_offsets_recover_exact_text(self):
        text = "Well, hey, how are you doing?  Are you able to hear me? Yep."
        fragments = segment_text_with_offsets(text)
        assert len(fragments) == 3
        for frag_text, start, end in fragments:
            assert text[start:end] == frag_text

    def test_empty_text_returns_empty_list(self):
        assert segment_text_with_offsets("") == []

    def test_whitespace_is_stripped_but_offsets_point_at_content(self):
        text = "  Hello there.   How are you?  "
        fragments = segment_text_with_offsets(text)
        assert fragments[0][0] == "Hello there."
        assert text[fragments[0][1]:fragments[0][2]] == "Hello there."
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/utils/test_text_processing.py::TestSegmentTextWithOffsets -v`
Expected: FAIL with `ImportError: cannot import name 'segment_text_with_offsets'`

- [ ] **Step 3: Implement** (append to `src/utils/text_processing.py`)

```python
def segment_text_with_offsets(text: str) -> List[Tuple[str, int, int]]:
    """
    Segment text into fragments with character offsets into the source.

    Returns a list of (fragment_text, start_char, end_char) tuples where
    text[start_char:end_char] == fragment_text. Whitespace is stripped from
    fragments; offsets point at the stripped content.
    """
    if nlp is None:
        logger.error("spaCy model not loaded. Cannot segment text.")
        return []
    if not text:
        return []

    doc = nlp(text)
    fragments: List[Tuple[str, int, int]] = []
    for sent in doc.sents:
        stripped = sent.text.strip()
        if not stripped:
            continue
        leading_ws = len(sent.text) - len(sent.text.lstrip())
        start = sent.start_char + leading_ws
        fragments.append((stripped, start, start + len(stripped)))
    return fragments
```

Also add `Tuple` to the existing `from typing import List` import.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/utils/test_text_processing.py::TestSegmentTextWithOffsets -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/utils/text_processing.py tests/utils/test_text_processing.py
git commit -m "feat: add offset-preserving segmentation for Layer 1 map grounding"
```

---

### Task 2: Ingestion models and format detector

**Files:**
- Create: `src/ingestion/__init__.py` (empty)
- Create: `src/ingestion/models.py`
- Create: `src/ingestion/format_detector.py`
- Test: `tests/ingestion/__init__.py` (empty), `tests/ingestion/test_format_detector.py`

**Interfaces:**
- Produces (in `models.py`):
  - `TranscriptFormat(str, Enum)` with members `LABELED`, `FLAT`
  - `RawFragment(BaseModel)`: `text: str`, `start_char: int`, `end_char: int`, `sequence_order: int`, `speaker_label: Optional[str] = None`
  - `NormalizedTranscript(BaseModel)`: `content_hash: str`, `format: TranscriptFormat`, `fragments: List[RawFragment]`, `speaker_labels: List[str]`
- Produces (in `format_detector.py`): `detect_format(text: str) -> TranscriptFormat` and module constant `SPEAKER_LINE_RE` (used again by the normalizer in Task 3).

- [ ] **Step 1: Write the failing tests** (`tests/ingestion/test_format_detector.py`)

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/ingestion/test_format_detector.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.ingestion'`

- [ ] **Step 3: Implement**

`src/ingestion/models.py`:

```python
"""Data models for Layer 1 ingestion (format detection and normalization)."""

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class TranscriptFormat(str, Enum):
    """Detected input transcript format."""

    LABELED = "labeled"  # speaker-prefixed lines, e.g. "Alice: ..."
    FLAT = "flat"  # undifferentiated prose, no speaker markers


class RawFragment(BaseModel):
    """A contiguous run of speech, verbatim, grounded in the source text."""

    text: str = Field(..., description="Verbatim fragment text (stripped)")
    start_char: int = Field(..., ge=0, description="Offset into source text")
    end_char: int = Field(..., gt=0, description="End offset into source text")
    sequence_order: int = Field(..., ge=0, description="As-spoken order, immutable")
    speaker_label: Optional[str] = Field(
        None, description="Speaker label parsed from the source, if the format had one"
    )


class NormalizedTranscript(BaseModel):
    """Result of normalizing a raw input into offset-grounded fragments."""

    content_hash: str = Field(..., description="sha256 hex digest of the source text")
    format: TranscriptFormat
    fragments: List[RawFragment]
    speaker_labels: List[str] = Field(
        default_factory=list, description="Distinct parsed labels, in order of first appearance"
    )
```

`src/ingestion/format_detector.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/ingestion/test_format_detector.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/ingestion tests/ingestion
git commit -m "feat: add ingestion models and transcript format detector"
```

---

### Task 3: Normalizer (any input → offset-grounded fragments)

**Files:**
- Create: `src/ingestion/normalizer.py`
- Test: `tests/ingestion/test_normalizer.py`

**Interfaces:**
- Consumes: `segment_text_with_offsets` (Task 1), `SPEAKER_LINE_RE`, models (Task 2).
- Produces: `normalize(text: str) -> NormalizedTranscript`. **The offsets invariant** — for every fragment, `text[start_char:end_char] == fragment.text` — is what all later grounding relies on.

- [ ] **Step 1: Write the failing tests** (`tests/ingestion/test_normalizer.py`)

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/ingestion/test_normalizer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.ingestion.normalizer'`

- [ ] **Step 3: Implement** (`src/ingestion/normalizer.py`)

```python
"""Normalizes raw transcript text into offset-grounded fragments.

The source text is never modified; every fragment carries offsets such that
source[start_char:end_char] == fragment.text.
"""

import hashlib
from typing import List

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


def _parse_labeled(text: str) -> tuple:
    """Parse speaker-prefixed lines; segment each speech block with offsets."""
    fragments: List[RawFragment] = []
    labels: List[str] = []
    order = 0
    offset = 0  # running offset of the current line within the source text

    current_label = None
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        m = SPEAKER_LINE_RE.match(stripped) if stripped else None
        if m:
            current_label = m.group(1)
            if current_label not in labels:
                labels.append(current_label)
            # Offset of the speech portion within the source line
            speech = m.group(2)
            speech_offset = offset + line.index(speech)
            _append_segments(fragments, speech, speech_offset, current_label, order)
        elif stripped:
            # Continuation line of the current speaker's speech
            line_offset = offset + line.index(stripped[0])
            _append_segments(fragments, stripped, line_offset, current_label, order)
        order = len(fragments)
        offset += len(line)

    return fragments, labels


def _append_segments(
    fragments: List[RawFragment], speech: str, base_offset: int, label, start_order: int
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/ingestion/test_normalizer.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/normalizer.py tests/ingestion/test_normalizer.py
git commit -m "feat: add transcript normalizer producing offset-grounded fragments"
```

---

### Task 4: Sentence aggregate — offsets and speaker attribution events

**Files:**
- Modify: `src/events/sentence_events.py`
- Modify: `src/events/aggregates.py` (Sentence class)
- Test: `tests/events/test_sentence_speaker_events.py`

**Interfaces:**
- Consumes: existing `Sentence(AggregateRoot)`, `EventEnvelope`, `_add_event`.
- Produces:
  - `SentenceCreatedData` gains `start_char: Optional[int] = None`, `end_char: Optional[int] = None`; `Sentence.create(...)` and `create_sentence_created_event(...)` gain the same two keyword params (backward compatible).
  - New payload models in `sentence_events.py`: `SpeakerAttributedData(speaker_id: str, confidence: float, method: str)`, `SpeakerReattributedData(old_speaker_id: Optional[str], new_speaker_id: str)`.
  - New `Sentence` methods: `attribute_speaker(speaker_id: str, confidence: float, method: str, **envelope_kwargs) -> EventEnvelope` (event type `"SpeakerAttributed"`; raises `ValueError` if `self.speaker_locked`), `reattribute_speaker(new_speaker_id: str, **envelope_kwargs) -> EventEnvelope` (event type `"SpeakerReattributed"`; sets `speaker_locked = True`).
  - New `Sentence` state: `speaker_id: Optional[str]`, `speaker_confidence: Optional[float]`, `speaker_locked: bool = False`, `start_char: Optional[int]`, `end_char: Optional[int]`.

- [ ] **Step 1: Write the failing tests** (`tests/events/test_sentence_speaker_events.py`)

```python
import pytest

from src.events.aggregates import Sentence


def make_sentence() -> Sentence:
    s = Sentence("11111111-1111-1111-1111-111111111111")
    s.create(
        interview_id="22222222-2222-2222-2222-222222222222",
        index=0,
        text="Can you hear me?",
        start_char=10,
        end_char=26,
    )
    return s


def test_create_stores_offsets():
    s = make_sentence()
    assert s.start_char == 10
    assert s.end_char == 26
    event = s.get_uncommitted_events()[0]
    assert event.data["start_char"] == 10
    assert event.data["end_char"] == 26


def test_attribute_speaker_sets_state_and_event():
    s = make_sentence()
    event = s.attribute_speaker(
        speaker_id="33333333-3333-3333-3333-333333333333", confidence=0.72, method="inference"
    )
    assert event.event_type == "SpeakerAttributed"
    assert s.speaker_id == "33333333-3333-3333-3333-333333333333"
    assert s.speaker_confidence == 0.72
    assert s.speaker_locked is False


def test_reattribute_speaker_locks_against_system_overwrite():
    s = make_sentence()
    s.attribute_speaker("33333333-3333-3333-3333-333333333333", 0.72, "inference")
    event = s.reattribute_speaker("44444444-4444-4444-4444-444444444444")
    assert event.event_type == "SpeakerReattributed"
    assert event.data["old_speaker_id"] == "33333333-3333-3333-3333-333333333333"
    assert s.speaker_locked is True
    with pytest.raises(ValueError, match="locked"):
        s.attribute_speaker("33333333-3333-3333-3333-333333333333", 0.9, "inference")


def test_attribute_speaker_requires_created_sentence():
    s = Sentence("11111111-1111-1111-1111-111111111111")
    with pytest.raises(ValueError):
        s.attribute_speaker("33333333-3333-3333-3333-333333333333", 0.5, "inference")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/events/test_sentence_speaker_events.py -v`
Expected: FAIL — `TypeError: Sentence.create() got an unexpected keyword argument 'start_char'`

- [ ] **Step 3: Implement**

In `src/events/sentence_events.py`, add to `SentenceCreatedData`:

```python
    start_char: Optional[int] = Field(None, ge=0, description="Offset into immutable source text")
    end_char: Optional[int] = Field(None, gt=0, description="End offset into immutable source text")
```

Add new payload models after `SentenceDeletedData`:

```python
class SpeakerAttributedData(BaseModel):
    """Data payload for SpeakerAttributed event (system inference or parsed label)."""

    speaker_id: str = Field(..., description="UUID of the attributed Speaker")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Attribution confidence")
    method: str = Field(..., description="How attribution was made: 'parsed' | 'inference'")


class SpeakerReattributedData(BaseModel):
    """Data payload for SpeakerReattributed event (human correction; locks attribution)."""

    old_speaker_id: Optional[str] = Field(None, description="Previously attributed Speaker UUID")
    new_speaker_id: str = Field(..., description="Corrected Speaker UUID")
```

Update `create_sentence_created_event` to accept and pass `start_char: Optional[int] = None, end_char: Optional[int] = None` into `SentenceCreatedData`.

In `src/events/aggregates.py`, `Sentence.__init__`, add state fields:

```python
        self.start_char: Optional[int] = None
        self.end_char: Optional[int] = None
        self.speaker_id: Optional[str] = None
        self.speaker_confidence: Optional[float] = None
        self.speaker_locked: bool = False
```

In `Sentence.apply_event`, add dispatch branches:

```python
        elif event.event_type == "SpeakerAttributed":
            self._apply_speaker_attributed(event)
        elif event.event_type == "SpeakerReattributed":
            self._apply_speaker_reattributed(event)
```

Apply methods:

```python
    def _apply_speaker_attributed(self, event: EventEnvelope) -> None:
        """Apply SpeakerAttributed event."""
        self.speaker_id = event.data.get("speaker_id")
        self.speaker_confidence = event.data.get("confidence")
        self.updated_at = event.occurred_at

    def _apply_speaker_reattributed(self, event: EventEnvelope) -> None:
        """Apply SpeakerReattributed event (human correction locks attribution)."""
        self.speaker_id = event.data.get("new_speaker_id")
        self.speaker_confidence = 1.0
        self.speaker_locked = True
        self.updated_at = event.occurred_at
```

In `_apply_sentence_created`, add:

```python
        self.start_char = data.get("start_char")
        self.end_char = data.get("end_char")
```

Command methods (after `edit`):

```python
    def attribute_speaker(
        self, speaker_id: str, confidence: float, method: str, **envelope_kwargs
    ) -> EventEnvelope:
        """Attribute this fragment to a speaker (system inference or parsed label)."""
        if self.version < 0:
            raise ValueError("Sentence must be created before attributing a speaker")
        if self.speaker_locked:
            raise ValueError("Speaker attribution is locked by a human correction")

        return self._add_event(
            event_type="SpeakerAttributed",
            data={"speaker_id": speaker_id, "confidence": confidence, "method": method},
            **envelope_kwargs,
        )

    def reattribute_speaker(self, new_speaker_id: str, **envelope_kwargs) -> EventEnvelope:
        """Human correction of speaker attribution; locks against system overwrite."""
        if self.version < 0:
            raise ValueError("Sentence must be created before reattributing a speaker")

        return self._add_event(
            event_type="SpeakerReattributed",
            data={"old_speaker_id": self.speaker_id, "new_speaker_id": new_speaker_id},
            **envelope_kwargs,
        )
```

Update `Sentence.create(...)` to accept `start_char: Optional[int] = None, end_char: Optional[int] = None` and include both in the event `data` dict.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/events/test_sentence_speaker_events.py tests/events -m "not integration" -q`
Expected: new tests PASS; no regressions in existing `tests/events` suite.

- [ ] **Step 5: Commit**

```bash
git add src/events/sentence_events.py src/events/aggregates.py tests/events/test_sentence_speaker_events.py
git commit -m "feat: fragment offsets and correctable speaker attribution on Sentence aggregate"
```

---

### Task 5: Interview aggregate — speaker lifecycle events

**Files:**
- Modify: `src/events/interview_events.py`
- Modify: `src/events/aggregates.py` (Interview class)
- Test: `tests/events/test_interview_speaker_events.py`

**Interfaces:**
- Produces:
  - Payload models in `interview_events.py`: `SpeakerCreatedData(speaker_id, handle, display_name, provisional, confidence, method)`, `SpeakerRenamedData(speaker_id, old_display_name, new_display_name)`, `SpeakerMergedData(surviving_speaker_id, merged_speaker_id)`.
  - `Interview` state: `speakers: Dict[str, Dict[str, Any]]` keyed by `speaker_id` with keys `handle`, `display_name`, `provisional`, `merged_into` (None unless merged).
  - `Interview` methods: `add_speaker(speaker_id, handle, display_name, provisional, confidence, method, **kw)` (event `"SpeakerCreated"`), `rename_speaker(speaker_id, new_display_name, **kw)` (event `"SpeakerRenamed"`; marks `provisional=False`), `merge_speakers(surviving_speaker_id, merged_speaker_id, **kw)` (event `"SpeakerMerged"`). Guards raise `ValueError` for: uncreated interview, unknown/duplicate speaker ids, merging a speaker into itself, operating on an already-merged speaker.

- [ ] **Step 1: Write the failing tests** (`tests/events/test_interview_speaker_events.py`)

```python
import pytest

from src.events.aggregates import Interview

IID = "22222222-2222-2222-2222-222222222222"
SP1 = "33333333-3333-3333-3333-333333333333"
SP2 = "44444444-4444-4444-4444-444444444444"


def make_interview() -> Interview:
    interview = Interview(IID)
    interview.create(title="test.txt", source="data/input/test.txt")
    return interview


def test_add_speaker_records_provisional_speaker():
    i = make_interview()
    event = i.add_speaker(SP1, handle="S1", display_name="S1", provisional=True,
                          confidence=0.8, method="inference")
    assert event.event_type == "SpeakerCreated"
    assert i.speakers[SP1]["handle"] == "S1"
    assert i.speakers[SP1]["provisional"] is True


def test_duplicate_speaker_id_rejected():
    i = make_interview()
    i.add_speaker(SP1, "S1", "S1", True, 0.8, "inference")
    with pytest.raises(ValueError, match="already exists"):
        i.add_speaker(SP1, "S1", "S1", True, 0.8, "inference")


def test_rename_speaker_clears_provisional_flag():
    i = make_interview()
    i.add_speaker(SP1, "S1", "S1", True, 0.8, "inference")
    event = i.rename_speaker(SP1, "Dana the ECU owner")
    assert event.event_type == "SpeakerRenamed"
    assert i.speakers[SP1]["display_name"] == "Dana the ECU owner"
    assert i.speakers[SP1]["provisional"] is False


def test_merge_speakers_marks_merged_into():
    i = make_interview()
    i.add_speaker(SP1, "S1", "S1", True, 0.8, "inference")
    i.add_speaker(SP2, "S2", "S2", True, 0.6, "inference")
    event = i.merge_speakers(surviving_speaker_id=SP1, merged_speaker_id=SP2)
    assert event.event_type == "SpeakerMerged"
    assert i.speakers[SP2]["merged_into"] == SP1


def test_merge_into_self_rejected():
    i = make_interview()
    i.add_speaker(SP1, "S1", "S1", True, 0.8, "inference")
    with pytest.raises(ValueError, match="itself"):
        i.merge_speakers(SP1, SP1)


def test_rename_unknown_speaker_rejected():
    i = make_interview()
    with pytest.raises(ValueError, match="Unknown speaker"):
        i.rename_speaker(SP1, "Nobody")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/events/test_interview_speaker_events.py -v`
Expected: FAIL — `AttributeError: 'Interview' object has no attribute 'add_speaker'`

- [ ] **Step 3: Implement**

In `src/events/interview_events.py`, add payload models:

```python
class SpeakerCreatedData(BaseModel):
    """Data payload for SpeakerCreated event."""

    speaker_id: str = Field(..., description="Deterministic UUID of the speaker")
    handle: str = Field(..., description="Stable short handle, e.g. 'S1' or parsed label")
    display_name: str = Field(..., description="Human-readable name (initially the handle)")
    provisional: bool = Field(..., description="True when inferred rather than confirmed")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Inference confidence")
    method: str = Field(..., description="'parsed' | 'inference'")


class SpeakerRenamedData(BaseModel):
    """Data payload for SpeakerRenamed event (human correction)."""

    speaker_id: str = Field(..., description="UUID of the speaker")
    old_display_name: str = Field(..., description="Previous display name")
    new_display_name: str = Field(..., description="New display name")


class SpeakerMergedData(BaseModel):
    """Data payload for SpeakerMerged event (human correction: two handles, one person)."""

    surviving_speaker_id: str = Field(..., description="Speaker that remains")
    merged_speaker_id: str = Field(..., description="Speaker merged away")
```

In `src/events/aggregates.py`, `Interview.__init__`, add:

```python
        self.speakers: Dict[str, Dict[str, Any]] = {}
```

In `Interview.apply_event`, add branches:

```python
        elif event.event_type == "SpeakerCreated":
            self._apply_speaker_created(event)
        elif event.event_type == "SpeakerRenamed":
            self._apply_speaker_renamed(event)
        elif event.event_type == "SpeakerMerged":
            self._apply_speaker_merged(event)
```

Apply methods:

```python
    def _apply_speaker_created(self, event: EventEnvelope) -> None:
        data = event.data
        self.speakers[data["speaker_id"]] = {
            "handle": data["handle"],
            "display_name": data["display_name"],
            "provisional": data["provisional"],
            "merged_into": None,
        }
        self.updated_at = event.occurred_at

    def _apply_speaker_renamed(self, event: EventEnvelope) -> None:
        data = event.data
        speaker = self.speakers[data["speaker_id"]]
        speaker["display_name"] = data["new_display_name"]
        speaker["provisional"] = False
        self.updated_at = event.occurred_at

    def _apply_speaker_merged(self, event: EventEnvelope) -> None:
        data = event.data
        self.speakers[data["merged_speaker_id"]]["merged_into"] = data["surviving_speaker_id"]
        self.updated_at = event.occurred_at
```

Command methods:

```python
    def add_speaker(
        self,
        speaker_id: str,
        handle: str,
        display_name: str,
        provisional: bool,
        confidence: float,
        method: str,
        **envelope_kwargs,
    ) -> EventEnvelope:
        """Register a speaker discovered by parsing or inference."""
        if self.version < 0:
            raise ValueError("Interview must be created before adding speakers")
        if speaker_id in self.speakers:
            raise ValueError(f"Speaker {speaker_id} already exists")

        return self._add_event(
            event_type="SpeakerCreated",
            data={
                "speaker_id": speaker_id,
                "handle": handle,
                "display_name": display_name,
                "provisional": provisional,
                "confidence": confidence,
                "method": method,
            },
            **envelope_kwargs,
        )

    def rename_speaker(self, speaker_id: str, new_display_name: str, **envelope_kwargs) -> EventEnvelope:
        """Human correction: give a provisional speaker a real name."""
        if speaker_id not in self.speakers:
            raise ValueError(f"Unknown speaker: {speaker_id}")

        return self._add_event(
            event_type="SpeakerRenamed",
            data={
                "speaker_id": speaker_id,
                "old_display_name": self.speakers[speaker_id]["display_name"],
                "new_display_name": new_display_name,
            },
            **envelope_kwargs,
        )

    def merge_speakers(
        self, surviving_speaker_id: str, merged_speaker_id: str, **envelope_kwargs
    ) -> EventEnvelope:
        """Human correction: two provisional handles were the same person."""
        if surviving_speaker_id == merged_speaker_id:
            raise ValueError("Cannot merge a speaker into itself")
        for sid in (surviving_speaker_id, merged_speaker_id):
            if sid not in self.speakers:
                raise ValueError(f"Unknown speaker: {sid}")
            if self.speakers[sid]["merged_into"]:
                raise ValueError(f"Speaker {sid} has already been merged")

        return self._add_event(
            event_type="SpeakerMerged",
            data={
                "surviving_speaker_id": surviving_speaker_id,
                "merged_speaker_id": merged_speaker_id,
            },
            **envelope_kwargs,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/events -m "not integration" -q`
Expected: new tests PASS; no regressions.

- [ ] **Step 5: Commit**

```bash
git add src/events/interview_events.py src/events/aggregates.py tests/events/test_interview_speaker_events.py
git commit -m "feat: speaker lifecycle events on Interview aggregate"
```

---

### Task 6: Interview aggregate — utterance stitching events

**Files:**
- Modify: `src/events/interview_events.py`
- Modify: `src/events/aggregates.py` (Interview class)
- Test: `tests/events/test_interview_stitch_events.py`

**Interfaces:**
- Produces:
  - Payload models: `UtteranceIdentifiedData(utterance_id: str, speaker_id: str, fragment_ids: List[str], confidence: float)`, `InterruptionRecordedData(interrupting_utterance_id: str, interrupted_utterance_id: str, at_fragment_id: str)`, `StitchRemovedData(utterance_id: str, reason: Optional[str])`.
  - `Interview` state: `utterances: Dict[str, Dict[str, Any]]` keyed by `utterance_id` with keys `speaker_id`, `fragment_ids`, `removed` (bool).
  - `Interview` methods: `identify_utterance(utterance_id, speaker_id, fragment_ids, confidence, **kw)` (event `"UtteranceIdentified"`; guards: speaker exists, `fragment_ids` non-empty, utterance_id unique), `record_interruption(interrupting_utterance_id, interrupted_utterance_id, at_fragment_id, **kw)` (event `"InterruptionRecorded"`; both utterances must exist), `remove_stitch(utterance_id, reason=None, **kw)` (event `"StitchRemoved"`; human correction).

- [ ] **Step 1: Write the failing tests** (`tests/events/test_interview_stitch_events.py`)

```python
import pytest

from src.events.aggregates import Interview

IID = "22222222-2222-2222-2222-222222222222"
SP1 = "33333333-3333-3333-3333-333333333333"
U1 = "55555555-5555-5555-5555-555555555555"
U2 = "66666666-6666-6666-6666-666666666666"
FRAGS = ["77777777-7777-7777-7777-777777777771", "77777777-7777-7777-7777-777777777772"]


def make_interview_with_speaker() -> Interview:
    i = Interview(IID)
    i.create(title="test.txt", source="data/input/test.txt")
    i.add_speaker(SP1, "S1", "S1", True, 0.8, "inference")
    return i


def test_identify_utterance_records_fragments_in_order():
    i = make_interview_with_speaker()
    event = i.identify_utterance(U1, SP1, FRAGS, confidence=0.75)
    assert event.event_type == "UtteranceIdentified"
    assert i.utterances[U1]["fragment_ids"] == FRAGS


def test_identify_utterance_requires_known_speaker():
    i = make_interview_with_speaker()
    with pytest.raises(ValueError, match="Unknown speaker"):
        i.identify_utterance(U1, "99999999-9999-9999-9999-999999999999", FRAGS, 0.5)


def test_identify_utterance_requires_fragments():
    i = make_interview_with_speaker()
    with pytest.raises(ValueError, match="at least one fragment"):
        i.identify_utterance(U1, SP1, [], 0.5)


def test_record_interruption_requires_both_utterances():
    i = make_interview_with_speaker()
    i.identify_utterance(U1, SP1, FRAGS, 0.75)
    with pytest.raises(ValueError, match="Unknown utterance"):
        i.record_interruption(U2, U1, FRAGS[0])


def test_record_interruption_and_remove_stitch():
    i = make_interview_with_speaker()
    i.identify_utterance(U1, SP1, [FRAGS[0]], 0.75)
    i.identify_utterance(U2, SP1, [FRAGS[1]], 0.6)
    event = i.record_interruption(U2, U1, FRAGS[1])
    assert event.event_type == "InterruptionRecorded"
    removed = i.remove_stitch(U2, reason="not actually a continuation")
    assert removed.event_type == "StitchRemoved"
    assert i.utterances[U2]["removed"] is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/events/test_interview_stitch_events.py -v`
Expected: FAIL — `AttributeError: 'Interview' object has no attribute 'identify_utterance'`

- [ ] **Step 3: Implement**

In `src/events/interview_events.py`:

```python
class UtteranceIdentifiedData(BaseModel):
    """Data payload for UtteranceIdentified event (stitching overlay)."""

    utterance_id: str = Field(..., description="Deterministic UUID of the utterance")
    speaker_id: str = Field(..., description="Speaker whose continuous thought this is")
    fragment_ids: List[str] = Field(..., description="Ordered fragment UUIDs composing the utterance")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Stitching confidence")


class InterruptionRecordedData(BaseModel):
    """Data payload for InterruptionRecorded event."""

    interrupting_utterance_id: str = Field(..., description="Utterance that broke in")
    interrupted_utterance_id: str = Field(..., description="Utterance that was broken into")
    at_fragment_id: str = Field(..., description="First fragment of the interruption")


class StitchRemovedData(BaseModel):
    """Data payload for StitchRemoved event (human correction)."""

    utterance_id: str = Field(..., description="Utterance whose stitch is removed")
    reason: Optional[str] = Field(None, description="Why the stitch was wrong")
```

(Add `List` to the typing import if missing.)

In `src/events/aggregates.py`, `Interview.__init__`:

```python
        self.utterances: Dict[str, Dict[str, Any]] = {}
```

Dispatch branches in `Interview.apply_event`:

```python
        elif event.event_type == "UtteranceIdentified":
            self._apply_utterance_identified(event)
        elif event.event_type == "InterruptionRecorded":
            self._apply_interruption_recorded(event)
        elif event.event_type == "StitchRemoved":
            self._apply_stitch_removed(event)
```

Apply methods:

```python
    def _apply_utterance_identified(self, event: EventEnvelope) -> None:
        data = event.data
        self.utterances[data["utterance_id"]] = {
            "speaker_id": data["speaker_id"],
            "fragment_ids": list(data["fragment_ids"]),
            "removed": False,
        }
        self.updated_at = event.occurred_at

    def _apply_interruption_recorded(self, event: EventEnvelope) -> None:
        self.updated_at = event.occurred_at

    def _apply_stitch_removed(self, event: EventEnvelope) -> None:
        self.utterances[event.data["utterance_id"]]["removed"] = True
        self.updated_at = event.occurred_at
```

Command methods:

```python
    def identify_utterance(
        self,
        utterance_id: str,
        speaker_id: str,
        fragment_ids: List[str],
        confidence: float,
        **envelope_kwargs,
    ) -> EventEnvelope:
        """Record a stitched utterance: one speaker's continuous thought (overlay only)."""
        if self.version < 0:
            raise ValueError("Interview must be created before identifying utterances")
        if speaker_id not in self.speakers:
            raise ValueError(f"Unknown speaker: {speaker_id}")
        if not fragment_ids:
            raise ValueError("Utterance requires at least one fragment")
        if utterance_id in self.utterances:
            raise ValueError(f"Utterance {utterance_id} already identified")

        return self._add_event(
            event_type="UtteranceIdentified",
            data={
                "utterance_id": utterance_id,
                "speaker_id": speaker_id,
                "fragment_ids": fragment_ids,
                "confidence": confidence,
            },
            **envelope_kwargs,
        )

    def record_interruption(
        self,
        interrupting_utterance_id: str,
        interrupted_utterance_id: str,
        at_fragment_id: str,
        **envelope_kwargs,
    ) -> EventEnvelope:
        """Record that one utterance broke into another."""
        for uid in (interrupting_utterance_id, interrupted_utterance_id):
            if uid not in self.utterances:
                raise ValueError(f"Unknown utterance: {uid}")

        return self._add_event(
            event_type="InterruptionRecorded",
            data={
                "interrupting_utterance_id": interrupting_utterance_id,
                "interrupted_utterance_id": interrupted_utterance_id,
                "at_fragment_id": at_fragment_id,
            },
            **envelope_kwargs,
        )

    def remove_stitch(
        self, utterance_id: str, reason: Optional[str] = None, **envelope_kwargs
    ) -> EventEnvelope:
        """Human correction: an identified utterance was wrong; remove the overlay."""
        if utterance_id not in self.utterances:
            raise ValueError(f"Unknown utterance: {utterance_id}")

        return self._add_event(
            event_type="StitchRemoved",
            data={"utterance_id": utterance_id, "reason": reason},
            **envelope_kwargs,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/events -m "not integration" -q`
Expected: PASS, no regressions.

- [ ] **Step 5: Commit**

```bash
git add src/events/interview_events.py src/events/aggregates.py tests/events/test_interview_stitch_events.py
git commit -m "feat: utterance stitching events on Interview aggregate"
```

---

### Task 7: Speaker inference — prompts and response models

**Files:**
- Create: `prompts/ingestion_prompts.yaml`
- Create: `src/models/ingestion_responses.py`
- Test: `tests/models/__init__.py` (create empty if missing), `tests/models/test_ingestion_responses.py`

**Interfaces:**
- Produces:
  - `SpeakerAssignment(BaseModel)`: `index: int`, `speaker: str`, `confidence: float (0..1)`
  - `SpeakerWindowResponse(BaseModel)`: `assignments: List[SpeakerAssignment]`
  - `UtteranceProposal(BaseModel)`: `speaker: str`, `fragment_indices: List[int]`, `confidence: float`
  - `InterruptionProposal(BaseModel)`: `interrupting: int`, `interrupted: int`, `at_index: int` (indices into the proposal list / fragment sequence)
  - `StitchWindowResponse(BaseModel)`: `utterances: List[UtteranceProposal]`, `interruptions: List[InterruptionProposal]`
  - Prompt keys in `prompts/ingestion_prompts.yaml`: `speaker_window` (placeholder `{fragments}`), `stitch_window` (placeholders `{fragments}` — numbered fragments each prefixed with its assigned speaker handle).

- [ ] **Step 1: Write the failing tests** (`tests/models/test_ingestion_responses.py`)

```python
import pytest
from pydantic import ValidationError

from src.models.ingestion_responses import (
    SpeakerWindowResponse,
    StitchWindowResponse,
)


def test_speaker_window_response_validates():
    resp = SpeakerWindowResponse.model_validate(
        {"assignments": [{"index": 0, "speaker": "S1", "confidence": 0.9}]}
    )
    assert resp.assignments[0].speaker == "S1"


def test_confidence_out_of_range_rejected():
    with pytest.raises(ValidationError):
        SpeakerWindowResponse.model_validate(
            {"assignments": [{"index": 0, "speaker": "S1", "confidence": 1.5}]}
        )


def test_stitch_window_response_validates():
    resp = StitchWindowResponse.model_validate(
        {
            "utterances": [{"speaker": "S1", "fragment_indices": [0, 2], "confidence": 0.7}],
            "interruptions": [{"interrupting": 1, "interrupted": 0, "at_index": 1}],
        }
    )
    assert resp.utterances[0].fragment_indices == [0, 2]
    assert resp.interruptions[0].at_index == 1


def test_prompts_file_has_required_keys():
    from src.utils.helpers import load_yaml

    prompts = load_yaml("prompts/ingestion_prompts.yaml")
    assert "{fragments}" in prompts["speaker_window"]["prompt"]
    assert "{fragments}" in prompts["stitch_window"]["prompt"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/models/test_ingestion_responses.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.models.ingestion_responses'`

- [ ] **Step 3: Implement**

`src/models/ingestion_responses.py`:

```python
"""Pydantic models validating LLM responses for Layer 1 ingestion passes."""

from typing import List

from pydantic import BaseModel, Field


class SpeakerAssignment(BaseModel):
    """One fragment's proposed speaker assignment."""

    index: int = Field(..., ge=0, description="Fragment index within the window")
    speaker: str = Field(..., description="Provisional handle, e.g. 'S1'")
    confidence: float = Field(..., ge=0.0, le=1.0)


class SpeakerWindowResponse(BaseModel):
    """Response for one speaker-inference window."""

    assignments: List[SpeakerAssignment]


class UtteranceProposal(BaseModel):
    """A proposed stitched utterance (possibly spanning non-adjacent fragments)."""

    speaker: str = Field(..., description="Speaker handle")
    fragment_indices: List[int] = Field(..., min_length=1, description="Ordered fragment indices")
    confidence: float = Field(..., ge=0.0, le=1.0)


class InterruptionProposal(BaseModel):
    """A proposed interruption between two utterances in this window."""

    interrupting: int = Field(..., ge=0, description="Index into the utterances list")
    interrupted: int = Field(..., ge=0, description="Index into the utterances list")
    at_index: int = Field(..., ge=0, description="Fragment index where the interruption begins")


class StitchWindowResponse(BaseModel):
    """Response for one stitching window."""

    utterances: List[UtteranceProposal]
    interruptions: List[InterruptionProposal] = Field(default_factory=list)
```

`prompts/ingestion_prompts.yaml`:

```yaml
# Speaker inference over a window of unlabeled transcript fragments.
# The transcript has NO speaker labels; speakers must be inferred from
# conversational cues (questions vs answers, first-person content, verbal tics).
speaker_window:
  prompt: |
    You are analyzing a raw conversation transcript that has NO speaker labels.
    The fragments below are numbered and appear in as-spoken order. Multiple people
    are talking, sometimes interrupting or talking over one another.

    Assign every fragment a provisional speaker handle (S1, S2, S3, ...). Use cues
    such as: who asks vs answers questions, first-person statements about roles or
    work, greetings and responses, and consistent verbal habits. Use the SAME handle
    for the same person throughout this window. Do not invent more speakers than
    the dialogue supports.

    For each fragment provide a confidence between 0.0 and 1.0.

    Format: {{"assignments": [{{"index": <int>, "speaker": "<handle>", "confidence": <float>}}, ...]}}

    Every fragment index must appear exactly once.

    Fragments:
    """
    {fragments}
    """

    Provide your response explicitly formatted as JSON.

# Stitching pass: group fragments into utterances (continuous thoughts) and
# detect interruptions. The fragment sequence is never modified; this only
# proposes grouping and interruption relationships.
stitch_window:
  prompt: |
    You are analyzing a conversation transcript. Each numbered fragment below is
    prefixed with its speaker handle. People sometimes interrupt each other, so a
    speaker's single continuous thought (an "utterance") may be split across
    NON-ADJACENT fragments, with other speakers' fragments in between.

    Group the fragments into utterances:
      - Consecutive fragments by the same speaker that express one thought belong
        to the same utterance.
      - If a speaker is interrupted and then RESUMES the same thought, include the
        resumption fragments in the SAME utterance (this is the important case).
      - A genuine topic change by the same speaker starts a new utterance.

    Also report interruptions: cases where one utterance breaks into another
    before it finishes.

    Format:
    {{"utterances": [{{"speaker": "<handle>", "fragment_indices": [<int>, ...], "confidence": <float>}}, ...],
      "interruptions": [{{"interrupting": <utterance list index>, "interrupted": <utterance list index>, "at_index": <fragment index>}}, ...]}}

    Every fragment index must appear in exactly one utterance. fragment_indices
    must be in ascending order. Confidence is between 0.0 and 1.0.

    Fragments:
    """
    {fragments}
    """

    Provide your response explicitly formatted as JSON.
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/models/test_ingestion_responses.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add prompts/ingestion_prompts.yaml src/models/ingestion_responses.py tests/models/test_ingestion_responses.py
git commit -m "feat: prompts and response models for speaker inference and stitching"
```

---

### Task 8: Speaker inference service (windowed pass + overlap reconciliation)

**Files:**
- Create: `src/ingestion/speaker_inference.py`
- Test: `tests/ingestion/test_speaker_inference.py`

**Interfaces:**
- Consumes: `agent` singleton (`from src.agents.agent_factory import agent`, awaitable `agent.call_model(prompt: str) -> Dict[str, Any]`), `RawFragment`, `SpeakerWindowResponse`, `load_yaml` (`src.utils.helpers`).
- Produces:
  - `FragmentSpeaker(BaseModel)`: `sequence_order: int`, `handle: str`, `confidence: float`
  - `SpeakerInferenceResult(BaseModel)`: `handles: List[str]` (distinct, order of first appearance), `assignments: List[FragmentSpeaker]` (one per input fragment, same order)
  - `SpeakerInferenceService(window_size: int = 40, overlap: int = 10)` with `async def infer(self, fragments: List[RawFragment]) -> SpeakerInferenceResult`
- Reconciliation strategy (deterministic, no extra LLM calls): windows overlap by `overlap` fragments. For each new window, map its handles onto global handles by majority vote over the overlap region (a window handle maps to whichever global handle it agrees with on most overlapping fragments). Unmatched window handles become new global handles. Overlap-region assignments keep the earlier window's values.

- [ ] **Step 1: Write the failing tests** (`tests/ingestion/test_speaker_inference.py`)

```python
from unittest.mock import AsyncMock, patch

import pytest

from src.ingestion.models import RawFragment
from src.ingestion.speaker_inference import SpeakerInferenceService


def frags(n: int):
    return [
        RawFragment(text=f"Fragment {i}.", start_char=i * 20, end_char=i * 20 + 11, sequence_order=i)
        for i in range(n)
    ]


def window_response(indices, speakers, confidence=0.9):
    return {
        "assignments": [
            {"index": i, "speaker": s, "confidence": confidence} for i, s in zip(indices, speakers)
        ]
    }


@pytest.mark.asyncio
async def test_single_window_assigns_all_fragments():
    service = SpeakerInferenceService(window_size=10, overlap=2)
    fragments = frags(4)
    mock_response = window_response([0, 1, 2, 3], ["S1", "S2", "S1", "S2"])
    with patch("src.ingestion.speaker_inference.agent") as mock_agent:
        mock_agent.call_model = AsyncMock(return_value=mock_response)
        result = await service.infer(fragments)
    assert result.handles == ["S1", "S2"]
    assert [a.handle for a in result.assignments] == ["S1", "S2", "S1", "S2"]
    assert all(0.0 <= a.confidence <= 1.0 for a in result.assignments)


@pytest.mark.asyncio
async def test_overlapping_windows_reconcile_handles():
    # Window 1 covers 0-3, window 2 covers 2-5 (overlap=2).
    # Window 2 calls the same people "A" and "B"; overlap voting must map them
    # back to the global S1/S2 handles.
    service = SpeakerInferenceService(window_size=4, overlap=2)
    fragments = frags(6)
    responses = [
        window_response([0, 1, 2, 3], ["S1", "S2", "S1", "S2"]),
        window_response([0, 1, 2, 3], ["A", "B", "A", "B"]),  # local indices of frags 2-5
    ]
    with patch("src.ingestion.speaker_inference.agent") as mock_agent:
        mock_agent.call_model = AsyncMock(side_effect=responses)
        result = await service.infer(fragments)
    assert result.handles == ["S1", "S2"]
    assert [a.handle for a in result.assignments] == ["S1", "S2", "S1", "S2", "S1", "S2"]


@pytest.mark.asyncio
async def test_missing_assignment_gets_zero_confidence_unknown():
    # LLM omitted fragment 1; service must still return one assignment per fragment.
    service = SpeakerInferenceService(window_size=10, overlap=2)
    fragments = frags(3)
    mock_response = window_response([0, 2], ["S1", "S1"])
    with patch("src.ingestion.speaker_inference.agent") as mock_agent:
        mock_agent.call_model = AsyncMock(return_value=mock_response)
        result = await service.infer(fragments)
    assert len(result.assignments) == 3
    assert result.assignments[1].confidence == 0.0
```

Note: if `pytest.ini` does not already set `asyncio_mode = auto`, keep the `@pytest.mark.asyncio` decorators (pytest-asyncio is already a project dependency — existing async tests use it).

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/ingestion/test_speaker_inference.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.ingestion.speaker_inference'`

- [ ] **Step 3: Implement** (`src/ingestion/speaker_inference.py`)

```python
"""Speaker genesis: infer provisional speakers for unlabeled transcripts.

Windowed LLM pass proposes handles per fragment; overlapping windows are
reconciled deterministically by majority vote over the overlap region.
"""

from typing import Dict, List

from pydantic import BaseModel, Field, ValidationError

from src.agents.agent_factory import agent
from src.ingestion.models import RawFragment
from src.models.ingestion_responses import SpeakerWindowResponse
from src.utils.helpers import load_yaml
from src.utils.logger import get_logger

logger = get_logger()

PROMPTS_PATH = "prompts/ingestion_prompts.yaml"
UNKNOWN_HANDLE = "S?"


class FragmentSpeaker(BaseModel):
    """Resolved speaker assignment for one fragment."""

    sequence_order: int = Field(..., ge=0)
    handle: str
    confidence: float = Field(..., ge=0.0, le=1.0)


class SpeakerInferenceResult(BaseModel):
    """Inference output: global handles and one assignment per fragment."""

    handles: List[str]
    assignments: List[FragmentSpeaker]


class SpeakerInferenceService:
    """Infers provisional speaker handles for unlabeled fragments."""

    def __init__(self, window_size: int = 40, overlap: int = 10):
        if overlap >= window_size:
            raise ValueError("overlap must be smaller than window_size")
        self.window_size = window_size
        self.overlap = overlap
        self.prompts = load_yaml(PROMPTS_PATH)

    async def infer(self, fragments: List[RawFragment]) -> SpeakerInferenceResult:
        """Assign a provisional speaker handle to every fragment."""
        # global assignment per sequence_order -> (handle, confidence)
        global_assignments: Dict[int, FragmentSpeaker] = {}
        handles: List[str] = []

        step = self.window_size - self.overlap
        for window_start in range(0, len(fragments), step):
            window = fragments[window_start:window_start + self.window_size]
            if not window:
                break
            window_result = await self._infer_window(window)

            mapping = self._reconcile(window_start, window, window_result, global_assignments)

            for local_index, (handle, confidence) in window_result.items():
                seq = window[local_index].sequence_order
                if seq in global_assignments:
                    continue  # overlap region: keep earlier window's assignment
                global_handle = mapping.get(handle, handle)
                if global_handle not in handles:
                    handles.append(global_handle)
                global_assignments[seq] = FragmentSpeaker(
                    sequence_order=seq, handle=global_handle, confidence=confidence
                )

            if window_start + self.window_size >= len(fragments):
                break

        # Guarantee one assignment per fragment; LLM omissions get UNKNOWN at 0.0.
        assignments = []
        for frag in fragments:
            if frag.sequence_order not in global_assignments:
                logger.warning(f"No speaker assignment for fragment {frag.sequence_order}; marking unknown")
                if UNKNOWN_HANDLE not in handles:
                    handles.append(UNKNOWN_HANDLE)
                global_assignments[frag.sequence_order] = FragmentSpeaker(
                    sequence_order=frag.sequence_order, handle=UNKNOWN_HANDLE, confidence=0.0
                )
            assignments.append(global_assignments[frag.sequence_order])

        return SpeakerInferenceResult(handles=handles, assignments=assignments)

    async def _infer_window(self, window: List[RawFragment]) -> Dict[int, tuple]:
        """Run one window through the LLM; returns {local_index: (handle, confidence)}."""
        numbered = "\n".join(f"{i}: {frag.text}" for i, frag in enumerate(window))
        prompt = self.prompts["speaker_window"]["prompt"].format(fragments=numbered)
        raw = await agent.call_model(prompt)
        try:
            response = SpeakerWindowResponse.model_validate(raw)
        except ValidationError as e:
            logger.warning(f"Invalid speaker window response, skipping window: {e}")
            return {}
        return {
            a.index: (a.speaker, a.confidence)
            for a in response.assignments
            if 0 <= a.index < len(window)
        }

    def _reconcile(
        self,
        window_start: int,
        window: List[RawFragment],
        window_result: Dict[int, tuple],
        global_assignments: Dict[int, FragmentSpeaker],
    ) -> Dict[str, str]:
        """Map window-local handles to global handles by overlap majority vote."""
        votes: Dict[str, Dict[str, int]] = {}
        for local_index, (handle, _confidence) in window_result.items():
            seq = window[local_index].sequence_order
            existing = global_assignments.get(seq)
            if existing is None:
                continue
            votes.setdefault(handle, {}).setdefault(existing.handle, 0)
            votes[handle][existing.handle] += 1

        return {
            local_handle: max(counts, key=counts.get)
            for local_handle, counts in votes.items()
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/ingestion/test_speaker_inference.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/speaker_inference.py tests/ingestion/test_speaker_inference.py
git commit -m "feat: windowed speaker inference with deterministic overlap reconciliation"
```

---

### Task 9: Stitcher (baseline grouping + LLM continuation/interruption pass)

**Files:**
- Create: `src/ingestion/stitcher.py`
- Test: `tests/ingestion/test_stitcher.py`

**Interfaces:**
- Consumes: `agent` singleton, `RawFragment`, `FragmentSpeaker` (Task 8), `StitchWindowResponse` (Task 7), `load_yaml`.
- Produces:
  - `StitchedUtterance(BaseModel)`: `ordinal: int`, `handle: str`, `sequence_orders: List[int]` (ascending), `confidence: float`
  - `Interruption(BaseModel)`: `interrupting_ordinal: int`, `interrupted_ordinal: int`, `at_sequence_order: int`
  - `StitchResult(BaseModel)`: `utterances: List[StitchedUtterance]`, `interruptions: List[Interruption]`
  - `Stitcher(window_size: int = 30)` with `async def stitch(self, fragments: List[RawFragment], assignments: List[FragmentSpeaker]) -> StitchResult`
- Behavior: **baseline first** — consecutive fragments with the same handle form one utterance (confidence 1.0, purely structural). Then the LLM pass runs per window over the baseline utterances (fragments rendered with speaker prefixes) and may (a) merge non-adjacent same-speaker utterances into one (continuation across interruption; confidence from LLM) and (b) report interruptions. Invalid LLM proposals (mixing speakers, out-of-range indices, descending order) are logged and dropped — the baseline grouping is the fallback, so stitching never fails the pipeline.

- [ ] **Step 1: Write the failing tests** (`tests/ingestion/test_stitcher.py`)

```python
from unittest.mock import AsyncMock, patch

import pytest

from src.ingestion.models import RawFragment
from src.ingestion.speaker_inference import FragmentSpeaker
from src.ingestion.stitcher import Stitcher


def make_inputs(handles):
    fragments = [
        RawFragment(text=f"Fragment {i}.", start_char=i * 20, end_char=i * 20 + 11, sequence_order=i)
        for i in range(len(handles))
    ]
    assignments = [
        FragmentSpeaker(sequence_order=i, handle=h, confidence=0.9) for i, h in enumerate(handles)
    ]
    return fragments, assignments


@pytest.mark.asyncio
async def test_baseline_groups_consecutive_same_speaker():
    fragments, assignments = make_inputs(["S1", "S1", "S2", "S1"])
    stitcher = Stitcher()
    empty = {"utterances": [], "interruptions": []}
    with patch("src.ingestion.stitcher.agent") as mock_agent:
        mock_agent.call_model = AsyncMock(return_value=empty)
        result = await stitcher.stitch(fragments, assignments)
    groups = [u.sequence_orders for u in result.utterances]
    assert groups == [[0, 1], [2], [3]]


@pytest.mark.asyncio
async def test_llm_merge_stitches_interrupted_utterance():
    # S1 speaks (0), S2 interrupts (1), S1 resumes the same thought (2).
    fragments, assignments = make_inputs(["S1", "S2", "S1"])
    llm_response = {
        "utterances": [
            {"speaker": "S1", "fragment_indices": [0, 2], "confidence": 0.8},
            {"speaker": "S2", "fragment_indices": [1], "confidence": 0.9},
        ],
        "interruptions": [{"interrupting": 1, "interrupted": 0, "at_index": 1}],
    }
    stitcher = Stitcher()
    with patch("src.ingestion.stitcher.agent") as mock_agent:
        mock_agent.call_model = AsyncMock(return_value=llm_response)
        result = await stitcher.stitch(fragments, assignments)
    s1_utterances = [u for u in result.utterances if u.handle == "S1"]
    assert len(s1_utterances) == 1
    assert s1_utterances[0].sequence_orders == [0, 2]
    assert len(result.interruptions) == 1
    assert result.interruptions[0].at_sequence_order == 1


@pytest.mark.asyncio
async def test_invalid_llm_proposal_falls_back_to_baseline():
    # Proposal mixes two speakers into one utterance -> must be dropped.
    fragments, assignments = make_inputs(["S1", "S2"])
    bad_response = {
        "utterances": [{"speaker": "S1", "fragment_indices": [0, 1], "confidence": 0.8}],
        "interruptions": [],
    }
    stitcher = Stitcher()
    with patch("src.ingestion.stitcher.agent") as mock_agent:
        mock_agent.call_model = AsyncMock(return_value=bad_response)
        result = await stitcher.stitch(fragments, assignments)
    groups = [u.sequence_orders for u in result.utterances]
    assert groups == [[0], [1]]  # baseline preserved
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/ingestion/test_stitcher.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.ingestion.stitcher'`

- [ ] **Step 3: Implement** (`src/ingestion/stitcher.py`)

```python
"""Stitching pass: overlay utterance structure on the immutable fragment sequence.

Baseline: consecutive same-speaker fragments form utterances. An LLM pass may
merge non-adjacent same-speaker utterances (continuation across interruption)
and report interruptions. The fragment sequence itself is never modified.
"""

from typing import Dict, List

from pydantic import BaseModel, Field, ValidationError

from src.agents.agent_factory import agent
from src.ingestion.models import RawFragment
from src.ingestion.speaker_inference import FragmentSpeaker
from src.models.ingestion_responses import StitchWindowResponse
from src.utils.helpers import load_yaml
from src.utils.logger import get_logger

logger = get_logger()

PROMPTS_PATH = "prompts/ingestion_prompts.yaml"


class StitchedUtterance(BaseModel):
    """A speaker's continuous thought spanning one or more fragments."""

    ordinal: int = Field(..., ge=0, description="Stable ordinal within the interview")
    handle: str
    sequence_orders: List[int] = Field(..., min_length=1)
    confidence: float = Field(..., ge=0.0, le=1.0)


class Interruption(BaseModel):
    """One utterance breaking into another."""

    interrupting_ordinal: int
    interrupted_ordinal: int
    at_sequence_order: int


class StitchResult(BaseModel):
    """Utterance overlay for an interview."""

    utterances: List[StitchedUtterance]
    interruptions: List[Interruption]


class Stitcher:
    """Builds the utterance overlay: baseline grouping refined by an LLM pass."""

    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.prompts = load_yaml(PROMPTS_PATH)

    async def stitch(
        self, fragments: List[RawFragment], assignments: List[FragmentSpeaker]
    ) -> StitchResult:
        """Produce the utterance overlay for the given attributed fragments."""
        handle_by_seq = {a.sequence_order: a.handle for a in assignments}
        groups = self._baseline_groups(fragments, handle_by_seq)

        merged_groups, interruptions = await self._refine_with_llm(
            fragments, handle_by_seq, groups
        )

        utterances = [
            StitchedUtterance(
                ordinal=i, handle=handle, sequence_orders=seqs, confidence=confidence
            )
            for i, (handle, seqs, confidence) in enumerate(merged_groups)
        ]
        return StitchResult(utterances=utterances, interruptions=interruptions)

    def _baseline_groups(self, fragments, handle_by_seq) -> List[tuple]:
        """Group consecutive same-speaker fragments: [(handle, [seq, ...], confidence)]."""
        groups: List[tuple] = []
        for frag in fragments:
            handle = handle_by_seq.get(frag.sequence_order, "S?")
            if groups and groups[-1][0] == handle:
                groups[-1][1].append(frag.sequence_order)
            else:
                groups.append((handle, [frag.sequence_order], 1.0))
        return groups

    async def _refine_with_llm(self, fragments, handle_by_seq, groups):
        """Ask the LLM for cross-interruption merges and interruption edges."""
        text_by_seq = {f.sequence_order: f.text for f in fragments}
        numbered = "\n".join(
            f"{f.sequence_order}: [{handle_by_seq.get(f.sequence_order, 'S?')}] {f.text}"
            for f in fragments[: self.window_size * 10]  # safety cap
        )
        prompt = self.prompts["stitch_window"]["prompt"].format(fragments=numbered)

        try:
            raw = await agent.call_model(prompt)
            response = StitchWindowResponse.model_validate(raw)
        except (ValidationError, Exception) as e:  # LLM failure -> baseline only
            logger.warning(f"Stitch pass failed or invalid; using baseline grouping: {e}")
            return groups, []

        valid_seqs = set(text_by_seq)
        merged: List[tuple] = []
        for proposal in response.utterances:
            seqs = proposal.fragment_indices
            if (
                seqs != sorted(seqs)
                or not set(seqs).issubset(valid_seqs)
                or len({handle_by_seq.get(s) for s in seqs}) != 1
                or handle_by_seq.get(seqs[0]) != proposal.speaker
            ):
                logger.warning(f"Dropping invalid utterance proposal: {proposal}")
                continue
            merged.append((proposal.speaker, seqs, proposal.confidence))

        # Fragments not covered by valid proposals keep their baseline groups.
        covered = {s for _, seqs, _ in merged for s in seqs}
        for handle, seqs, confidence in groups:
            remaining = [s for s in seqs if s not in covered]
            if remaining:
                merged.append((handle, remaining, confidence))
        merged.sort(key=lambda g: g[1][0])

        interruptions = []
        for prop in response.interruptions:
            if prop.interrupting < len(merged) and prop.interrupted < len(merged):
                interruptions.append(
                    Interruption(
                        interrupting_ordinal=prop.interrupting,
                        interrupted_ordinal=prop.interrupted,
                        at_sequence_order=prop.at_index,
                    )
                )
        return merged, interruptions
```

Note for the implementer: interruption ordinals index into the *final sorted* `merged` list; because proposals are validated per-utterance and the LLM sees the same ordering it proposed, mismatches are possible — if `at_sequence_order` is not within the interrupted utterance's span, drop the interruption (add that check while implementing; test 2 covers the happy path).

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/ingestion/test_stitcher.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/stitcher.py tests/ingestion/test_stitcher.py
git commit -m "feat: utterance stitcher with baseline grouping and LLM refinement"
```

---

### Task 10: Ingestion orchestrator (events + upgraded map)

**Files:**
- Create: `src/ingestion/orchestrator.py`
- Create: `src/ingestion/__main__.py`
- Test: `tests/ingestion/test_orchestrator.py`

**Interfaces:**
- Consumes: `normalize` (Task 3), `SpeakerInferenceService` (Task 8), `Stitcher` (Task 9), repositories (`src/events/repository.py`: `get_interview_repository()`, `get_sentence_repository()`, each with `async load(id)`, `async save(aggregate)`), aggregates (Tasks 4–6), `LocalJsonlMapStorage` (`src/io/local_storage.py`).
- Produces:
  - `IngestionResult(BaseModel)`: `interview_id: str`, `fragment_count: int`, `speaker_count: int`, `utterance_count: int`, `interruption_count: int`, `low_confidence_count: int` (attributions below `confidence_review_threshold`, default 0.7)
  - `IngestionOrchestrator(project_id: str, map_dir: Path, confidence_review_threshold: float = 0.7)` with `async def ingest_file(self, file_path: Path, correlation_id: Optional[str] = None) -> IngestionResult`
- Event flow (exact order):
  1. Read file, `normalize(text)`.
  2. New `Interview` aggregate (fresh UUID4): `create(title=<filename>, source=<path>, metadata={"content_hash", "format", "fragment_count"})`.
  3. Speakers: LABELED → one per parsed label (`provisional=False`, `confidence=1.0`, `method="parsed"`); FLAT → run `SpeakerInferenceService`, one per handle (`provisional=True`, confidence = mean of that handle's assignment confidences, `method="inference"`). `add_speaker(...)` each; speaker UUID = `uuid5(NAMESPACE_DNS, f"{interview_id}:speaker:{handle}")`.
  4. `interview_repo.save(interview)` — commits InterviewCreated + SpeakerCreated events.
  5. Per fragment: new `Sentence` aggregate (UUID = `uuid5(NAMESPACE_DNS, f"{interview_id}:{index}")`), `create(interview_id, index=sequence_order, text, start_char, end_char)`, then `attribute_speaker(speaker_id, confidence, method)`; `sentence_repo.save(sentence)`.
  6. Stitch: FLAT uses inference assignments; LABELED builds `FragmentSpeaker` list from parsed labels (confidence 1.0). `identify_utterance` per stitched utterance (utterance UUID = `uuid5(NAMESPACE_DNS, f"{interview_id}:utterance:{ordinal}")`, translating fragment `sequence_orders` → fragment UUIDs) and `record_interruption` per interruption (translating ordinals → utterance UUIDs); then `interview_repo.save(interview)` again.
  7. Write map file `<map_dir>/<stem>_map.jsonl`, one line per fragment: `{"sentence_id", "sequence_order", "sentence", "start_char", "end_char", "speaker_id", "speaker_confidence", "utterance_id"}` — the upgraded map schema.
  - All system events carry `actor=Actor(actor_type=ActorType.SYSTEM, user_id="ingestion")` and the shared `correlation_id`.

- [ ] **Step 1: Write the failing test** (`tests/ingestion/test_orchestrator.py`)

```python
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.ingestion.orchestrator import IngestionOrchestrator

LABELED = """Alice: Hi, thanks for joining today.
Bob: Happy to be here.
Alice: Let's get started.
"""


@pytest.mark.asyncio
async def test_labeled_ingestion_emits_events_and_writes_map(tmp_path: Path):
    input_file = tmp_path / "meeting.txt"
    input_file.write_text(LABELED)
    map_dir = tmp_path / "maps"

    saved_aggregates = []

    async def capture_save(aggregate, expected_version=None):
        saved_aggregates.append(aggregate)
        aggregate.mark_events_as_committed()

    mock_repo = MagicMock()
    mock_repo.save = AsyncMock(side_effect=capture_save)

    with patch("src.ingestion.orchestrator.get_interview_repository", return_value=mock_repo), \
         patch("src.ingestion.orchestrator.get_sentence_repository", return_value=mock_repo):
        orchestrator = IngestionOrchestrator(project_id="proj-1", map_dir=map_dir)
        result = await orchestrator.ingest_file(input_file)

    assert result.fragment_count == 3
    assert result.speaker_count == 2
    assert result.utterance_count == 3  # baseline: three speaker turns
    assert result.low_confidence_count == 0  # parsed labels are confidence 1.0

    map_file = map_dir / "meeting_map.jsonl"
    assert map_file.exists()
    entries = [json.loads(line) for line in map_file.read_text().splitlines()]
    assert len(entries) == 3
    for entry in entries:
        assert LABELED[entry["start_char"]:entry["end_char"]] == entry["sentence"]
        assert entry["speaker_id"] is not None
        assert entry["utterance_id"] is not None
```

Note: labeled input must not call the LLM at all — no `agent` mocking is present, so any accidental agent call in the labeled path will fail loudly (no API key in unit tests). The Stitcher's LLM refinement is skipped when every attribution has confidence 1.0 and method `parsed`? No — keep it simpler and explicit: the orchestrator calls `Stitcher.stitch` only for FLAT transcripts; for LABELED it uses the baseline grouping directly via `Stitcher._baseline_groups` equivalent — expose this as a public method `Stitcher.baseline(fragments, assignments) -> StitchResult` (no LLM) and have the orchestrator use `baseline()` for LABELED and `stitch()` for FLAT. Add `baseline()` while implementing this task (it wraps `_baseline_groups` and returns a `StitchResult` with no interruptions).

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/ingestion/test_orchestrator.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.ingestion.orchestrator'`

- [ ] **Step 3: Implement**

`src/ingestion/orchestrator.py`:

```python
"""Layer 1 ingestion orchestrator.

Flow: read -> normalize -> speakers (parse or infer) -> fragment events ->
stitch overlay -> upgraded map file. All writes go through the event-sourced
repositories; Neo4j is populated by the projection service downstream.
"""

import uuid
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel

from src.events.envelope import Actor, ActorType, generate_correlation_id
from src.events.aggregates import Interview, Sentence
from src.events.repository import get_interview_repository, get_sentence_repository
from src.ingestion.models import NormalizedTranscript, TranscriptFormat
from src.ingestion.normalizer import normalize
from src.ingestion.speaker_inference import (
    FragmentSpeaker,
    SpeakerInferenceResult,
    SpeakerInferenceService,
)
from src.ingestion.stitcher import Stitcher, StitchResult
from src.utils.helpers import append_json_line
from src.utils.logger import get_logger

logger = get_logger()


class IngestionResult(BaseModel):
    """Summary of one ingested transcript."""

    interview_id: str
    fragment_count: int
    speaker_count: int
    utterance_count: int
    interruption_count: int
    low_confidence_count: int


class IngestionOrchestrator:
    """Runs the full Layer 1 ingestion flow for one file."""

    def __init__(
        self,
        project_id: str,
        map_dir: Path,
        confidence_review_threshold: float = 0.7,
    ):
        self.project_id = project_id
        self.map_dir = Path(map_dir)
        self.threshold = confidence_review_threshold
        self.inference = SpeakerInferenceService()
        self.stitcher = Stitcher()

    async def ingest_file(
        self, file_path: Path, correlation_id: Optional[str] = None
    ) -> IngestionResult:
        """Ingest one transcript file end to end."""
        correlation_id = correlation_id or generate_correlation_id()
        actor = Actor(actor_type=ActorType.SYSTEM, user_id="ingestion")
        text = Path(file_path).read_text(encoding="utf-8")
        transcript = normalize(text)
        interview_id = str(uuid.uuid4())

        interview = Interview(interview_id)
        interview.create(
            title=Path(file_path).name,
            source=str(file_path),
            metadata={
                "content_hash": transcript.content_hash,
                "format": transcript.format.value,
                "fragment_count": len(transcript.fragments),
            },
            actor=actor,
            correlation_id=correlation_id,
            project_id=self.project_id,
        )

        assignments = await self._resolve_speakers(
            interview, transcript, actor, correlation_id
        )
        interview_repo = get_interview_repository()
        await interview_repo.save(interview)

        speaker_ids = {
            info["handle"]: sid for sid, info in interview.speakers.items()
        }
        fragment_uuids = await self._emit_fragments(
            interview_id, transcript, assignments, speaker_ids, actor, correlation_id
        )

        stitch = await self._stitch(transcript, assignments)
        utterance_ids = self._emit_stitch(
            interview, stitch, speaker_ids, fragment_uuids, actor, correlation_id
        )
        await interview_repo.save(interview)

        self._write_map(
            Path(file_path), transcript, assignments, speaker_ids, fragment_uuids,
            stitch, utterance_ids,
        )

        low_confidence = sum(1 for a in assignments if a.confidence < self.threshold)
        return IngestionResult(
            interview_id=interview_id,
            fragment_count=len(transcript.fragments),
            speaker_count=len(speaker_ids),
            utterance_count=len(stitch.utterances),
            interruption_count=len(stitch.interruptions),
            low_confidence_count=low_confidence,
        )

    async def _resolve_speakers(
        self, interview: Interview, transcript: NormalizedTranscript, actor, correlation_id
    ) -> List[FragmentSpeaker]:
        """Create speakers (parsed or inferred) and return per-fragment assignments."""
        if transcript.format == TranscriptFormat.LABELED:
            handles = transcript.speaker_labels
            assignments = [
                FragmentSpeaker(
                    sequence_order=f.sequence_order, handle=f.speaker_label or "S?", confidence=1.0
                )
                for f in transcript.fragments
            ]
            provisional, method, confidences = False, "parsed", {h: 1.0 for h in handles}
        else:
            result: SpeakerInferenceResult = await self.inference.infer(transcript.fragments)
            handles = result.handles
            assignments = result.assignments
            confidences = {
                h: (
                    sum(a.confidence for a in assignments if a.handle == h)
                    / max(1, sum(1 for a in assignments if a.handle == h))
                )
                for h in handles
            }
            provisional, method = True, "inference"

        for handle in handles:
            speaker_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{interview.aggregate_id}:speaker:{handle}"))
            interview.add_speaker(
                speaker_id,
                handle=handle,
                display_name=handle,
                provisional=provisional,
                confidence=round(confidences[handle], 4),
                method=method,
                actor=actor,
                correlation_id=correlation_id,
            )
        return assignments

    async def _emit_fragments(
        self, interview_id, transcript, assignments, speaker_ids, actor, correlation_id
    ) -> dict:
        """Create Sentence aggregates with offsets and speaker attribution."""
        sentence_repo = get_sentence_repository()
        assignment_by_seq = {a.sequence_order: a for a in assignments}
        fragment_uuids = {}
        for frag in transcript.fragments:
            sentence_id = str(
                uuid.uuid5(uuid.NAMESPACE_DNS, f"{interview_id}:{frag.sequence_order}")
            )
            fragment_uuids[frag.sequence_order] = sentence_id
            sentence = Sentence(sentence_id)
            sentence.create(
                interview_id=interview_id,
                index=frag.sequence_order,
                text=frag.text,
                start_char=frag.start_char,
                end_char=frag.end_char,
                actor=actor,
                correlation_id=correlation_id,
            )
            assignment = assignment_by_seq[frag.sequence_order]
            speaker_id = speaker_ids.get(assignment.handle)
            if speaker_id:
                sentence.attribute_speaker(
                    speaker_id,
                    confidence=assignment.confidence,
                    method="parsed" if assignment.confidence == 1.0 else "inference",
                    actor=actor,
                    correlation_id=correlation_id,
                )
            await sentence_repo.save(sentence)
        return fragment_uuids

    async def _stitch(self, transcript, assignments) -> StitchResult:
        if transcript.format == TranscriptFormat.LABELED:
            return self.stitcher.baseline(transcript.fragments, assignments)
        return await self.stitcher.stitch(transcript.fragments, assignments)

    def _emit_stitch(
        self, interview, stitch, speaker_ids, fragment_uuids, actor, correlation_id
    ) -> dict:
        """Emit UtteranceIdentified and InterruptionRecorded events."""
        utterance_ids = {}
        for utt in stitch.utterances:
            utterance_id = str(
                uuid.uuid5(
                    uuid.NAMESPACE_DNS, f"{interview.aggregate_id}:utterance:{utt.ordinal}"
                )
            )
            utterance_ids[utt.ordinal] = utterance_id
            interview.identify_utterance(
                utterance_id,
                speaker_id=speaker_ids[utt.handle],
                fragment_ids=[fragment_uuids[s] for s in utt.sequence_orders],
                confidence=utt.confidence,
                actor=actor,
                correlation_id=correlation_id,
            )
        for intr in stitch.interruptions:
            interview.record_interruption(
                interrupting_utterance_id=utterance_ids[intr.interrupting_ordinal],
                interrupted_utterance_id=utterance_ids[intr.interrupted_ordinal],
                at_fragment_id=fragment_uuids[intr.at_sequence_order],
                actor=actor,
                correlation_id=correlation_id,
            )
        return utterance_ids

    def _write_map(
        self, file_path, transcript, assignments, speaker_ids, fragment_uuids, stitch, utterance_ids
    ) -> None:
        """Write the upgraded map file: the pipeline's coordinate system."""
        self.map_dir.mkdir(parents=True, exist_ok=True)
        map_path = self.map_dir / f"{file_path.stem}_map.jsonl"
        if map_path.exists():
            map_path.unlink()
        assignment_by_seq = {a.sequence_order: a for a in assignments}
        utterance_by_seq = {
            seq: utterance_ids[u.ordinal]
            for u in stitch.utterances
            for seq in u.sequence_orders
        }
        for frag in transcript.fragments:
            a = assignment_by_seq[frag.sequence_order]
            append_json_line(
                {
                    "sentence_id": fragment_uuids[frag.sequence_order],
                    "sequence_order": frag.sequence_order,
                    "sentence": frag.text,
                    "start_char": frag.start_char,
                    "end_char": frag.end_char,
                    "speaker_id": speaker_ids.get(a.handle),
                    "speaker_confidence": a.confidence,
                    "utterance_id": utterance_by_seq.get(frag.sequence_order),
                },
                map_path,
            )
```

Check `append_json_line`'s exact signature in `src/utils/helpers.py` before use (argument order may be `(data, file_path)` or `(file_path, data)`) and adjust the call accordingly.

Also add to `src/ingestion/stitcher.py` (public baseline, no LLM):

```python
    def baseline(
        self, fragments: List[RawFragment], assignments: List[FragmentSpeaker]
    ) -> StitchResult:
        """Baseline grouping only (no LLM): consecutive same-speaker fragments."""
        handle_by_seq = {a.sequence_order: a.handle for a in assignments}
        groups = self._baseline_groups(fragments, handle_by_seq)
        utterances = [
            StitchedUtterance(ordinal=i, handle=h, sequence_orders=seqs, confidence=c)
            for i, (h, seqs, c) in enumerate(groups)
        ]
        return StitchResult(utterances=utterances, interruptions=[])
```

`src/ingestion/__main__.py` (manual runner):

```python
"""Run Layer 1 ingestion from the command line.

Usage: python -m src.ingestion <file> [--project-id ID] [--map-dir DIR]
"""

import argparse
import asyncio
from pathlib import Path

from src.ingestion.orchestrator import IngestionOrchestrator


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest a transcript (Layer 1)")
    parser.add_argument("file", type=Path)
    parser.add_argument("--project-id", default="default-project")
    parser.add_argument("--map-dir", type=Path, default=Path("data/maps"))
    args = parser.parse_args()

    orchestrator = IngestionOrchestrator(project_id=args.project_id, map_dir=args.map_dir)
    result = asyncio.run(orchestrator.ingest_file(args.file))
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/ingestion -m "not integration" -q`
Expected: PASS (all ingestion tests)

- [ ] **Step 5: Commit**

```bash
git add src/ingestion tests/ingestion
git commit -m "feat: Layer 1 ingestion orchestrator with upgraded map output"
```

---

### Task 11: Projection handlers — Speaker nodes

**Files:**
- Create: `src/projections/handlers/speaker_handlers.py`
- Modify: `src/projections/bootstrap.py` (register handlers)
- Modify: `src/projections/handlers/__init__.py` (exports, matching existing style)
- Test: `tests/projections/test_speaker_handlers.py`

**Interfaces:**
- Consumes: `BaseProjectionHandler` (`apply(self, tx, event)` contract; version checking against the aggregate's node via `aggregate_id` happens in the base class), event payloads from Tasks 4–5.
- Produces Neo4j structure:
  - `(:Speaker {speaker_id, handle, display_name, provisional, confidence, method, interview_id, merged_into})`
  - `(:Interview)-[:HAS_PARTICIPANT]->(:Speaker)`
  - `(:Sentence)-[:SPOKEN_BY {confidence, method, locked}]->(:Speaker)`
- Handlers: `SpeakerCreatedHandler`, `SpeakerRenamedHandler`, `SpeakerMergedHandler` (Interview stream), `SpeakerAttributedHandler`, `SpeakerReattributedHandler` (Sentence stream).

- [ ] **Step 1: Write the failing tests** (`tests/projections/test_speaker_handlers.py`) — follow the existing mock pattern in `tests/projections/` (mock `tx.run` as `AsyncMock`; assert on the Cypher and parameters passed):

```python
from unittest.mock import AsyncMock

import pytest

from src.events.envelope import AggregateType, EventEnvelope
from src.projections.handlers.speaker_handlers import (
    SpeakerAttributedHandler,
    SpeakerCreatedHandler,
    SpeakerMergedHandler,
)

IID = "22222222-2222-2222-2222-222222222222"
SP1 = "33333333-3333-3333-3333-333333333333"
SP2 = "44444444-4444-4444-4444-444444444444"
SENT = "77777777-7777-7777-7777-777777777771"


def make_event(event_type, aggregate_type, aggregate_id, data, version=1):
    return EventEnvelope(
        event_type=event_type,
        aggregate_type=aggregate_type,
        aggregate_id=aggregate_id,
        version=version,
        data=data,
    )


@pytest.mark.asyncio
async def test_speaker_created_merges_speaker_and_participant_link():
    handler = SpeakerCreatedHandler()
    tx = AsyncMock()
    event = make_event(
        "SpeakerCreated", AggregateType.INTERVIEW, IID,
        {"speaker_id": SP1, "handle": "S1", "display_name": "S1",
         "provisional": True, "confidence": 0.8, "method": "inference"},
    )
    await handler.apply(tx, event)
    query = tx.run.call_args[0][0]
    params = tx.run.call_args[1]
    assert "MERGE (sp:Speaker {speaker_id: $speaker_id})" in query
    assert "HAS_PARTICIPANT" in query
    assert params["speaker_id"] == SP1
    assert params["interview_id"] == IID


@pytest.mark.asyncio
async def test_speaker_attributed_creates_spoken_by():
    handler = SpeakerAttributedHandler()
    tx = AsyncMock()
    event = make_event(
        "SpeakerAttributed", AggregateType.SENTENCE, SENT,
        {"speaker_id": SP1, "confidence": 0.72, "method": "inference"},
    )
    await handler.apply(tx, event)
    query = tx.run.call_args[0][0]
    params = tx.run.call_args[1]
    assert "SPOKEN_BY" in query
    assert params["confidence"] == 0.72


@pytest.mark.asyncio
async def test_speaker_merged_moves_spoken_by_edges():
    handler = SpeakerMergedHandler()
    tx = AsyncMock()
    event = make_event(
        "SpeakerMerged", AggregateType.INTERVIEW, IID,
        {"surviving_speaker_id": SP1, "merged_speaker_id": SP2},
    )
    await handler.apply(tx, event)
    query = tx.run.call_args[0][0]
    assert "merged_into" in query
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/projections/test_speaker_handlers.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement** (`src/projections/handlers/speaker_handlers.py`)

```python
"""Projection handlers for Speaker-related events."""

import logging

from src.events.envelope import EventEnvelope

from .base_handler import BaseProjectionHandler

logger = logging.getLogger(__name__)


class SpeakerCreatedHandler(BaseProjectionHandler):
    """Creates Speaker node and links it to its Interview."""

    async def apply(self, tx, event: EventEnvelope):
        data = event.data
        query = """
        MATCH (i:Interview {interview_id: $interview_id})
        MERGE (sp:Speaker {speaker_id: $speaker_id})
        SET sp.handle = $handle,
            sp.display_name = $display_name,
            sp.provisional = $provisional,
            sp.confidence = $confidence,
            sp.method = $method,
            sp.interview_id = $interview_id,
            sp.merged_into = null
        MERGE (i)-[:HAS_PARTICIPANT]->(sp)
        """
        await tx.run(
            query,
            interview_id=event.aggregate_id,
            speaker_id=data["speaker_id"],
            handle=data["handle"],
            display_name=data["display_name"],
            provisional=data["provisional"],
            confidence=data.get("confidence"),
            method=data.get("method"),
        )
        logger.info(f"Applied SpeakerCreated for speaker {data['speaker_id']}")


class SpeakerRenamedHandler(BaseProjectionHandler):
    """Updates Speaker display name; clears provisional flag."""

    async def apply(self, tx, event: EventEnvelope):
        data = event.data
        query = """
        MATCH (sp:Speaker {speaker_id: $speaker_id})
        SET sp.display_name = $new_display_name,
            sp.provisional = false
        """
        await tx.run(
            query,
            speaker_id=data["speaker_id"],
            new_display_name=data["new_display_name"],
        )


class SpeakerMergedHandler(BaseProjectionHandler):
    """Moves SPOKEN_BY and SPOKE edges from merged speaker to survivor."""

    async def apply(self, tx, event: EventEnvelope):
        data = event.data
        query = """
        MATCH (merged:Speaker {speaker_id: $merged_speaker_id})
        MATCH (surviving:Speaker {speaker_id: $surviving_speaker_id})
        SET merged.merged_into = $surviving_speaker_id
        WITH merged, surviving
        OPTIONAL MATCH (s:Sentence)-[r:SPOKEN_BY]->(merged)
        FOREACH (_ IN CASE WHEN s IS NULL THEN [] ELSE [1] END |
            MERGE (s)-[nr:SPOKEN_BY]->(surviving)
            SET nr.confidence = r.confidence, nr.method = r.method, nr.locked = r.locked
            DELETE r
        )
        WITH merged, surviving
        OPTIONAL MATCH (merged)-[sp:SPOKE]->(u:Utterance)
        FOREACH (_ IN CASE WHEN u IS NULL THEN [] ELSE [1] END |
            MERGE (surviving)-[:SPOKE]->(u)
            DELETE sp
        )
        """
        await tx.run(
            query,
            merged_speaker_id=data["merged_speaker_id"],
            surviving_speaker_id=data["surviving_speaker_id"],
        )


class SpeakerAttributedHandler(BaseProjectionHandler):
    """Attributes a Sentence (fragment) to a Speaker via SPOKEN_BY."""

    async def apply(self, tx, event: EventEnvelope):
        data = event.data
        query = """
        MATCH (s:Sentence {aggregate_id: $aggregate_id})
        MATCH (sp:Speaker {speaker_id: $speaker_id})
        OPTIONAL MATCH (s)-[old:SPOKEN_BY]->(:Speaker)
        DELETE old
        MERGE (s)-[r:SPOKEN_BY]->(sp)
        SET r.confidence = $confidence, r.method = $method, r.locked = false
        """
        await tx.run(
            query,
            aggregate_id=event.aggregate_id,
            speaker_id=data["speaker_id"],
            confidence=data["confidence"],
            method=data["method"],
        )


class SpeakerReattributedHandler(BaseProjectionHandler):
    """Human correction: reattribute a fragment and lock it."""

    async def apply(self, tx, event: EventEnvelope):
        data = event.data
        query = """
        MATCH (s:Sentence {aggregate_id: $aggregate_id})
        MATCH (sp:Speaker {speaker_id: $new_speaker_id})
        OPTIONAL MATCH (s)-[old:SPOKEN_BY]->(:Speaker)
        DELETE old
        MERGE (s)-[r:SPOKEN_BY]->(sp)
        SET r.confidence = 1.0, r.method = 'human', r.locked = true
        """
        await tx.run(
            query,
            aggregate_id=event.aggregate_id,
            new_speaker_id=data["new_speaker_id"],
        )
```

In `src/projections/bootstrap.py`, inside `create_handler_registry` after the existing sentence handler registrations, add (with matching imports at the top of the file):

```python
    # Speaker handlers (Layer 1)
    registry.register("SpeakerCreated", SpeakerCreatedHandler(parked_events_manager))
    registry.register("SpeakerRenamed", SpeakerRenamedHandler(parked_events_manager))
    registry.register("SpeakerMerged", SpeakerMergedHandler(parked_events_manager))
    registry.register("SpeakerAttributed", SpeakerAttributedHandler(parked_events_manager))
    registry.register("SpeakerReattributed", SpeakerReattributedHandler(parked_events_manager))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/projections -m "not integration" -q`
Expected: PASS, no regressions.

- [ ] **Step 5: Commit**

```bash
git add src/projections tests/projections/test_speaker_handlers.py
git commit -m "feat: Speaker projection handlers and registration"
```

---

### Task 12: Projection handlers — Utterance overlay

**Files:**
- Create: `src/projections/handlers/utterance_handlers.py`
- Modify: `src/projections/bootstrap.py`
- Test: `tests/projections/test_utterance_handlers.py`

**Interfaces:**
- Produces Neo4j structure:
  - `(:Utterance {utterance_id, interview_id, confidence})`
  - `(:Speaker)-[:SPOKE]->(:Utterance)`
  - `(:Sentence)-[:PART_OF_UTTERANCE {position}]->(:Utterance)`
  - `(:Utterance)-[:INTERRUPTS {at_fragment_id}]->(:Utterance)`
- Handlers: `UtteranceIdentifiedHandler`, `InterruptionRecordedHandler`, `StitchRemovedHandler` (all Interview stream).

- [ ] **Step 1: Write the failing tests** (`tests/projections/test_utterance_handlers.py`)

```python
from unittest.mock import AsyncMock

import pytest

from src.events.envelope import AggregateType, EventEnvelope
from src.projections.handlers.utterance_handlers import (
    InterruptionRecordedHandler,
    StitchRemovedHandler,
    UtteranceIdentifiedHandler,
)

IID = "22222222-2222-2222-2222-222222222222"
SP1 = "33333333-3333-3333-3333-333333333333"
U1 = "55555555-5555-5555-5555-555555555555"
U2 = "66666666-6666-6666-6666-666666666666"
F1 = "77777777-7777-7777-7777-777777777771"
F2 = "77777777-7777-7777-7777-777777777772"


def make_event(event_type, data, version=3):
    return EventEnvelope(
        event_type=event_type,
        aggregate_type=AggregateType.INTERVIEW,
        aggregate_id=IID,
        version=version,
        data=data,
    )


@pytest.mark.asyncio
async def test_utterance_identified_links_fragments_with_position():
    handler = UtteranceIdentifiedHandler()
    tx = AsyncMock()
    event = make_event(
        "UtteranceIdentified",
        {"utterance_id": U1, "speaker_id": SP1, "fragment_ids": [F1, F2], "confidence": 0.75},
    )
    await handler.apply(tx, event)
    query = tx.run.call_args[0][0]
    params = tx.run.call_args[1]
    assert "PART_OF_UTTERANCE" in query
    assert "SPOKE" in query
    assert params["fragments"] == [
        {"id": F1, "position": 0},
        {"id": F2, "position": 1},
    ]


@pytest.mark.asyncio
async def test_interruption_recorded_creates_edge():
    handler = InterruptionRecordedHandler()
    tx = AsyncMock()
    event = make_event(
        "InterruptionRecorded",
        {"interrupting_utterance_id": U2, "interrupted_utterance_id": U1, "at_fragment_id": F2},
    )
    await handler.apply(tx, event)
    assert "INTERRUPTS" in tx.run.call_args[0][0]


@pytest.mark.asyncio
async def test_stitch_removed_detaches_utterance():
    handler = StitchRemovedHandler()
    tx = AsyncMock()
    event = make_event("StitchRemoved", {"utterance_id": U1, "reason": "wrong"})
    await handler.apply(tx, event)
    assert "DETACH DELETE" in tx.run.call_args[0][0]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/projections/test_utterance_handlers.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement** (`src/projections/handlers/utterance_handlers.py`)

```python
"""Projection handlers for utterance stitching events (overlay on fragments)."""

import logging

from src.events.envelope import EventEnvelope

from .base_handler import BaseProjectionHandler

logger = logging.getLogger(__name__)


class UtteranceIdentifiedHandler(BaseProjectionHandler):
    """Creates an Utterance node and PART_OF_UTTERANCE overlay edges."""

    async def apply(self, tx, event: EventEnvelope):
        data = event.data
        fragments = [
            {"id": fid, "position": pos} for pos, fid in enumerate(data["fragment_ids"])
        ]
        query = """
        MATCH (sp:Speaker {speaker_id: $speaker_id})
        MERGE (u:Utterance {utterance_id: $utterance_id})
        SET u.interview_id = $interview_id, u.confidence = $confidence
        MERGE (sp)-[:SPOKE]->(u)
        WITH u
        UNWIND $fragments AS frag
        MATCH (s:Sentence {aggregate_id: frag.id})
        MERGE (s)-[p:PART_OF_UTTERANCE]->(u)
        SET p.position = frag.position
        """
        await tx.run(
            query,
            utterance_id=data["utterance_id"],
            speaker_id=data["speaker_id"],
            interview_id=event.aggregate_id,
            confidence=data["confidence"],
            fragments=fragments,
        )
        logger.info(f"Applied UtteranceIdentified for utterance {data['utterance_id']}")


class InterruptionRecordedHandler(BaseProjectionHandler):
    """Creates INTERRUPTS edge between utterances."""

    async def apply(self, tx, event: EventEnvelope):
        data = event.data
        query = """
        MATCH (a:Utterance {utterance_id: $interrupting})
        MATCH (b:Utterance {utterance_id: $interrupted})
        MERGE (a)-[r:INTERRUPTS]->(b)
        SET r.at_fragment_id = $at_fragment_id
        """
        await tx.run(
            query,
            interrupting=data["interrupting_utterance_id"],
            interrupted=data["interrupted_utterance_id"],
            at_fragment_id=data["at_fragment_id"],
        )


class StitchRemovedHandler(BaseProjectionHandler):
    """Human correction: removes an utterance overlay node entirely."""

    async def apply(self, tx, event: EventEnvelope):
        query = """
        MATCH (u:Utterance {utterance_id: $utterance_id})
        DETACH DELETE u
        """
        await tx.run(query, utterance_id=event.data["utterance_id"])
```

Register in `create_handler_registry` in `src/projections/bootstrap.py`:

```python
    # Utterance handlers (Layer 1)
    registry.register("UtteranceIdentified", UtteranceIdentifiedHandler(parked_events_manager))
    registry.register("InterruptionRecorded", InterruptionRecordedHandler(parked_events_manager))
    registry.register("StitchRemoved", StitchRemovedHandler(parked_events_manager))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/projections -m "not integration" -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/projections tests/projections/test_utterance_handlers.py
git commit -m "feat: Utterance projection handlers for stitching overlay"
```

---

### Task 13: Correction API — speakers and stitches

**Files:**
- Create: `src/api/routers/speakers.py`
- Modify: `src/main.py` (include router — follow how `edits`/`files`/`analysis` routers are included)
- Test: `tests/api/test_speakers_router.py`

**Interfaces:**
- Consumes: repositories (`get_interview_repository`, `get_sentence_repository`), aggregate correction methods (Tasks 4–6), `Actor`/`ActorType` (human).
- Produces endpoints (all return `202 Accepted` with `{"status": "accepted", "version": <aggregate version>}`, matching the edits API convention; 404 when the aggregate is missing; 409 on domain `ValueError`):
  - `POST /speakers/{interview_id}/{speaker_id}/rename` — body `{"new_display_name": str}`
  - `POST /speakers/{interview_id}/merge` — body `{"surviving_speaker_id": str, "merged_speaker_id": str}`
  - `POST /speakers/{interview_id}/split` — body `{"new_handle": str, "new_display_name": str, "fragment_indices": [int]}`. Composition, no new event type: `add_speaker` (provisional=False, confidence=1.0, method="human") on the Interview, then `reattribute_speaker` on each listed fragment (sentence UUID derived via `uuid5(f"{interview_id}:{index}")`).
  - `POST /speakers/{interview_id}/fragments/{index}/reattribute` — body `{"new_speaker_id": str}` → `Sentence.reattribute_speaker`
  - `DELETE /stitches/{interview_id}/{utterance_id}` — optional body `{"reason": str}` → `Interview.remove_stitch`
- All events carry `actor=Actor(actor_type=ActorType.HUMAN, user_id="api")`.

- [ ] **Step 1: Study the existing pattern, then write the failing tests**

Read `src/api/routers/edits.py` and `tests/api/` first; mirror the dependency-injection and error-mapping style exactly. Then write `tests/api/test_speakers_router.py`:

```python
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.main import app

IID = "22222222-2222-2222-2222-222222222222"
SP1 = "33333333-3333-3333-3333-333333333333"
SP2 = "44444444-4444-4444-4444-444444444444"


@pytest.fixture
def client():
    return TestClient(app)


def make_interview_mock(version=5):
    interview = MagicMock()
    interview.version = version
    return interview


def test_rename_speaker_returns_202(client):
    interview = make_interview_mock()
    repo = MagicMock()
    repo.load = AsyncMock(return_value=interview)
    repo.save = AsyncMock()
    with patch("src.api.routers.speakers.get_interview_repository", return_value=repo):
        resp = client.post(
            f"/speakers/{IID}/{SP1}/rename", json={"new_display_name": "Dana"}
        )
    assert resp.status_code == 202
    interview.rename_speaker.assert_called_once()


def test_rename_unknown_interview_returns_404(client):
    repo = MagicMock()
    repo.load = AsyncMock(return_value=None)
    with patch("src.api.routers.speakers.get_interview_repository", return_value=repo):
        resp = client.post(
            f"/speakers/{IID}/{SP1}/rename", json={"new_display_name": "Dana"}
        )
    assert resp.status_code == 404


def test_merge_speaker_domain_error_returns_409(client):
    interview = make_interview_mock()
    interview.merge_speakers.side_effect = ValueError("Cannot merge a speaker into itself")
    repo = MagicMock()
    repo.load = AsyncMock(return_value=interview)
    with patch("src.api.routers.speakers.get_interview_repository", return_value=repo):
        resp = client.post(
            f"/speakers/{IID}/merge",
            json={"surviving_speaker_id": SP1, "merged_speaker_id": SP1},
        )
    assert resp.status_code == 409
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/api/test_speakers_router.py -v`
Expected: FAIL (`ModuleNotFoundError` or 404s from unregistered routes)

- [ ] **Step 3: Implement** (`src/api/routers/speakers.py`)

```python
"""API endpoints for correcting speaker attribution and stitching (Layer 1)."""

import uuid
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.events.envelope import Actor, ActorType
from src.events.repository import get_interview_repository, get_sentence_repository

router = APIRouter()

HUMAN = Actor(actor_type=ActorType.HUMAN, user_id="api")


class RenameSpeakerRequest(BaseModel):
    new_display_name: str = Field(..., min_length=1)


class MergeSpeakersRequest(BaseModel):
    surviving_speaker_id: str
    merged_speaker_id: str


class SplitSpeakerRequest(BaseModel):
    new_handle: str = Field(..., min_length=1)
    new_display_name: str = Field(..., min_length=1)
    fragment_indices: List[int] = Field(..., min_length=1)


class ReattributeRequest(BaseModel):
    new_speaker_id: str


class RemoveStitchRequest(BaseModel):
    reason: Optional[str] = None


def _accepted(version: int) -> dict:
    return {"status": "accepted", "version": version}


async def _load_interview(interview_id: str):
    repo = get_interview_repository()
    interview = await repo.load(interview_id)
    if interview is None:
        raise HTTPException(status_code=404, detail="Interview not found")
    return repo, interview


@router.post("/speakers/{interview_id}/{speaker_id}/rename", status_code=202)
async def rename_speaker(interview_id: str, speaker_id: str, body: RenameSpeakerRequest):
    repo, interview = await _load_interview(interview_id)
    try:
        interview.rename_speaker(speaker_id, body.new_display_name, actor=HUMAN)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    await repo.save(interview)
    return _accepted(interview.version)


@router.post("/speakers/{interview_id}/merge", status_code=202)
async def merge_speakers(interview_id: str, body: MergeSpeakersRequest):
    repo, interview = await _load_interview(interview_id)
    try:
        interview.merge_speakers(
            body.surviving_speaker_id, body.merged_speaker_id, actor=HUMAN
        )
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    await repo.save(interview)
    return _accepted(interview.version)


@router.post("/speakers/{interview_id}/split", status_code=202)
async def split_speaker(interview_id: str, body: SplitSpeakerRequest):
    """Create a new speaker and reattribute the listed fragments to them."""
    repo, interview = await _load_interview(interview_id)
    new_speaker_id = str(
        uuid.uuid5(uuid.NAMESPACE_DNS, f"{interview_id}:speaker:{body.new_handle}")
    )
    try:
        interview.add_speaker(
            new_speaker_id,
            handle=body.new_handle,
            display_name=body.new_display_name,
            provisional=False,
            confidence=1.0,
            method="human",
            actor=HUMAN,
        )
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    await repo.save(interview)

    sentence_repo = get_sentence_repository()
    for index in body.fragment_indices:
        sentence_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{interview_id}:{index}"))
        sentence = await sentence_repo.load(sentence_id)
        if sentence is None:
            raise HTTPException(status_code=404, detail=f"Fragment {index} not found")
        sentence.reattribute_speaker(new_speaker_id, actor=HUMAN)
        await sentence_repo.save(sentence)
    return _accepted(interview.version)


@router.post("/speakers/{interview_id}/fragments/{index}/reattribute", status_code=202)
async def reattribute_fragment(interview_id: str, index: int, body: ReattributeRequest):
    sentence_repo = get_sentence_repository()
    sentence_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{interview_id}:{index}"))
    sentence = await sentence_repo.load(sentence_id)
    if sentence is None:
        raise HTTPException(status_code=404, detail="Fragment not found")
    try:
        sentence.reattribute_speaker(body.new_speaker_id, actor=HUMAN)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    await sentence_repo.save(sentence)
    return _accepted(sentence.version)


@router.delete("/stitches/{interview_id}/{utterance_id}", status_code=202)
async def remove_stitch(
    interview_id: str, utterance_id: str, body: Optional[RemoveStitchRequest] = None
):
    repo, interview = await _load_interview(interview_id)
    try:
        interview.remove_stitch(
            utterance_id, reason=body.reason if body else None, actor=HUMAN
        )
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    await repo.save(interview)
    return _accepted(interview.version)
```

In `src/main.py`, mirror the existing router includes (e.g. `app.include_router(edits.router, ...)`) with:

```python
from src.api.routers import speakers
app.include_router(speakers.router, tags=["speakers"])
```

Adjust import style to match how the other routers are imported in `main.py`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/api -m "not integration" -q`
Expected: PASS, no regressions.

- [ ] **Step 5: Commit**

```bash
git add src/api/routers/speakers.py src/main.py tests/api/test_speakers_router.py
git commit -m "feat: correction API for speakers and stitches"
```

---

### Task 14: Golden transcript fixtures and end-to-end unit test

**Files:**
- Create: `tests/fixtures/golden/crosstalk_excerpt.txt`
- Create: `tests/fixtures/golden/crosstalk_expected.json`
- Create: `tests/ingestion/test_golden_transcript.py`

**Interfaces:**
- Consumes: everything from Tasks 1–10.
- Produces: a deterministic regression net for the messy-input path. The LLM is mocked with recorded responses stored in the expected-JSON file, so the test is exact and runs in CI without API keys. (A live-LLM variant is Layer 1 follow-up work once prompts are tuned; see Verification below.)

- [ ] **Step 1: Create the fixture** — `tests/fixtures/golden/crosstalk_excerpt.txt`, the opening of the real Zoom transcript (verbatim from `data/input/GMT20231026-210203_Recording.txt`):

```
 Well, hey, how are you doing? Are you able to hear me? Yep. Hello? I can hear you. Can you hear me? Oh, I can hear you. Can you hear me? Yes. OK, awesome. All right. Sorry, I'm a couple of minutes late. How are you doing? Good. Thank you. How about yourself? I'm really good. I'm really good today. OK, just want to close some things out. But thank you for making time out of your day today to meet with me.
```

- [ ] **Step 2: Create expected data** — `tests/fixtures/golden/crosstalk_expected.json`. Hand-verify against the excerpt: the interviewer (S1) opens, the participant (S2) responds. Include the mocked LLM window response and the structural expectations:

```json
{
  "speaker_window_response": {
    "assignments": [
      {"index": 0, "speaker": "S1", "confidence": 0.85},
      {"index": 1, "speaker": "S1", "confidence": 0.8},
      {"index": 2, "speaker": "S2", "confidence": 0.9},
      {"index": 3, "speaker": "S2", "confidence": 0.6},
      {"index": 4, "speaker": "S2", "confidence": 0.85},
      {"index": 5, "speaker": "S2", "confidence": 0.8},
      {"index": 6, "speaker": "S1", "confidence": 0.8},
      {"index": 7, "speaker": "S1", "confidence": 0.75},
      {"index": 8, "speaker": "S2", "confidence": 0.85},
      {"index": 9, "speaker": "S1", "confidence": 0.8},
      {"index": 10, "speaker": "S1", "confidence": 0.7},
      {"index": 11, "speaker": "S1", "confidence": 0.85},
      {"index": 12, "speaker": "S1", "confidence": 0.8},
      {"index": 13, "speaker": "S2", "confidence": 0.85},
      {"index": 14, "speaker": "S2", "confidence": 0.8},
      {"index": 15, "speaker": "S2", "confidence": 0.8},
      {"index": 16, "speaker": "S1", "confidence": 0.8},
      {"index": 17, "speaker": "S1", "confidence": 0.75},
      {"index": 18, "speaker": "S1", "confidence": 0.8},
      {"index": 19, "speaker": "S1", "confidence": 0.85}
    ]
  },
  "stitch_window_response": {"utterances": [], "interruptions": []},
  "expected": {
    "speaker_count": 2,
    "min_utterances": 6,
    "every_fragment_attributed": true
  }
}
```

Adjust `assignments` length to the actual fragment count produced by the normalizer for the excerpt (run the normalizer once and align indices; spaCy segmentation determines the count — hand-verify each index against the excerpt when aligning).

- [ ] **Step 3: Write the test** (`tests/ingestion/test_golden_transcript.py`)

```python
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.ingestion.orchestrator import IngestionOrchestrator

GOLDEN_DIR = Path(__file__).parent.parent / "fixtures" / "golden"


@pytest.mark.asyncio
async def test_crosstalk_excerpt_end_to_end(tmp_path: Path):
    source = (GOLDEN_DIR / "crosstalk_excerpt.txt").read_text()
    expected = json.loads((GOLDEN_DIR / "crosstalk_expected.json").read_text())

    input_file = tmp_path / "crosstalk.txt"
    input_file.write_text(source)

    async def fake_save(aggregate, expected_version=None):
        aggregate.mark_events_as_committed()

    mock_repo = MagicMock()
    mock_repo.save = AsyncMock(side_effect=fake_save)

    llm_responses = [expected["speaker_window_response"], expected["stitch_window_response"]]

    with patch("src.ingestion.orchestrator.get_interview_repository", return_value=mock_repo), \
         patch("src.ingestion.orchestrator.get_sentence_repository", return_value=mock_repo), \
         patch("src.ingestion.speaker_inference.agent") as sp_agent, \
         patch("src.ingestion.stitcher.agent") as st_agent:
        sp_agent.call_model = AsyncMock(return_value=llm_responses[0])
        st_agent.call_model = AsyncMock(return_value=llm_responses[1])
        orchestrator = IngestionOrchestrator(project_id="golden", map_dir=tmp_path / "maps")
        result = await orchestrator.ingest_file(input_file)

    exp = expected["expected"]
    assert result.speaker_count == exp["speaker_count"]
    assert result.utterance_count >= exp["min_utterances"]

    # The map grounds every fragment: offsets must recover the exact text.
    map_file = tmp_path / "maps" / "crosstalk_map.jsonl"
    entries = [json.loads(line) for line in map_file.read_text().splitlines()]
    assert len(entries) == result.fragment_count
    for entry in entries:
        assert source[entry["start_char"]:entry["end_char"]] == entry["sentence"]
        if exp["every_fragment_attributed"]:
            assert entry["speaker_id"] is not None
            assert entry["utterance_id"] is not None
```

- [ ] **Step 4: Run test, align fixture, verify pass**

Run: `python -m pytest tests/ingestion/test_golden_transcript.py -v`
First run will likely fail on assignment count vs. actual fragment count — print `result.fragment_count`, align `crosstalk_expected.json` indices to the normalizer's actual segmentation (hand-verifying speaker labels per fragment), then re-run.
Expected: PASS

- [ ] **Step 5: Run the full unit suite and commit**

Run: `python -m pytest tests -m "not integration" -q`
Expected: full suite green.

```bash
git add tests/fixtures/golden tests/ingestion/test_golden_transcript.py
git commit -m "test: golden crosstalk transcript fixture for Layer 1 regression"
```

---

### Task 15: Documentation and roadmap update

**Files:**
- Modify: `docs/ROADMAP.md`
- Modify: `docs/architecture/database-schema.md`
- Modify: `README.md` (What It Does + architecture snippet)

**Interfaces:** none (documentation).

- [ ] **Step 1: Update `docs/ROADMAP.md`**

Add a completed **M4.1: Layer 1 — Ingestion, Map, Speaker Genesis & Stitching** milestone section (checklist mirroring Tasks 1–14: normalizer, speaker inference, stitching overlay, projection handlers, correction API, golden fixtures), update the Quick Status table and "Current Phase", and add a Decision Log entry:

```
| <today> | Layer 1 complete: speakers, utterances, offset-grounded map | Spec: docs/superpowers/specs/2026-07-04-mine-layers-design.md |
```

Note in the M3.1 section that vector search will build on fragment/utterance nodes from Layer 1.

- [ ] **Step 2: Update `docs/architecture/database-schema.md`**

Add node type documentation for `:Speaker` and `:Utterance` (properties and relationships as implemented in Tasks 11–12: `HAS_PARTICIPANT`, `SPOKEN_BY`, `SPOKE`, `PART_OF_UTTERANCE`, `INTERRUPTS`), plus the new `Sentence` properties (`start_char`, `end_char`). Add the new event types to the "Event Types Stored" table: `SpeakerCreated`, `SpeakerRenamed`, `SpeakerMerged`, `SpeakerAttributed`, `SpeakerReattributed`, `UtteranceIdentified`, `InterruptionRecorded`, `StitchRemoved`.

- [ ] **Step 3: Update `README.md`**

In "What It Does", add ingestion/speaker/stitching between steps 2 and 3:

```
3. **Attributes** speakers (parsed from labels, or inferred with confidence when absent)
4. **Stitches** interrupted utterances via relationship overlay (verbatim text untouched)
```

(Renumber the following steps.)

- [ ] **Step 4: Verify docs render and commit**

Run: `python -m pytest tests -m "not integration" -q` (final green check)

```bash
git add docs/ROADMAP.md docs/architecture/database-schema.md README.md
git commit -m "docs: Layer 1 milestone, schema additions, README update"
```

---

## Verification (whole-plan)

1. `python -m pytest tests -m "not integration" -q` — entire unit suite green.
2. `make test-infra-up && python -m pytest tests -m integration -q` (optional, needs Docker) — existing integration suite unaffected.
3. Manual smoke (needs ESDB + API key): `python -m src.ingestion data/input/GMT20231026-210203_Recording.txt` then run the projection service and inspect Neo4j: `MATCH (sp:Speaker)-[:SPOKE]->(u:Utterance)<-[:PART_OF_UTTERANCE]-(s:Sentence) RETURN sp.handle, u.utterance_id, s.text LIMIT 20`.
4. Follow-up (not in this plan): live-LLM golden evaluation for prompt tuning; `black` + `flake8` pass on all new files.

## Deferred / explicitly out of scope for Layer 1

- `OVERLAPS` relationship: the flat-text Layer 0 output gives no reliable simultaneity signal; deferred until timestamped input formats are added to the normalizer (spec lists it as optional).
- Interview front-matter YAML capture and participant seeding: metadata dict on `InterviewCreated` carries `content_hash`/`format` now; the OKF front-matter schema lands with the Layer 5 exporter work.
- Speaker inference LLM-based reconciliation for ambiguous overlaps (deterministic voting suffices until golden evaluation says otherwise).
- `StitchCorrected` as a distinct event: v1 models stitch correction as `StitchRemoved` (this plan) followed by a future "identify utterance manually" endpoint. Editing an utterance's fragment set in place adds event complexity with no consumer yet.
```
