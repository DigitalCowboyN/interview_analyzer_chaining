# M4.4: OKF Export + Richer Queries Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the mine consumable — OKF v0.1 bundles (one interview × one lens) via CLI + zip API, transcript front-matter capture with speaker seeding on ingest, and three Neo4j-backed query endpoints.

**Architecture:** Read-side exporter over Neo4j: `src/export/reader.py` is the single Cypher layer (shared by the bundle and the query endpoints), `renderer.py` is pure dicts→files, `bundler.py` orchestrates with a projection-lag consistency guard that compares aggregate lens-item ids against the graph. Front matter integrates at `normalize()` (offsets stay absolute into the unmodified source) and seeds speaker genesis. No new event types anywhere.

**Tech Stack:** Python 3.10, Pydantic v2, PyYAML (already a dependency), Neo4j 5.26, FastAPI. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-07-10-okf-export-design.md` (approved). OKF rules verified against `github.com/GoogleCloudPlatform/knowledge-catalog/okf/SPEC.md`.

## Global Constraints

- **OKF v0.1 conformance:** every generated non-reserved `.md` file has YAML frontmatter with a non-empty `type`; `index.md` is reserved and has NO frontmatter (plain listing with relative links + descriptions); `log.md` is reserved (history grouped by ISO date, newest first); internal links are bundle-absolute (`/`-prefixed).
- **Zero per-lens code:** the exporter renders lens items from `node_type` + node properties + the lens YAML's `projects_to`. No `if node_type == "Decision"` anywhere.
- **No new event types.** Front matter stores through the existing `InterviewCreated` payload; export is pure egress.
- **Offsets invariant:** with front matter present, `source[start_char:end_char] == fragment.text` still holds against the UNMODIFIED source file.
- **Naming (mechanical):** item directory = `kebab_case(node_type) + "s"`; item filename = `<kebab(node_type)>-<first 8 hex of item_id>.md`; speaker/entity slugs lowercase with non-alphanumerics → `-`.
- **Consistency guard expected set:** aggregate items for the lens where `lens_version == lens_runs[lens]` PLUS locked items of any version (aggregate `lens_items` alone retains superseded ids the graph correctly deleted).
- **Interview header renders from the aggregate** (which the guard loads anyway) — the Interview *projection* has no metadata/participants and is not extended.
- Front-matter parsing is tolerant: malformed YAML → warning + treat as body; ingest never fails on front matter. Participants are hints, never constraints; ambiguous first-name matches don't seed.
- Query endpoints paginate with `limit` (default 50) / `offset` (default 0).
- Environment: `set -a; source .env; set +a; ~/.pyenv/versions/3.10.7/bin/python -m pytest ... -q --no-cov` (plain `python` not on PATH). Integration env: `NEO4J_URI=bolt://localhost:7688 NEO4J_USER=neo4j NEO4J_PASSWORD=testpassword ESDB_CONNECTION_STRING="esdb://localhost:2113?tls=false"`.
- Style: `black` (only keep files that were already black-clean clean; repo is not uniformly formatted), `flake8` (max line 120) must pass on all touched files. Existing patterns: payload models + aggregate command methods; routers mirror `src/api/routers/speakers.py`; CLIs mirror `src/lens/__main__.py`.

## Graph shape reference (as projected today — read-only for this milestone)

```
(:Project {project_id})-[:CONTAINS_INTERVIEW]->(:Interview {interview_id, title, source, status})
(:Interview)-[:HAS_SENTENCE]->(:Sentence {sentence_id, aggregate_id, text, sequence_order})
(:Sentence)-[:SPOKEN_BY]->(:Speaker {speaker_id, handle, display_name, provisional, merged_into, interview_id})
(:Interview)-[:HAS_PARTICIPANT]->(:Speaker);  (:Speaker)-[:SPOKE]->(:Utterance {utterance_id})
(:Sentence)-[:PART_OF_UTTERANCE]->(:Utterance)
(:Sentence)-[:HAS_ANALYSIS]->(:Analysis {analysis_id, model, provider, confidence,
    dimension_confidences (json str), flags (json str), created_at})
(:Analysis)-[:HAS_FUNCTION]->(:FunctionType {name}); -[:HAS_STRUCTURE]->(:StructureType {name});
    -[:HAS_PURPOSE]->(:Purpose {name}); -[:MENTIONS_TOPIC]->(:Topic {name});
    -[:MENTIONS_OVERALL_KEYWORD]->(:Keyword {text}); -[:MENTIONS_DOMAIN_KEYWORD]->(:DomainKeyword {text})
(:Sentence)-[:MENTIONS {start, end, text, confidence}]->(:Entity {surface, entity_type})
(:Claim {claim_id, text, kind, confidence, model, provider, interview_id})-[:MADE_BY]->(:Speaker);
    (:Claim)-[:SUPPORTED_BY]->(:Sentence)
(:LensItem:<Label> {item_id, lens, lens_version, node_type, confidence, model, provider,
    interview_id, locked?, <extracted fields>})-[:SUPPORTED_BY]->(:Sentence);
    (:LensItem)-[:<REL from lens YAML>]->(:Speaker)
```

## File Structure (new/major)

```
src/ingestion/front_matter.py    # parse_front_matter(text) -> (dict|None, body_start)
src/ingestion/models.py          # NormalizedTranscript.front_matter
src/ingestion/normalizer.py      # front-matter-aware normalize (offset shift)
src/ingestion/orchestrator.py    # title/started_at/metadata from fm; speaker seeding
src/ingestion/speaker_inference.py  # infer(fragments, participants=None) prompt hint
prompts/ingestion_prompts.yaml   # {participants_hint} placeholder in speaker_window
src/export/
  __init__.py                    # empty
  reader.py                      # ALL Layer 5 Cypher (bundle + queries), returns dicts
  renderer.py                    # pure: dicts -> [(relative_path, content)]
  bundler.py                     # OkfExporter (guard, read, render, write/zip, log.md)
  __main__.py                    # python -m src.export <interview_id> <lens> [--out] [--zip]
src/api/routers/exports.py       # GET /exports/{interview_id}/{lens_name} -> zip
src/api/routers/queries.py       # items / worklist / rollup endpoints
src/main.py                      # include both routers
```

---

### Task 1: Front-matter parser + offset-preserving normalize

**Files:**
- Create: `src/ingestion/front_matter.py`
- Modify: `src/ingestion/models.py` (add `front_matter` to `NormalizedTranscript`)
- Modify: `src/ingestion/normalizer.py` (parse first, segment body, shift offsets)
- Test: `tests/ingestion/test_front_matter.py`

**Interfaces:**
- Produces: `parse_front_matter(text: str) -> Tuple[Optional[Dict[str, Any]], int]` — `(dict, body_start_offset)` when a valid leading `---` YAML block exists and parses to a mapping; `(None, 0)` otherwise (absent, malformed, or non-mapping — malformed logs a warning). `body_start` is the offset of the first character AFTER the closing `---` line.
- Produces: `NormalizedTranscript.front_matter: Optional[Dict[str, Any]] = None`.
- `normalize(text)` unchanged signature; fragments' offsets remain absolute into the ORIGINAL text.

- [ ] **Step 1: Write the failing tests** (`tests/ingestion/test_front_matter.py`):

```python
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
```

- [ ] **Step 2: Run to verify fail** — `pytest tests/ingestion/test_front_matter.py -q --no-cov` → ModuleNotFoundError.

- [ ] **Step 3: Implement.** `src/ingestion/front_matter.py`:

```python
"""Tolerant OKF-compatible front-matter parsing for incoming transcripts.

Malformed or non-mapping YAML degrades to "no front matter" with a warning;
ingest never fails on front matter.
"""

import re
from typing import Any, Dict, Optional, Tuple

import yaml

from src.utils.logger import get_logger

logger = get_logger()

_FRONT_MATTER_RE = re.compile(r"\A---\r?\n(.*?)\r?\n---\r?\n", re.DOTALL)


def parse_front_matter(text: str) -> Tuple[Optional[Dict[str, Any]], int]:
    """Return (front_matter, body_start). (None, 0) when absent or invalid."""
    m = _FRONT_MATTER_RE.match(text)
    if not m:
        return None, 0
    try:
        data = yaml.safe_load(m.group(1))
    except yaml.YAMLError as exc:
        logger.warning(f"Malformed front matter ignored ({exc}); treating as body")
        return None, 0
    if not isinstance(data, dict):
        logger.warning("Front matter is not a mapping; treating as body")
        return None, 0
    return data, m.end()
```

`models.py`: add to `NormalizedTranscript`:

```python
    front_matter: Optional[Dict[str, Any]] = Field(
        None, description="Parsed leading YAML block, if the source had one"
    )
```

(add `Any, Dict, Optional` to the models.py imports as needed).

`normalizer.py` — `normalize()` becomes:

```python
def normalize(text: str) -> NormalizedTranscript:
    """Normalize raw transcript text into a NormalizedTranscript."""
    content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    front_matter, body_start = parse_front_matter(text)
    body = text[body_start:]
    fmt = detect_format(body)

    if fmt == TranscriptFormat.LABELED:
        fragments, labels = _parse_labeled(body)
    else:
        fragments = [
            RawFragment(text=t, start_char=s, end_char=e, sequence_order=i)
            for i, (t, s, e) in enumerate(segment_text_with_offsets(body))
        ]
        labels = []

    if body_start:
        # Offsets must stay absolute into the UNMODIFIED source text.
        fragments = [
            f.model_copy(
                update={
                    "start_char": f.start_char + body_start,
                    "end_char": f.end_char + body_start,
                }
            )
            for f in fragments
        ]

    return NormalizedTranscript(
        content_hash=content_hash, format=fmt, fragments=fragments,
        speaker_labels=labels, front_matter=front_matter,
    )
```

(import `parse_front_matter` from `.front_matter`; content_hash stays the hash of the FULL text).

- [ ] **Step 4: Run** — `pytest tests/ingestion -q --no-cov` → green (new + all pre-existing normalizer tests).
- [ ] **Step 5: Commit** — `git commit -am "feat: OKF front-matter parsing in normalize (offsets stay absolute)"`

---

### Task 2: Orchestrator metadata + speaker seeding

**Files:**
- Modify: `src/ingestion/orchestrator.py` (title/started_at/metadata from front matter; labeled seeding)
- Modify: `src/ingestion/speaker_inference.py` (`infer(fragments, participants=None)` prompt hint)
- Modify: `prompts/ingestion_prompts.yaml` (`{participants_hint}` placeholder in `speaker_window`)
- Test: `tests/ingestion/test_front_matter_seeding.py`

**Interfaces:**
- Consumes: `NormalizedTranscript.front_matter` (Task 1).
- Produces: `_match_participant(handle: str, participants: List[str]) -> Optional[str]` (module-level in orchestrator) — case-insensitive full-name match, else first-name match when exactly ONE participant has that first name, else None.
- `SpeakerInferenceService.infer(self, fragments, participants: Optional[List[str]] = None)` — participants render into the prompt as `Known participants: A, B.\n` via a `{participants_hint}` template placeholder (empty string when absent).
- Seeded labeled speakers: `display_name=<participant full name>`, `provisional=False`, `method="front_matter"`, `confidence=1.0`.

- [ ] **Step 1: Write the failing tests** (`tests/ingestion/test_front_matter_seeding.py`):

```python
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.ingestion.orchestrator import IngestionOrchestrator, _match_participant

FM_TEXT = """---
title: Q3 Vendor Selection
date: 2026-07-01
participants: [Alice Johnson, Bob Reyes]
---
Alice: We will go with vendor X for the telemetry pipeline work.
Bob Reyes: Sounds good to me, let us proceed with it.
"""


def test_match_participant_full_first_and_ambiguous():
    participants = ["Alice Johnson", "Bob Reyes", "Alice Smith"]
    assert _match_participant("Bob Reyes", participants) == "Bob Reyes"
    assert _match_participant("bob", participants) == "Bob Reyes"      # unique first name
    assert _match_participant("Alice", participants) is None           # ambiguous
    assert _match_participant("Carol", participants) is None           # unlisted


@pytest.mark.asyncio
async def test_labeled_ingest_seeds_confirmed_speakers(tmp_path):
    input_file = tmp_path / "m.txt"
    input_file.write_text(FM_TEXT)
    saved = {}

    async def fake_save(agg, **k):
        saved["interview"] = agg
        agg.mark_events_as_committed()

    interview_repo = MagicMock(save=AsyncMock(side_effect=fake_save))
    sentence_repo = MagicMock(save=AsyncMock())
    with patch("src.ingestion.orchestrator.get_interview_repository", return_value=interview_repo), \
         patch("src.ingestion.orchestrator.get_sentence_repository", return_value=sentence_repo):
        orch = IngestionOrchestrator(project_id="p", map_dir=tmp_path / "maps")
        await orch.ingest_file(input_file)

    interview = saved["interview"]
    assert interview.title == "Q3 Vendor Selection"
    assert interview.metadata["front_matter"]["participants"] == ["Alice Johnson", "Bob Reyes"]
    by_handle = {info["handle"]: info for info in interview.speakers.values()}
    assert by_handle["Alice"]["display_name"] == "Alice Johnson"   # first-name seed
    assert by_handle["Bob Reyes"]["display_name"] == "Bob Reyes"   # full-name seed
    assert by_handle["Alice"]["provisional"] is False


@pytest.mark.asyncio
async def test_inference_prompt_receives_participants_hint():
    from src.ingestion.models import RawFragment
    from src.ingestion.speaker_inference import SpeakerInferenceService

    service = SpeakerInferenceService()
    captured = {}

    async def fake_call(prompt, schema=None):
        captured["prompt"] = prompt
        return {"assignments": []}

    with patch("src.ingestion.speaker_inference.get_failover_agent") as get_agent:
        get_agent.return_value.call_model = AsyncMock(side_effect=fake_call)
        frags = [RawFragment(text="Hello there.", start_char=0, end_char=12, sequence_order=0)]
        await service.infer(frags, participants=["Alice Johnson", "Bob Reyes"])
    assert "Known participants: Alice Johnson, Bob Reyes." in captured["prompt"]
```

NOTE for the implementer: before writing Step 3, read `src/ingestion/speaker_inference.py:48-106` — the test above patches whatever agent-acquisition seam exists there (`get_failover_agent` at module import path); adjust the patch target to the actual symbol if it differs, keeping the assertion about the prompt string.

- [ ] **Step 2: Run to verify fail.**

- [ ] **Step 3: Implement.**

`orchestrator.py` — module-level helper:

```python
def _match_participant(handle: str, participants: List[str]) -> Optional[str]:
    """Full-name match, else UNIQUE first-name match; ambiguity never seeds."""
    needle = handle.strip().lower()
    full = [p for p in participants if p.strip().lower() == needle]
    if len(full) == 1:
        return full[0]
    first = [p for p in participants if p.split()[0].strip().lower() == needle]
    if len(first) == 1:
        return first[0]
    return None
```

In `ingest_file`, after `transcript = normalize(text)`:

```python
        fm = transcript.front_matter or {}
        participants = [p for p in fm.get("participants") or [] if isinstance(p, str)]
        started_at = _parse_fm_date(fm.get("date"))

        interview = Interview(interview_id)
        interview.create(
            title=fm.get("title") or file_path.name,
            source=str(file_path),
            started_at=started_at,
            metadata={
                "content_hash": transcript.content_hash,
                "format": transcript.format.value,
                "fragment_count": len(transcript.fragments),
                **({"front_matter": fm} if fm else {}),
            },
            ...
        )
```

with:

```python
def _parse_fm_date(value: Any) -> Optional[datetime]:
    """YAML gives date/datetime objects or strings; anything unparseable -> None."""
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day)
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
    return None
```

`_resolve_speakers` gains a `participants: List[str]` parameter (orchestrator passes it). In the LABELED branch, per-handle seeding replaces the uniform values:

```python
            provisional, method = False, "parsed"
            confidences = {h: 1.0 for h in handles}
            display_names = {}
            methods = {}
            for h in handles:
                matched = _match_participant(h, participants) if participants else None
                display_names[h] = matched or h
                methods[h] = "front_matter" if matched else method
```

and the `add_speaker` loop uses `display_name=display_names.get(handle, handle)` and `method=... methods.get(handle, method)` for the labeled path (inference path unchanged; the `S?` placeholder rules stay exactly as they are). The FLAT branch passes participants through: `result = await self.inference.infer(transcript.fragments, participants=participants)`.

`speaker_inference.py`: `infer` and `_infer_window` accept `participants: Optional[List[str]] = None`; `_infer_window` builds:

```python
        hint = f"Known participants: {', '.join(participants)}.\n" if participants else ""
        prompt = self.prompts["speaker_window"]["prompt"].format(
            fragments=numbered, participants_hint=hint
        )
```

`prompts/ingestion_prompts.yaml`: add the `{participants_hint}` placeholder to the `speaker_window` prompt template on its own line near the top of the instructions (read the existing template first; insert, don't rewrite the voice).

- [ ] **Step 4: Run** — `pytest tests/ingestion -q --no-cov` → green including all pre-existing orchestrator/inference tests.
- [ ] **Step 5: Commit** — `git commit -am "feat: front-matter metadata + participant speaker seeding on ingest"`

---

### Task 3: Export reader — the single Cypher layer

**Files:**
- Create: `src/export/__init__.py` (empty), `src/export/reader.py`
- Test: `tests/export/__init__.py` (empty), `tests/export/test_reader.py`

**Interfaces:**
- Produces (all `async`, all take an active Neo4j `session` first — callers own session lifecycle):
  - `transcript_rows(session, interview_id) -> List[dict]` — `{sentence_id, sequence_order, text, speaker_id, speaker, utterance_id}` ordered by `sequence_order` (`speaker` = display_name or None; `utterance_id` None when unstitched).
  - `speaker_rows(session, interview_id) -> List[dict]` — `{speaker_id, handle, display_name, provisional}`; excludes merged speakers.
  - `lens_item_rows(session, interview_id, lens, node_type=None, min_confidence=None, limit=None, offset=0) -> List[dict]` — `{item_id, node_type, lens_version, confidence, model, provider, locked, props (full property map), speaker_links: [{relationship, speaker_id, display_name}], supporting_fragment_ids: [str]}` ordered by node_type then item_id.
  - `claim_rows(session, interview_id) -> List[dict]` — `{claim_id, text, kind, confidence, model, provider, speaker_id, speaker, supporting_fragment_ids}`.
  - `entity_rows(session, interview_id) -> List[dict]` — `{surface, entity_type, mentions: [{sentence_id, start, end, text, confidence}]}` (entities reached via this interview's sentences only).
  - `analysis_rows(session, interview_id) -> List[dict]` — `{sequence_order, text, speaker, function, structure, purpose, topics: [str], keywords: [str], confidence, flags (json str|None)}` using the LATEST Analysis per sentence (`ORDER BY a.created_at DESC` → first).
- Consumers: Task 5 bundler, Task 7 queries, Task 8 integration.

- [ ] **Step 1: Write the failing tests** (`tests/export/test_reader.py`) — mocked-session shape tests (the real-Cypher correctness check is Task 8's integration run):

```python
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.export import reader

IID = "22222222-2222-2222-2222-222222222222"


def make_session(records):
    async def aiter(self):
        for r in records:
            yield r
    result = MagicMock()
    result.__aiter__ = aiter
    session = MagicMock()
    session.run = AsyncMock(return_value=result)
    return session


@pytest.mark.asyncio
async def test_transcript_rows_query_and_shape():
    session = make_session([
        {"sentence_id": "f1", "sequence_order": 0, "text": "Hi.",
         "speaker_id": "sp1", "speaker": "Alice", "utterance_id": "u1"},
    ])
    rows = await reader.transcript_rows(session, IID)
    assert rows[0]["speaker"] == "Alice"
    query = session.run.call_args[0][0]
    assert "HAS_SENTENCE" in query and "SPOKEN_BY" in query and "PART_OF_UTTERANCE" in query
    assert session.run.call_args.kwargs["interview_id"] == IID


@pytest.mark.asyncio
async def test_lens_item_rows_optional_filters():
    session = make_session([])
    await reader.lens_item_rows(session, IID, "meeting_minutes",
                                node_type="Decision", min_confidence=0.5, limit=10)
    query = session.run.call_args[0][0]
    kwargs = session.run.call_args.kwargs
    assert "LensItem" in query and "SUPPORTED_BY" in query
    assert kwargs["lens"] == "meeting_minutes"
    assert kwargs["node_type"] == "Decision" and kwargs["min_confidence"] == 0.5


@pytest.mark.asyncio
async def test_analysis_rows_takes_latest_analysis():
    session = make_session([])
    await reader.analysis_rows(session, IID)
    query = session.run.call_args[0][0]
    assert "HAS_ANALYSIS" in query and "created_at DESC" in query
```

- [ ] **Step 2: Run to verify fail** — ModuleNotFoundError.

- [ ] **Step 3: Implement** `src/export/reader.py`. Representative queries (write all six; the rest follow these patterns exactly):

```python
"""All Layer 5 Cypher: the exporter and the query endpoints read through here.

Every function takes an active async Neo4j session and returns plain dicts —
no rendering, no domain objects, no session management.
"""

from typing import Any, Dict, List, Optional


async def transcript_rows(session, interview_id: str) -> List[Dict[str, Any]]:
    query = """
    MATCH (i:Interview {interview_id: $interview_id})-[:HAS_SENTENCE]->(s:Sentence)
    OPTIONAL MATCH (s)-[:SPOKEN_BY]->(sp:Speaker)
    OPTIONAL MATCH (s)-[:PART_OF_UTTERANCE]->(u:Utterance)
    RETURN s.sentence_id AS sentence_id, s.sequence_order AS sequence_order,
           s.text AS text, sp.speaker_id AS speaker_id,
           sp.display_name AS speaker, u.utterance_id AS utterance_id
    ORDER BY s.sequence_order
    """
    result = await session.run(query, interview_id=interview_id)
    return [dict(r) async for r in result]


async def lens_item_rows(
    session,
    interview_id: str,
    lens: str,
    node_type: Optional[str] = None,
    min_confidence: Optional[float] = None,
    limit: Optional[int] = None,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    query = """
    MATCH (n:LensItem {interview_id: $interview_id, lens: $lens})
    WHERE ($node_type IS NULL OR n.node_type = $node_type)
      AND ($min_confidence IS NULL OR n.confidence >= $min_confidence)
    OPTIONAL MATCH (n)-[r]->(sp:Speaker)
    OPTIONAL MATCH (n)-[:SUPPORTED_BY]->(s:Sentence)
    WITH n,
         [x IN collect(DISTINCT {relationship: type(r), speaker_id: sp.speaker_id,
                                 display_name: sp.display_name})
          WHERE x.speaker_id IS NOT NULL] AS speaker_links,
         [x IN collect(DISTINCT s.aggregate_id) WHERE x IS NOT NULL] AS supporting
    RETURN n.item_id AS item_id, n.node_type AS node_type,
           n.lens_version AS lens_version, n.confidence AS confidence,
           n.model AS model, n.provider AS provider,
           coalesce(n.locked, false) AS locked, properties(n) AS props,
           speaker_links, supporting AS supporting_fragment_ids
    ORDER BY n.node_type, n.item_id
    SKIP $offset LIMIT $limit
    """
    result = await session.run(
        query, interview_id=interview_id, lens=lens, node_type=node_type,
        min_confidence=min_confidence, offset=offset,
        limit=limit if limit is not None else 10_000,
    )
    return [dict(r) async for r in result]
```

`analysis_rows` uses latest-analysis selection:

```cypher
    MATCH (i:Interview {interview_id: $interview_id})-[:HAS_SENTENCE]->(s:Sentence)
    OPTIONAL MATCH (s)-[:SPOKEN_BY]->(sp:Speaker)
    OPTIONAL MATCH (s)-[:HAS_ANALYSIS]->(a:Analysis)
    WITH s, sp, a ORDER BY a.created_at DESC
    WITH s, sp, collect(a)[0] AS a
    OPTIONAL MATCH (a)-[:HAS_FUNCTION]->(ft:FunctionType)
    OPTIONAL MATCH (a)-[:HAS_STRUCTURE]->(st:StructureType)
    OPTIONAL MATCH (a)-[:HAS_PURPOSE]->(p:Purpose)
    OPTIONAL MATCH (a)-[:MENTIONS_TOPIC]->(t:Topic)
    OPTIONAL MATCH (a)-[:MENTIONS_OVERALL_KEYWORD]->(k:Keyword)
    RETURN s.sequence_order AS sequence_order, s.text AS text,
           sp.display_name AS speaker,
           ft.name AS function, st.name AS structure, p.name AS purpose,
           [x IN collect(DISTINCT t.name) WHERE x IS NOT NULL] AS topics,
           [x IN collect(DISTINCT k.text) WHERE x IS NOT NULL] AS keywords,
           a.confidence AS confidence, a.flags AS flags
    ORDER BY s.sequence_order
```

`speaker_rows`: `MATCH (i:Interview {interview_id: $interview_id})-[:HAS_PARTICIPANT]->(sp:Speaker) WHERE sp.merged_into IS NULL RETURN ... ORDER BY sp.handle`.
`claim_rows`: `MATCH (c:Claim {interview_id: $interview_id}) OPTIONAL MATCH (c)-[:MADE_BY]->(sp) OPTIONAL MATCH (c)-[:SUPPORTED_BY]->(s) ... ORDER BY c.claim_id`.
`entity_rows`: `MATCH (i:Interview {interview_id: $interview_id})-[:HAS_SENTENCE]->(s)-[m:MENTIONS]->(e:Entity) WITH e, collect({sentence_id: s.aggregate_id, start: m.start, end: m.end, text: m.text, confidence: m.confidence}) AS mentions RETURN e.surface AS surface, e.entity_type AS entity_type, mentions ORDER BY e.surface`.

- [ ] **Step 4: Run** — `pytest tests/export -q --no-cov` → green.
- [ ] **Step 5: Commit** — `git commit -am "feat: export reader — single Cypher layer for bundles and queries"`

---

### Task 4: OKF renderer (pure)

**Files:**
- Create: `src/export/renderer.py`
- Test: `tests/export/test_renderer.py`

**Interfaces:**
- Consumes: reader row shapes (Task 3), `LensSpec` (`src/lens/models.py: load_lens`).
- Produces:
  - `render_bundle(header: dict, transcript: List[dict], speakers: List[dict], items: List[dict], claims: List[dict], entities: List[dict], analysis: List[dict], lens: LensSpec, exported_at: str) -> List[Tuple[str, str]]` — `(relative_path, content)` pairs for EVERY file except `log.md` (the bundler owns log history). `header` keys: `interview_id, title, source, started_at (iso str|None), project_id, metadata (dict), participants: [{handle, display_name, provisional}], fragment_count, utterance_count, lens, lens_version`.
  - `slugify(value: str) -> str` — lowercase, `[^a-z0-9]+` → `-`, strip `-`.
  - `item_dir(node_type: str) -> str` — kebab-case + `"s"` (`ActionItem` → `action-items`).
  - `item_filename(node_type: str, item_id: str) -> str` — `f"{kebab(node_type)}-{item_id[:8]}.md"`.
- Reserved-prop split: `props` keys in `{item_id, lens, lens_version, node_type, confidence, model, provider, interview_id, locked, overridden_at, override_note}` are NOT extracted fields; everything else in `props` passes into frontmatter as-is.
- Anchors: utterances get `u-<n>` by order of first appearance in the transcript; unstitched fragments get `f-<sequence_order>`. Rendered as `<a id="u-1"></a>` lines. `render_bundle` builds a `sentence_id -> anchor` map internally and uses it for all grounding links.

- [ ] **Step 1: Write the failing tests** (`tests/export/test_renderer.py`):

```python
import yaml

from src.export.renderer import item_dir, item_filename, render_bundle, slugify
from src.lens.models import load_lens

HEADER = {
    "interview_id": "iid-1", "title": "Q3 Vendor Selection", "source": "m.txt",
    "started_at": "2026-07-01T00:00:00", "project_id": "telemetry",
    "metadata": {"front_matter": {"participants": ["Alice Johnson"]}},
    "participants": [{"handle": "Alice", "display_name": "Alice Johnson", "provisional": False}],
    "fragment_count": 2, "utterance_count": 1,
    "lens": "meeting_minutes", "lens_version": 1,
}
TRANSCRIPT = [
    {"sentence_id": "f1", "sequence_order": 0, "text": "Let's go with X.",
     "speaker_id": "sp1", "speaker": "Alice Johnson", "utterance_id": "u-abc"},
    {"sentence_id": "f2", "sequence_order": 1, "text": "I'll draft the doc.",
     "speaker_id": "sp1", "speaker": "Alice Johnson", "utterance_id": "u-abc"},
]
SPEAKERS = [{"speaker_id": "sp1", "handle": "Alice", "display_name": "Alice Johnson", "provisional": False}]
ITEMS = [{
    "item_id": "8888888812345678", "node_type": "Decision", "lens_version": 1,
    "confidence": 0.9, "model": "haiku", "provider": "anthropic", "locked": False,
    "props": {"item_id": "8888888812345678", "lens": "meeting_minutes", "node_type": "Decision",
              "text": "Go with X", "made_by": "Alice", "confidence": 0.9},
    "speaker_links": [{"relationship": "DECIDED_BY", "speaker_id": "sp1", "display_name": "Alice Johnson"}],
    "supporting_fragment_ids": ["f1"],
}]
CLAIMS = [{"claim_id": "c1", "text": "We will ship", "kind": "commitment", "confidence": 0.8,
           "model": "haiku", "provider": "anthropic", "speaker_id": "sp1",
           "speaker": "Alice Johnson", "supporting_fragment_ids": ["f2"]}]
ENTITIES = [{"surface": "ecu", "entity_type": "product",
             "mentions": [{"sentence_id": "f1", "start": 0, "end": 3, "text": "ECU", "confidence": 0.9}]}]
ANALYSIS = [{"sequence_order": 0, "text": "Let's go with X.", "speaker": "Alice Johnson",
             "function": "declarative", "structure": "simple", "purpose": "Statement",
             "topics": ["vendors"], "keywords": ["telemetry"], "confidence": 0.9, "flags": None}]


def render():
    lens = load_lens("meeting_minutes")
    files = dict(render_bundle(HEADER, TRANSCRIPT, SPEAKERS, ITEMS, CLAIMS,
                               ENTITIES, ANALYSIS, lens, exported_at="2026-07-10T12:00:00+00:00"))
    return files


def test_naming_helpers():
    assert slugify("Alice Johnson") == "alice-johnson"
    assert item_dir("ActionItem") == "action-items"
    assert item_filename("ActionItem", "8888888812345678") == "action-item-88888888.md"


def test_every_nonreserved_file_is_okf_conformant():
    files = render()
    assert "index.md" in files and "interview.md" in files
    for path, content in files.items():
        if path == "index.md":
            assert not content.startswith("---"), "index.md is reserved: no frontmatter"
            continue
        assert content.startswith("---\n"), f"{path} missing frontmatter"
        fm = yaml.safe_load(content.split("---\n")[1])
        assert fm.get("type"), f"{path} missing required type"


def test_lens_item_file_content():
    files = render()
    content = files["decisions/decision-88888888.md"]
    fm = yaml.safe_load(content.split("---\n")[1])
    assert fm["type"] == "Decision" and fm["lens"] == "meeting_minutes"
    assert fm["made_by"] == "Alice"                      # extracted field passes through
    assert "item_id: 8888888812345678" in content        # id preserved (as `id`)
    assert "DECIDED_BY: [Alice Johnson](/speakers/alice-johnson.md)" in content
    assert "> Let's go with X." in content               # verbatim grounding quote
    assert "(/transcript.md#u-1)" in content             # bundle-absolute anchor link


def test_transcript_anchors_and_index_links():
    files = render()
    assert '<a id="u-1"></a>' in files["transcript.md"]
    assert "](decisions/decision-88888888.md)" in files["index.md"]  # relative link
    assert "claims/claim-c1.md" in files  # claim file named by id prefix rule
    assert "entities/ecu.md" in files
```

- [ ] **Step 2: Run to verify fail.**

- [ ] **Step 3: Implement** `src/export/renderer.py`. Core skeleton (write all renderers; each is a short function returning one `(path, content)`):

```python
"""Pure OKF rendering: reader dicts in, (relative_path, content) pairs out.

No I/O, no Neo4j. Lens items render generically from node_type + properties +
the lens YAML's projects_to — zero per-lens code.
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple

import yaml

from src.lens.models import LensSpec

# Public: the queries router (Task 7) uses this same split.
RESERVED_PROPS = {
    "item_id", "lens", "lens_version", "node_type", "confidence", "model",
    "provider", "interview_id", "locked", "overridden_at", "override_note",
}


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _kebab(node_type: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "-", node_type).lower()


def item_dir(node_type: str) -> str:
    return _kebab(node_type) + "s"


def item_filename(node_type: str, item_id: str) -> str:
    return f"{_kebab(node_type)}-{item_id[:8]}.md"


def _frontmatter(fields: Dict[str, Any]) -> str:
    clean = {k: v for k, v in fields.items() if v is not None}
    return "---\n" + yaml.safe_dump(clean, sort_keys=False, allow_unicode=True) + "---\n"


def _anchors(transcript: List[Dict[str, Any]]) -> Dict[str, str]:
    """sentence_id -> anchor. Utterances u-<n> by first appearance; loose fragments f-<seq>."""
    anchors: Dict[str, str] = {}
    utterance_anchor: Dict[str, str] = {}
    n = 0
    for row in transcript:
        uid = row.get("utterance_id")
        if uid:
            if uid not in utterance_anchor:
                n += 1
                utterance_anchor[uid] = f"u-{n}"
            anchors[row["sentence_id"]] = utterance_anchor[uid]
        else:
            anchors[row["sentence_id"]] = f"f-{row['sequence_order']}"
    return anchors
```

`render_bundle` assembles: `interview.md` (type `Interview`; participants, counts, `metadata.front_matter` round-tripped under a `front_matter:` key), `transcript.md` (type `Transcript`; consecutive rows grouped by utterance under `<a id="..."></a>` + `**Speaker:**` blocks), `analysis.md` (type `AnalysisSummary`; markdown table, aggregated topic/keyword tallies first), `speakers/<slug>.md` (type `Speaker`; links to items whose `speaker_links` reference them), one file per lens item (frontmatter: `type` = node_type, `title` = first 80 chars of `props["text"]` (fallback: item_id), `description` = props text, `id`, `lens`, `lens_version`, `confidence`, `model`, `provider`, `locked`, `tags: [lens:<lens>]`, `timestamp` = exported_at, plus every non-reserved prop; body = text + `## Relationships` lines `<REL>: [<display>](/speakers/<slug>.md)` + `## Grounding` blockquotes with `— [<speaker>](/speakers/<slug>.md), [<anchor>](/transcript.md#<anchor>)`), `claims/claim-<id8>.md` (same anatomy, type `Claim`, `kind` in frontmatter), `entities/<slug>.md` (type `Entity`; mentions table with transcript links), and `index.md` LAST (no frontmatter; one section per directory, relative links, one-line descriptions). Grounding quote text comes from a `sentence_id -> text` map built from transcript rows; a grounding id missing from the map renders the link without a quote (broken links are tolerated by OKF consumers, never fatal here).

- [ ] **Step 4: Run** — `pytest tests/export -q --no-cov` → green.
- [ ] **Step 5: Commit** — `git commit -am "feat: pure OKF renderer (conformant frontmatter, anchors, links-as-edges)"`

---

### Task 5: Bundler with consistency guard + CLI

**Files:**
- Create: `src/export/bundler.py`, `src/export/__main__.py`
- Test: `tests/export/test_bundler.py`

**Interfaces:**
- Consumes: `reader.*` (Task 3), `render_bundle` (Task 4), `load_lens`, `get_interview_repository`, `Neo4jConnectionManager.get_session()`.
- Produces:
  - `ExportResult(BaseModel)`: `interview_id, lens, lens_version, bundle_path: str, files_written: int, items: int, claims: int, entities: int`.
  - `OkfExporter(config_dict=None)` with `async export(interview_id: str, lens_name: str, out_dir: str = "exports", zip_bundle: bool = False) -> ExportResult`.
  - Raises: `ValueError(f"Interview {id} not found")`; `ValueError("Unknown lens: ...")` (via `load_lens`); `RuntimeError("projection lag: ...")` when the expected item-id set ≠ projected set.
- CLI: `python -m src.export <interview_id> <lens_name> [--out exports] [--zip]` printing `ExportResult` JSON; exit 1 on any raise.

- [ ] **Step 1: Write the failing tests** (`tests/export/test_bundler.py`):

```python
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.events.aggregates import Interview
from src.export.bundler import OkfExporter

IID = "22222222-2222-2222-2222-222222222222"
SP1 = "33333333-3333-3333-3333-333333333333"
ITEM = "88888888-8888-8888-8888-888888888801"


def make_interview(with_item=True):
    i = Interview(IID)
    i.create(title="t", source="s", metadata={"fragment_count": 1})
    i.add_speaker(SP1, "S1", "Alice", False, 1.0, "parsed")
    i.apply_lens("meeting_minutes", 1)
    if with_item:
        i.record_lens_extraction(
            lens="meeting_minutes", lens_version=1, node_type="Decision", item_id=ITEM,
            fields={"text": "Go with X"}, supporting_fragment_ids=[], speaker_links=[],
            confidence=0.9, model="haiku", provider="anthropic",
        )
    i.mark_events_as_committed()
    return i


def patch_world(interview, projected_items):
    repo = MagicMock(load=AsyncMock(return_value=interview))
    session = AsyncMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    reader_rows = {
        "transcript_rows": [], "speaker_rows": [], "claim_rows": [],
        "entity_rows": [], "analysis_rows": [],
    }
    patches = [
        patch("src.export.bundler.get_interview_repository", return_value=repo),
        patch("src.export.bundler.Neo4jConnectionManager.get_session",
              new=AsyncMock(return_value=session)),
        patch("src.export.bundler.reader.lens_item_rows",
              new=AsyncMock(return_value=projected_items)),
    ]
    for name, rows in reader_rows.items():
        patches.append(patch(f"src.export.bundler.reader.{name}", new=AsyncMock(return_value=rows)))
    return patches


PROJECTED = [{
    "item_id": ITEM, "node_type": "Decision", "lens_version": 1, "confidence": 0.9,
    "model": "haiku", "provider": "anthropic", "locked": False,
    "props": {"item_id": ITEM, "text": "Go with X"},
    "speaker_links": [], "supporting_fragment_ids": [],
}]


@pytest.mark.asyncio
async def test_export_writes_conformant_bundle(tmp_path):
    import yaml
    patches = patch_world(make_interview(), PROJECTED)
    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7]:
        result = await OkfExporter().export(IID, "meeting_minutes", out_dir=str(tmp_path))
    bundle = tmp_path / f"{IID}-meeting_minutes"
    assert (bundle / "index.md").exists() and (bundle / "interview.md").exists()
    assert (bundle / "log.md").exists()
    assert result.items == 1 and result.files_written >= 4
    content = (bundle / "decisions" / f"decision-{ITEM[:8]}.md").read_text()
    assert yaml.safe_load(content.split("---\n")[1])["type"] == "Decision"


@pytest.mark.asyncio
async def test_projection_lag_raises(tmp_path):
    patches = patch_world(make_interview(with_item=True), projected_items=[])  # graph empty
    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7]:
        with pytest.raises(RuntimeError, match="projection lag"):
            await OkfExporter().export(IID, "meeting_minutes", out_dir=str(tmp_path))


@pytest.mark.asyncio
async def test_unknown_interview_raises(tmp_path):
    repo = MagicMock(load=AsyncMock(return_value=None))
    with patch("src.export.bundler.get_interview_repository", return_value=repo):
        with pytest.raises(ValueError, match="not found"):
            await OkfExporter().export(IID, "meeting_minutes", out_dir=str(tmp_path))


@pytest.mark.asyncio
async def test_reexport_prepends_log_entry(tmp_path):
    patches = patch_world(make_interview(), PROJECTED)
    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7]:
        await OkfExporter().export(IID, "meeting_minutes", out_dir=str(tmp_path))
        await OkfExporter().export(IID, "meeting_minutes", out_dir=str(tmp_path))
    log = (tmp_path / f"{IID}-meeting_minutes" / "log.md").read_text()
    assert log.count("exported") == 2
```

- [ ] **Step 2: Run to verify fail.**

- [ ] **Step 3: Implement** `src/export/bundler.py`:

```python
"""Bundle orchestration: guard -> read -> render (fully in memory) -> write."""

import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel

from src.events.repository import get_interview_repository
from src.export import reader
from src.export.renderer import render_bundle
from src.lens.models import load_lens
from src.utils.logger import get_logger
from src.utils.neo4j_driver import Neo4jConnectionManager

logger = get_logger()


class ExportResult(BaseModel):
    interview_id: str
    lens: str
    lens_version: int
    bundle_path: str
    files_written: int
    items: int
    claims: int
    entities: int


class OkfExporter:
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        from src.config import config as global_config

        self.config = config_dict if config_dict is not None else global_config

    async def export(
        self, interview_id: str, lens_name: str,
        out_dir: str = "exports", zip_bundle: bool = False,
    ) -> ExportResult:
        lens = load_lens(lens_name)
        interview = await get_interview_repository().load(interview_id)
        if interview is None:
            raise ValueError(f"Interview {interview_id} not found")

        async with await Neo4jConnectionManager.get_session() as session:
            items = await reader.lens_item_rows(session, interview_id, lens.name)
            self._guard(interview, lens.name, items)
            transcript = await reader.transcript_rows(session, interview_id)
            speakers = await reader.speaker_rows(session, interview_id)
            claims = await reader.claim_rows(session, interview_id)
            entities = await reader.entity_rows(session, interview_id)
            analysis = await reader.analysis_rows(session, interview_id)

        header = self._header(interview, lens)
        exported_at = datetime.now(timezone.utc).isoformat()
        files = render_bundle(header, transcript, speakers, items, claims,
                              entities, analysis, lens, exported_at)

        bundle_dir = Path(out_dir) / f"{interview_id}-{lens.name}"
        log_content = self._log_entry(bundle_dir, lens, len(items), exported_at)
        if bundle_dir.exists():
            shutil.rmtree(bundle_dir)
        for rel_path, content in files:
            target = bundle_dir / rel_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        (bundle_dir / "log.md").write_text(log_content, encoding="utf-8")

        bundle_path = str(bundle_dir)
        if zip_bundle:
            zip_path = bundle_dir.with_suffix(".zip")
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for f in sorted(bundle_dir.rglob("*")):
                    if f.is_file():
                        zf.write(f, f.relative_to(bundle_dir.parent))
            bundle_path = str(zip_path)

        return ExportResult(
            interview_id=interview_id, lens=lens.name, lens_version=lens.version,
            bundle_path=bundle_path, files_written=len(files) + 1,
            items=len(items), claims=len(claims), entities=len(entities),
        )

    def _guard(self, interview, lens_name: str, projected_rows) -> None:
        """Expected = current-version items + locked items of any version."""
        current = interview.lens_runs.get(lens_name)
        expected = {
            iid for iid, v in interview.lens_items.items()
            if v["lens"] == lens_name and (v["lens_version"] == current or v["locked"])
        }
        projected = {r["item_id"] for r in projected_rows}
        if expected != projected:
            raise RuntimeError(
                f"projection lag: aggregate expects {len(expected)} items for lens "
                f"{lens_name!r}, graph has {len(projected)}; retry shortly"
            )
```

`_header` builds the header dict from aggregate state (`interview.title/source/started_at/metadata`, participants from `interview.speakers` excluding merged, `fragment_count` from metadata, `utterance_count` = non-removed utterances, `project_id` from `interview.metadata.get("front_matter", {}).get("project")` or None). `_log_entry` reads the existing `log.md` if present and prepends `## <YYYY-MM-DD>\n\n- <exported_at>: exported <n> items from <lens> v<version>\n` (grouped newest-first per OKF).

`src/export/__main__.py` mirrors `src/lens/__main__.py`:

```python
"""Export an interview x lens as an OKF bundle.

Usage: python -m src.export <interview_id> <lens_name> [--out exports] [--zip]
"""

import argparse
import asyncio
import sys

from src.export.bundler import OkfExporter


def main() -> None:
    parser = argparse.ArgumentParser(description="Export an OKF bundle (Layer 5)")
    parser.add_argument("interview_id")
    parser.add_argument("lens_name")
    parser.add_argument("--out", default="exports", help="Output directory")
    parser.add_argument("--zip", action="store_true", help="Also produce a .zip archive")
    args = parser.parse_args()

    try:
        result = asyncio.run(
            OkfExporter().export(args.interview_id, args.lens_name,
                                 out_dir=args.out, zip_bundle=args.zip)
        )
    except (ValueError, RuntimeError) as exc:
        print(f"export failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run** — `pytest tests/export -q --no-cov` → green.
- [ ] **Step 5: Commit** — `git commit -am "feat: OkfExporter with projection-lag guard; python -m src.export CLI"`

---

### Task 6: Exports API router (zip download)

**Files:**
- Create: `src/api/routers/exports.py`
- Modify: `src/main.py` (import + `app.include_router(exports_router.router)`)
- Test: `tests/api/test_exports_router.py`

**Interfaces:**
- Consumes: `OkfExporter` (Task 5).
- Produces: `GET /exports/{interview_id}/{lens_name}` → 200 `application/zip` with `Content-Disposition: attachment; filename=<interview_id>-<lens>.zip`; 404 (`ValueError` containing "not found"), 422 (`ValueError` containing "Unknown lens"), 409 (`RuntimeError` projection lag).

- [ ] **Step 1: Write the failing tests** (`tests/api/test_exports_router.py`):

```python
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from src.main import app

IID = "22222222-2222-2222-2222-222222222222"


@pytest.fixture
def client():
    return TestClient(app)


def fake_export(tmp_path):
    async def _export(self, interview_id, lens_name, out_dir="exports", zip_bundle=False):
        from src.export.bundler import ExportResult
        zip_path = tmp_path / f"{interview_id}-{lens_name}.zip"
        zip_path.write_bytes(b"PK\x05\x06" + b"\x00" * 18)  # minimal empty zip
        return ExportResult(interview_id=interview_id, lens=lens_name, lens_version=1,
                            bundle_path=str(zip_path), files_written=1,
                            items=0, claims=0, entities=0)
    return _export


def test_export_returns_zip(client, tmp_path):
    with patch("src.api.routers.exports.OkfExporter.export", new=fake_export(tmp_path)):
        resp = client.get(f"/exports/{IID}/meeting_minutes")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/zip"
    assert "attachment" in resp.headers["content-disposition"]


@pytest.mark.parametrize("exc,status", [
    (ValueError(f"Interview {IID} not found"), 404),
    (ValueError("Unknown lens: nope"), 422),
    (RuntimeError("projection lag: retry shortly"), 409),
])
def test_export_error_mapping(client, exc, status):
    with patch("src.api.routers.exports.OkfExporter.export", new=AsyncMock(side_effect=exc)):
        resp = client.get(f"/exports/{IID}/meeting_minutes")
    assert resp.status_code == status
```

- [ ] **Step 2: Run to verify fail.**

- [ ] **Step 3: Implement** `src/api/routers/exports.py`:

```python
"""OKF bundle download (Layer 5 egress)."""

import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from src.export.bundler import OkfExporter
from src.utils.logger import get_logger

router = APIRouter(tags=["exports"])
logger = get_logger()


@router.get("/exports/{interview_id}/{lens_name}")
async def download_bundle(interview_id: str, lens_name: str):
    """Export the interview x lens as an OKF bundle and return it zipped."""
    with tempfile.TemporaryDirectory() as tmp:
        try:
            result = await OkfExporter().export(
                interview_id, lens_name, out_dir=tmp, zip_bundle=True
            )
        except ValueError as e:
            status = 404 if "not found" in str(e) else 422
            raise HTTPException(status_code=status, detail=str(e))
        except RuntimeError as e:
            raise HTTPException(status_code=409, detail=str(e))
        payload = Path(result.bundle_path).read_bytes()
    filename = f"{interview_id}-{lens_name}.zip"
    return Response(
        content=payload,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )
```

`src/main.py`: `from src.api.routers import exports as exports_router` + `app.include_router(exports_router.router)` next to the existing includes.

- [ ] **Step 4: Run** — `pytest tests/api -q --no-cov` → green.
- [ ] **Step 5: Commit** — `git commit -am "feat: OKF bundle zip download endpoint"`

---

### Task 7: Richer query endpoints

**Files:**
- Modify: `src/export/reader.py` (add `worklist_rows`, `speaker_rollup_rows`)
- Create: `src/api/routers/queries.py`
- Modify: `src/main.py` (include router)
- Test: `tests/api/test_queries_router.py`, extend `tests/export/test_reader.py`

**Interfaces:**
- Consumes: `reader.lens_item_rows` (Task 3), `Neo4jConnectionManager.get_session()`.
- Produces (reader):
  - `worklist_rows(session, project_id=None, threshold=0.7, limit=50, offset=0) -> Dict[str, List[dict]]` — `{"lens_items": [{interview_id, item_id, node_type, lens, confidence, reason}], "claims": [{interview_id, claim_id, text, kind, confidence, reason}]}`; `reason` ∈ `"low_confidence"` / `"unresolved_reference"` (lens item matches when `confidence < threshold` OR `any(k IN keys(n) WHERE k ENDS WITH '_unresolved')`); each list independently ordered by confidence ASC and paginated.
  - `speaker_rollup_rows(session, project_id=None, name=None, limit=50, offset=0) -> List[dict]` — one row per (non-merged) speaker display_name: `{display_name, items: [{node_type, relationship, text, interview_id, item_id}], claims: [{text, kind, interview_id, claim_id}]}`; cross-interview identity = display-name string match (grouped in Python from per-speaker rows); ordered by display_name.
- Produces (router `queries.py`):
  - `GET /interviews/{interview_id}/lenses/{lens}/items?node_type=&min_confidence=&limit=&offset=` → `{"items": [...]}` (each item: reader row minus `props`, plus `fields` = non-reserved props).
  - `GET /review/worklist?project_id=&threshold=0.7&limit=&offset=` → worklist_rows result verbatim.
  - `GET /speakers/rollup?project_id=&name=&limit=&offset=` → `{"speakers": [...]}`.
- `project_id` filtering uses `EXISTS { MATCH (:Project {project_id: $project_id})-[:CONTAINS_INTERVIEW]->(:Interview {interview_id: n.interview_id}) }`.

- [ ] **Step 1: Write the failing tests.** Add to `tests/export/test_reader.py`:

```python
@pytest.mark.asyncio
async def test_worklist_rows_filters_and_reasons():
    session = make_session([])
    result = await reader.worklist_rows(session, threshold=0.6)
    assert set(result) == {"lens_items", "claims"}
    lens_query = session.run.call_args_list[0][0][0]
    assert "_unresolved" in lens_query and "confidence < $threshold" in lens_query


@pytest.mark.asyncio
async def test_rollup_groups_by_display_name():
    session = make_session([])
    rows = await reader.speaker_rollup_rows(session, name="Alice Johnson")
    assert rows == []
    query = session.run.call_args_list[0][0][0]
    assert "display_name" in query and "merged_into IS NULL" in query
```

New `tests/api/test_queries_router.py`:

```python
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.main import app

IID = "22222222-2222-2222-2222-222222222222"


@pytest.fixture
def client():
    return TestClient(app)


def patch_session():
    session = AsyncMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    return patch(
        "src.api.routers.queries.Neo4jConnectionManager.get_session",
        new=AsyncMock(return_value=session),
    )


def test_lens_items_endpoint_splits_fields(client):
    row = {"item_id": "i1", "node_type": "Decision", "lens_version": 1, "confidence": 0.9,
           "model": "haiku", "provider": "anthropic", "locked": False,
           "props": {"item_id": "i1", "lens": "meeting_minutes", "text": "Go with X"},
           "speaker_links": [], "supporting_fragment_ids": []}
    with patch_session(), \
         patch("src.api.routers.queries.reader.lens_item_rows", new=AsyncMock(return_value=[row])):
        resp = client.get(f"/interviews/{IID}/lenses/meeting_minutes/items")
    assert resp.status_code == 200
    item = resp.json()["items"][0]
    assert item["fields"] == {"text": "Go with X"}   # reserved props stripped
    assert "props" not in item


def test_worklist_endpoint(client):
    result = {"lens_items": [], "claims": []}
    with patch_session(), \
         patch("src.api.routers.queries.reader.worklist_rows", new=AsyncMock(return_value=result)):
        resp = client.get("/review/worklist?threshold=0.5")
    assert resp.status_code == 200 and resp.json() == result


def test_rollup_endpoint(client):
    with patch_session(), \
         patch("src.api.routers.queries.reader.speaker_rollup_rows", new=AsyncMock(return_value=[])):
        resp = client.get("/speakers/rollup?name=Alice")
    assert resp.status_code == 200 and resp.json() == {"speakers": []}
```

- [ ] **Step 2: Run to verify fail.**

- [ ] **Step 3: Implement.** Reader additions — worklist lens-item query:

```cypher
    MATCH (n:LensItem)
    WHERE ($project_id IS NULL OR EXISTS {
        MATCH (:Project {project_id: $project_id})-[:CONTAINS_INTERVIEW]->
              (:Interview {interview_id: n.interview_id}) })
      AND (n.confidence < $threshold
           OR any(k IN keys(n) WHERE k ENDS WITH '_unresolved'))
    RETURN n.interview_id AS interview_id, n.item_id AS item_id,
           n.node_type AS node_type, n.lens AS lens, n.confidence AS confidence,
           CASE WHEN n.confidence < $threshold THEN 'low_confidence'
                ELSE 'unresolved_reference' END AS reason
    ORDER BY n.confidence ASC SKIP $offset LIMIT $limit
```

claims query analogous (`MATCH (c:Claim) WHERE ... c.confidence < $threshold`, reason fixed `'low_confidence'`). Rollup runs two queries (items via `MATCH (n:LensItem)-[r]->(sp:Speaker) WHERE sp.merged_into IS NULL AND type(r) <> 'SUPPORTED_BY' ...`, claims via `MADE_BY`) returning flat per-speaker rows; Python groups by `display_name` into the row shape, applies `name` filter case-insensitively and `limit/offset` over the grouped list.

Router `src/api/routers/queries.py` — pattern for all three endpoints:

```python
"""Richer read-model queries (Layer 5)."""

from typing import Optional

from fastapi import APIRouter, Query

from src.export import reader
from src.export.renderer import RESERVED_PROPS
from src.utils.neo4j_driver import Neo4jConnectionManager

router = APIRouter(tags=["queries"])


@router.get("/interviews/{interview_id}/lenses/{lens}/items")
async def lens_items(
    interview_id: str,
    lens: str,
    node_type: Optional[str] = None,
    min_confidence: Optional[float] = Query(None, ge=0.0, le=1.0),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    async with await Neo4jConnectionManager.get_session() as session:
        rows = await reader.lens_item_rows(
            session, interview_id, lens, node_type=node_type,
            min_confidence=min_confidence, limit=limit, offset=offset,
        )
    items = []
    for row in rows:
        props = row.pop("props", {})
        row["fields"] = {k: v for k, v in props.items() if k not in RESERVED_PROPS}
        items.append(row)
    return {"items": items}
```

(worklist and rollup endpoints follow the same session pattern; thresholds validated `ge=0.0, le=1.0`). Register the router in `src/main.py`.

- [ ] **Step 4: Run** — `pytest tests/api tests/export -q --no-cov` → green.
- [ ] **Step 5: Commit** — `git commit -am "feat: lens items / review worklist / speaker rollup query endpoints"`

---

### Task 8: Integration smoke — front matter in, bundle out

**Files:**
- Create: `tests/integration/test_layer5_export_smoke.py`

**Interfaces:** none new. Mirrors `tests/integration/test_layer3_lens_smoke.py` (read it first and reuse its canned-outcome/replay structure verbatim where possible).

- [ ] **Step 1: Write the test.** Structure:

```python
"""Layer 5 export smoke (integration).

Front-mattered transcript -> ingest -> canned meeting_minutes lens -> replay
through the real registry into real Neo4j -> OkfExporter -> assert the bundle
is OKF-conformant with links, grounding, and front-matter round-trip.
Requires `make test-infra-up`.
"""
```

Body: copy the Layer 3 smoke's CANNED/EMPTY/canned_outcome pattern and replay loop, with transcript:

```python
LABELED = """---
title: Q3 Vendor Selection
project: telemetry
date: 2026-07-01
participants: [Alice Johnson, Bob Reyes]
---
Alice: We will go with vendor X and I'll draft the doc by Friday.
Bob: Sounds good to me.
"""
```

After ingest + lens + replay, run `await OkfExporter().export(interview_id, "meeting_minutes", out_dir=str(tmp_path / "exports"))`, then assert:

```python
    import yaml as yaml_mod
    bundle = tmp_path / "exports" / f"{interview_id}-meeting_minutes"
    md_files = [p for p in bundle.rglob("*.md")]
    assert len(md_files) >= 6
    for p in md_files:
        content = p.read_text()
        if p.name in ("index.md", "log.md"):
            assert not content.startswith("---")
            continue
        fm = yaml_mod.safe_load(content.split("---\n")[1])
        assert fm.get("type"), f"{p} missing type"
    interview_md = (bundle / "interview.md").read_text()
    assert "Alice Johnson" in interview_md          # front matter round-trip
    decision = next(bundle.glob("decisions/decision-*.md")).read_text()
    assert "DECIDED_BY" in decision and "(/speakers/" in decision
    assert "> " in decision                          # verbatim grounding quote
    # Ingest seeded 'Alice' from participants:
    assert (bundle / "speakers" / "alice-johnson.md").exists()
    # Query readers against the same projected graph:
    from src.export import reader
    from src.utils.neo4j_driver import Neo4jConnectionManager
    async with await Neo4jConnectionManager.get_session() as session:
        worklist = await reader.worklist_rows(session, threshold=1.1)  # catches everything
        rollup = await reader.speaker_rollup_rows(session, name="Alice Johnson")
    assert worklist["lens_items"], "worklist should surface items below threshold 1.1"
    assert rollup and rollup[0]["display_name"] == "Alice Johnson"
```

(The canned decision must use `made_by: "Alice Johnson"` — the seeded display name — so DECIDED_BY resolves; canned action item uses `owner: "SELF"`.) Mark `pytestmark = pytest.mark.integration`.

- [ ] **Step 2: Run against live infra** — `set -a; source .env; set +a; NEO4J_URI=bolt://localhost:7688 NEO4J_USER=neo4j NEO4J_PASSWORD=testpassword ESDB_CONNECTION_STRING="esdb://localhost:2113?tls=false" ~/.pyenv/versions/3.10.7/bin/python -m pytest tests/integration/test_layer5_export_smoke.py -q --no-cov` → PASS. Also re-run the three existing layer smokes (same command with their paths) → all PASS.
- [ ] **Step 3: Commit** — `git commit -am "test: Layer 5 export smoke (front matter in, OKF bundle out)"`

---

### Task 9: Docs + full-suite gate

**Files:**
- Modify: `docs/ROADMAP.md` (M4.4 ✅ milestone checklist mirroring Tasks 1–8; Quick Status; Current Phase → next milestone planning; decision log rows: read-side exporter decision, front-matter round-trip)
- Modify: `docs/architecture/database-schema.md` (note: front matter stored in Interview aggregate metadata — NOT projected; exporter reads header from the aggregate; no graph schema changes in M4.4)
- Modify: `README.md` (What-It-Does: export step; Run-it: `python -m src.export <interview_id> meeting_minutes`; API table: the four new endpoints)

- [ ] **Step 1: Edit all three docs** (follow the M4.3 entries' voice; keep the Quick Status test count honestly matching Step 2's output).
- [ ] **Step 2: Full unit suite** — `pytest tests -m "not integration" -q --no-cov` → green, zero collection errors; `flake8` clean on all files touched in this plan.
- [ ] **Step 3: Commit** — `git commit -am "docs: M4.4 milestone — OKF export, front-matter capture, query endpoints"`

---

## Verification (whole-plan)

1. Full unit suite green; all four integration smokes (Layers 1/2/3/5) green against live infra.
2. Harness e2e (no paid LLM): ingest a front-mattered transcript, apply meeting_minutes via the `claude_code` provider chain, export, open `index.md` — links resolve, items grounded.
3. OKF conformance: the smoke's conformance walk passes on a real bundle.
4. Zero-per-lens-code check: a hypothetical second lens still requires only YAML + prompts (exporter renders it with no code change).

## Deferred / out of scope (M4.4)

- GraphRAG retrieval; corpus-level bundles; incremental/diff exports; importing arbitrary OKF bundles; cross-interview speaker identity resolution (rollup is display-name match, documented); export audit events.
