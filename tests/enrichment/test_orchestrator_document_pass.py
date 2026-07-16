"""Orchestrator document pass: canned executor, real aggregates, no LLM."""

import uuid as uuid_mod
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.enrichment.executor import SpecOutcome
from src.enrichment.models import ExtractorSpec
from src.enrichment.orchestrator import EnrichmentOrchestrator
from src.events.aggregates import Interview, Sentence
from src.events.interview_events import segment_id_for

IID = "22222222-2222-2222-2222-222222222222"
SP1 = "33333333-3333-3333-3333-333333333333"

TOPIC_SEGMENTS_SPEC = ExtractorSpec(
    name="topic_segments",
    prompt_key="topic_segments",
    response_model="SegmentsResult",
    scope="document",
)


def build_world():
    """Interview with 3 analyzed fragments (all passes upstream of the document pass skip)."""
    interview = Interview(IID)
    interview.create(title="t.txt", source="s", metadata={"fragment_count": 3})
    interview.add_speaker(SP1, "S1", "S1", True, 0.9, "inference")
    f_ids = [str(uuid_mod.uuid5(uuid_mod.NAMESPACE_DNS, f"{IID}:{i}")) for i in range(3)]
    sentences = []
    texts = ["Let's discuss roadmap.", "Roadmap continues.", "Now budget."]
    for i, fid in enumerate(f_ids):
        s = Sentence(fid)
        s.create(interview_id=IID, index=i, text=texts[i])
        s.attribute_speaker(SP1, 0.9, "inference")
        s.generate_analysis(model="haiku", model_version="m4.2", classification={"purpose": "Q"})
        sentences.append(s)
    utterance_id = str(uuid_mod.uuid4())
    interview.identify_utterance(utterance_id, SP1, f_ids, 0.9)
    # Claim already recorded for the utterance so the utterance pass skips.
    claim_id = str(
        uuid_mod.uuid5(uuid_mod.NAMESPACE_DNS, f"{IID}:claim:{utterance_id}:0")
    )
    interview.record_claim(
        claim_id, utterance_id, "existing claim", "assertion", 0.9, "haiku", "anthropic",
    )
    interview.mark_events_as_committed()
    for s in sentences:
        s.mark_events_as_committed()
    return interview, {s.aggregate_id: s for s in sentences}, texts


def make_repos(interview, sentences):
    interview_repo = MagicMock()
    interview_repo.load = AsyncMock(return_value=interview)
    interview_repo.save = AsyncMock(side_effect=lambda a, **k: a.mark_events_as_committed())
    sentence_repo = MagicMock()
    sentence_repo.load = AsyncMock(side_effect=lambda sid: sentences.get(sid))
    sentence_repo.save = AsyncMock(side_effect=lambda a, **k: a.mark_events_as_committed())
    return interview_repo, sentence_repo


def make_executor(document_specs, run_spec_on_text_return):
    executor = MagicMock()
    executor.enrich_fragments = AsyncMock(return_value=[])
    executor.enrich_utterances = AsyncMock(return_value=[])
    executor.document_specs = document_specs
    executor.run_spec_on_text = AsyncMock(return_value=run_spec_on_text_return)
    return executor


def make_embedder():
    embedder = MagicMock(model_name="m", dim=3)
    embedder.embed = AsyncMock(return_value=[])
    return embedder


async def run_with(interview, sentences, executor, embedder, force=False):
    interview_repo, sentence_repo = make_repos(interview, sentences)
    with patch("src.enrichment.orchestrator.get_interview_repository", return_value=interview_repo), \
         patch("src.enrichment.orchestrator.get_sentence_repository", return_value=sentence_repo), \
         patch.object(EnrichmentOrchestrator, "_build_executor", return_value=executor), \
         patch("src.enrichment.orchestrator.get_embedder", return_value=embedder):
        orchestrator = EnrichmentOrchestrator()
        result = await orchestrator.enrich_interview(IID, force=force)
    return result


@pytest.mark.asyncio
async def test_document_pass_records_validated_segments():
    interview, sentences, texts = build_world()
    canned = SpecOutcome(
        data={
            "segments": [
                {"topic": "Roadmap", "start_index": 0, "end_index": 1, "confidence": 0.9},
                {"topic": "Budget", "start_index": 2, "end_index": 2, "confidence": 0.8},
            ]
        },
        provider="anthropic",
        model="haiku",
    )
    executor = make_executor([TOPIC_SEGMENTS_SPEC], canned)
    embedder = make_embedder()

    result = await run_with(interview, sentences, executor, embedder)

    assert result.segments_extracted == 2
    assert result.flags == {}

    sid0 = segment_id_for(IID, 0)
    sid1 = segment_id_for(IID, 1)
    assert interview.segments[sid0] == {
        "topic": "Roadmap", "start_index": 0, "end_index": 1, "removed": False,
    }
    assert interview.segments[sid1] == {
        "topic": "Budget", "start_index": 2, "end_index": 2, "removed": False,
    }

    executor.run_spec_on_text.assert_awaited_once()
    call_args = executor.run_spec_on_text.call_args
    assert call_args.args[0] is TOPIC_SEGMENTS_SPEC
    expected_text = "\n".join(f"[{i}] [S1]: {t}" for i, t in enumerate(texts))
    assert call_args.args[1] == expected_text


@pytest.mark.asyncio
async def test_invalid_proposal_drops_all_and_flags():
    interview, sentences, _ = build_world()
    canned = SpecOutcome(
        data={
            "segments": [
                {"topic": "A", "start_index": 0, "end_index": 2, "confidence": 0.9},
                {"topic": "B", "start_index": 1, "end_index": 2, "confidence": 0.8},
            ]
        },
        provider="anthropic",
        model="haiku",
    )
    executor = make_executor([TOPIC_SEGMENTS_SPEC], canned)
    embedder = make_embedder()

    interview_repo, sentence_repo = make_repos(interview, sentences)
    with patch("src.enrichment.orchestrator.get_interview_repository", return_value=interview_repo), \
         patch("src.enrichment.orchestrator.get_sentence_repository", return_value=sentence_repo), \
         patch.object(EnrichmentOrchestrator, "_build_executor", return_value=executor), \
         patch("src.enrichment.orchestrator.get_embedder", return_value=embedder):
        orchestrator = EnrichmentOrchestrator()
        result = await orchestrator.enrich_interview(IID, force=False)

    assert result.segments_extracted == 0
    assert result.flags == {"topic_segments_invalid": "bad indices or overlapping ranges"}
    assert interview.segments == {}
    # save still called (enrichment continues past a dropped proposal)
    interview_repo.save.assert_awaited_once_with(interview)


@pytest.mark.asyncio
async def test_document_pass_skips_when_live_segments_exist_unless_forced():
    interview, sentences, _ = build_world()
    existing_sid = segment_id_for(IID, 0)
    interview.record_segment(existing_sid, "Existing", 0, 0, 0.9)
    interview.mark_events_as_committed()

    canned = SpecOutcome(
        data={
            "segments": [
                {"topic": "Roadmap", "start_index": 0, "end_index": 0, "confidence": 0.9},
                {"topic": "Budget", "start_index": 1, "end_index": 2, "confidence": 0.8},
            ]
        },
        provider="anthropic",
        model="haiku",
    )
    executor = make_executor([TOPIC_SEGMENTS_SPEC], canned)
    embedder = make_embedder()

    # Not forced: skip entirely since a live segment already exists.
    result = await run_with(interview, sentences, executor, embedder, force=False)
    executor.run_spec_on_text.assert_not_awaited()
    assert result.segments_extracted == 0
    assert result.flags == {}

    # Forced: awaited, existing live segment_id (ordinal 0) skipped idempotently.
    result = await run_with(interview, sentences, executor, embedder, force=True)
    executor.run_spec_on_text.assert_awaited_once()
    assert result.segments_extracted == 1  # only ordinal 1 ("Budget") is new
    assert interview.segments[existing_sid]["topic"] == "Existing"
    new_sid = segment_id_for(IID, 1)
    assert interview.segments[new_sid] == {
        "topic": "Budget", "start_index": 1, "end_index": 2, "removed": False,
    }


@pytest.mark.asyncio
async def test_call_error_degrades_to_flags():
    interview, sentences, _ = build_world()
    canned = SpecOutcome(data=None, flags={"topic_segments_call_error": "X"})
    executor = make_executor([TOPIC_SEGMENTS_SPEC], canned)
    embedder = make_embedder()

    result = await run_with(interview, sentences, executor, embedder)

    assert result.segments_extracted == 0
    assert result.flags == {"topic_segments_call_error": "X"}
    assert interview.segments == {}


@pytest.mark.asyncio
async def test_no_document_spec_is_a_noop():
    interview, sentences, _ = build_world()
    canned = SpecOutcome(data=None, flags={})
    executor = make_executor([], canned)
    embedder = make_embedder()

    result = await run_with(interview, sentences, executor, embedder)

    executor.run_spec_on_text.assert_not_awaited()
    assert result.segments_extracted == 0
    assert result.flags == {}


@pytest.mark.asyncio
async def test_redraw_merge_conflict_drops_all_and_flags():
    """Partial removal + forced re-run proposing a merge that covers a
    surviving live segment's range must drop ALL segments, never record
    an overlap."""
    interview, sentences, _ = build_world()
    sid_a = segment_id_for(IID, 0)
    sid_b = segment_id_for(IID, 1)
    interview.record_segment(sid_a, "A", 0, 1, 0.9)
    interview.record_segment(sid_b, "B", 2, 2, 0.9)
    interview.remove_segment(sid_a)
    interview.mark_events_as_committed()
    segments_before = dict(interview.segments)

    canned = SpecOutcome(
        data={
            "segments": [
                {"topic": "Merged", "start_index": 0, "end_index": 2, "confidence": 0.9},
            ]
        },
        provider="anthropic",
        model="haiku",
    )
    executor = make_executor([TOPIC_SEGMENTS_SPEC], canned)
    embedder = make_embedder()

    result = await run_with(interview, sentences, executor, embedder, force=True)

    assert result.segments_extracted == 0
    assert result.flags == {"topic_segments_conflict": "proposal overlaps surviving live segments"}
    # No new SegmentIdentified events: state unchanged apart from the removal.
    assert interview.segments == segments_before


@pytest.mark.asyncio
async def test_redraw_clean_no_conflict_with_live_survivor():
    """Removed segment's ordinal can be redrawn as long as the new range
    doesn't touch any segment that's still live."""
    interview, sentences, _ = build_world()
    sid_a = segment_id_for(IID, 0)
    sid_b = segment_id_for(IID, 1)
    interview.record_segment(sid_a, "A", 0, 1, 0.9)
    interview.record_segment(sid_b, "B", 2, 2, 0.9)
    interview.remove_segment(sid_b)
    interview.mark_events_as_committed()

    canned = SpecOutcome(
        data={
            "segments": [
                {"topic": "A", "start_index": 0, "end_index": 1, "confidence": 0.9},
                {"topic": "B redrawn", "start_index": 2, "end_index": 2, "confidence": 0.8},
            ]
        },
        provider="anthropic",
        model="haiku",
    )
    executor = make_executor([TOPIC_SEGMENTS_SPEC], canned)
    embedder = make_embedder()

    result = await run_with(interview, sentences, executor, embedder, force=True)

    assert result.segments_extracted == 1
    assert result.flags == {}
    assert interview.segments[sid_a]["topic"] == "A"
    assert interview.segments[sid_b]["topic"] == "B redrawn"
    assert interview.segments[sid_b]["removed"] is False


@pytest.mark.asyncio
async def test_forced_identical_rerun_no_conflict():
    """Both segments still live, proposal identical: candidate set is empty,
    so there's nothing to conflict — existing idempotence, no flag."""
    interview, sentences, _ = build_world()
    sid_a = segment_id_for(IID, 0)
    sid_b = segment_id_for(IID, 1)
    interview.record_segment(sid_a, "A", 0, 1, 0.9)
    interview.record_segment(sid_b, "B", 2, 2, 0.9)
    interview.mark_events_as_committed()

    canned = SpecOutcome(
        data={
            "segments": [
                {"topic": "A", "start_index": 0, "end_index": 1, "confidence": 0.9},
                {"topic": "B", "start_index": 2, "end_index": 2, "confidence": 0.8},
            ]
        },
        provider="anthropic",
        model="haiku",
    )
    executor = make_executor([TOPIC_SEGMENTS_SPEC], canned)
    embedder = make_embedder()

    result = await run_with(interview, sentences, executor, embedder, force=True)

    assert result.segments_extracted == 0
    assert result.flags == {}
