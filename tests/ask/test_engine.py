"""AskEngine: hybrid retrieval, degradation, one synthesis call (M4.6)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from src.ask.engine import AskEngine, AskResult, SynthesisUnavailable

PID = "proj-1"

CONTEXT_ROW = {
    "fragment_id": "f1", "text": "We chose Acme.", "sequence_order": 1,
    "interview_id": "i1", "title": "Kickoff", "speaker": "S1 Alice",
    "person": "Alice Smith", "segment_topics": ["Vendor choice"],
    "entities": ["acme"], "siblings": [],
}

NAME_ROWS = [
    {"kind": "entity", "id": "c1", "name": "Acme Corp", "surfaces": ["Acme", "acme corp"]},
    {"kind": "person", "id": "p1", "name": "Alice Smith", "surfaces": []},
]


def make_session():
    session = AsyncMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    return session


class _FakeEmbedder:
    model_name = "text-embedding-3-small"

    async def embed(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]


class _RaisingEmbedder:
    model_name = "text-embedding-3-small"

    async def embed(self, texts):
        raise RuntimeError("embedder offline")


def make_agent(data=None, side_effect=None):
    agent = MagicMock()
    if side_effect is not None:
        agent.call = AsyncMock(side_effect=side_effect)
    else:
        result = MagicMock(data=data, provider="anthropic", model="claude-x")
        agent.call = AsyncMock(return_value=result)
    return agent


def patch_engine(
    project_exists=True,
    frag_rows=None,
    utt_rows=None,
    ft_rows=None,
    name_rows=None,
    anchor_rows=None,
    ctx_rows=None,
    embedder=None,
):
    session = make_session()
    patches = [
        patch("src.ask.engine.Neo4jConnectionManager.get_session",
              new=AsyncMock(return_value=session)),
        patch("src.ask.engine.reader.project_exists", new=AsyncMock(return_value=project_exists)),
        patch("src.ask.engine.reader.ensure_fulltext_index", new=AsyncMock(return_value=None)),
        patch("src.ask.engine.reader.vector_fragment_rows", new=AsyncMock(return_value=frag_rows or [])),
        patch("src.ask.engine.reader.vector_utterance_rows", new=AsyncMock(return_value=utt_rows or [])),
        patch("src.ask.engine.reader.fulltext_rows", new=AsyncMock(return_value=ft_rows or [])),
        patch("src.ask.engine.reader.name_rows", new=AsyncMock(return_value=name_rows or [])),
        patch("src.ask.engine.reader.graph_anchor_rows", new=AsyncMock(return_value=anchor_rows or [])),
        patch("src.ask.engine.reader.context_rows", new=AsyncMock(return_value=ctx_rows or [])),
        patch.object(AskEngine, "_build_embedder", return_value=embedder or _FakeEmbedder()),
    ]
    return patches, session


def apply_all(patches):
    for p in patches:
        p.start()


def stop_all(patches):
    for p in patches:
        p.stop()


@pytest.mark.asyncio
async def test_ask_happy_path_fuses_channels_and_attaches_verbatim_quotes():
    frag_rows = [{"fragment_id": "f1", "score": 0.9}]
    utt_rows = [{"fragment_id": "f2", "score": 0.5}]
    ft_rows = [{"fragment_id": "f1", "score": 1.0}]
    anchor_rows = [{"fragment_id": "f1"}, {"fragment_id": "f1"}]
    ctx_rows = [CONTEXT_ROW]

    agent = make_agent(data={"answer": "Acme was chosen.", "citations": [{"fragment_id": "f1"}]})
    patches, session = patch_engine(
        frag_rows=frag_rows, utt_rows=utt_rows, ft_rows=ft_rows,
        name_rows=NAME_ROWS, anchor_rows=anchor_rows, ctx_rows=ctx_rows,
    )
    apply_all(patches)
    try:
        with patch.object(AskEngine, "_build_agent", return_value=agent):
            result = await AskEngine(config_dict={}).ask(PID, "Why did we pick Acme?")

        assert result.citations == [
            {"fragment_id": "f1", "interview_id": "i1", "quote": "We chose Acme."}
        ]
        assert set(result.retrieval["channels"].keys()) == {"vector", "fulltext", "graph"}
        assert result.retrieval["channels"]["vector"] == 2  # f1 + f2 deduped by max score
        assert result.retrieval["channels"]["fulltext"] == 1
        assert result.retrieval["channels"]["graph"] == 1
        assert result.provider == "anthropic"
        assert result.model == "claude-x"
        assert result.answer == "Acme was chosen."
        from src.ask import reader
        reader.ensure_fulltext_index.assert_awaited_once()
    finally:
        stop_all(patches)


@pytest.mark.asyncio
async def test_vector_channel_degrades_when_embedder_raises():
    ft_rows = [{"fragment_id": "f1", "score": 1.0}]
    anchor_rows = [{"fragment_id": "f1"}]
    ctx_rows = [CONTEXT_ROW]

    agent = make_agent(data={"answer": "Answer.", "citations": []})
    patches, session = patch_engine(
        ft_rows=ft_rows, name_rows=NAME_ROWS, anchor_rows=anchor_rows,
        ctx_rows=ctx_rows, embedder=_RaisingEmbedder(),
    )
    apply_all(patches)
    try:
        with patch.object(AskEngine, "_build_agent", return_value=agent):
            result = await AskEngine(config_dict={}).ask(PID, "Why Acme?")
    finally:
        stop_all(patches)

    assert result.retrieval["flags"] == {"vector_unavailable": "RuntimeError"}
    assert set(result.retrieval["channels"].keys()) == {"fulltext", "graph"}
    assert result.answer == "Answer."


@pytest.mark.asyncio
async def test_no_hits_skips_llm():
    patches, session = patch_engine()  # everything empty, no name matches
    agent = make_agent(data={"answer": "should not be used", "citations": []})
    apply_all(patches)
    try:
        with patch.object(AskEngine, "_build_agent", return_value=agent):
            result = await AskEngine(config_dict={}).ask(PID, "anything?")
    finally:
        stop_all(patches)

    assert result.answer == "No grounding found in this project for that question."
    assert result.citations == []
    agent.call.assert_not_called()


@pytest.mark.asyncio
async def test_unknown_project_raises_value_error():
    patches, session = patch_engine(project_exists=False)
    apply_all(patches)
    try:
        with pytest.raises(ValueError):
            await AskEngine(config_dict={}).ask("nope", "question?")

        from src.ask import reader
        reader.ensure_fulltext_index.assert_not_awaited()
        reader.vector_fragment_rows.assert_not_awaited()
    finally:
        stop_all(patches)


@pytest.mark.asyncio
async def test_synthesis_failure_raises_with_partial_result():
    ft_rows = [{"fragment_id": "f1", "score": 1.0}]
    ctx_rows = [CONTEXT_ROW]
    agent = make_agent(side_effect=RuntimeError("provider down"))
    patches, session = patch_engine(ft_rows=ft_rows, ctx_rows=ctx_rows)
    apply_all(patches)
    try:
        with patch.object(AskEngine, "_build_agent", return_value=agent):
            with pytest.raises(SynthesisUnavailable) as exc_info:
                await AskEngine(config_dict={}).ask(PID, "why?")
    finally:
        stop_all(patches)

    result = exc_info.value.result
    assert isinstance(result, AskResult)
    assert result.answer is None
    assert result.retrieval["fragments"] == ["f1"]
    assert result.retrieval["channels"]["fulltext"] == 1


@pytest.mark.asyncio
async def test_invalid_synthesis_response_raises_with_partial_result():
    ft_rows = [{"fragment_id": "f1", "score": 1.0}]
    ctx_rows = [CONTEXT_ROW]
    agent = make_agent(data={"citations": []})  # missing "answer"
    patches, session = patch_engine(ft_rows=ft_rows, ctx_rows=ctx_rows)
    apply_all(patches)
    try:
        with patch.object(AskEngine, "_build_agent", return_value=agent):
            with pytest.raises(SynthesisUnavailable) as exc_info:
                await AskEngine(config_dict={}).ask(PID, "why?")
    finally:
        stop_all(patches)

    assert exc_info.value.result.answer is None


def test_match_names_is_case_insensitive_and_checks_surfaces():
    engine = AskEngine(config_dict={})
    names = [
        {"kind": "entity", "id": "c1", "name": "Acme Corp", "surfaces": ["Acme", "acme corp"]},
        {"kind": "person", "id": "p1", "name": "Alice Smith", "surfaces": []},
        {"kind": "entity", "id": "c2", "name": "Zeta Ltd", "surfaces": []},
    ]
    canonical_ids, person_ids = engine._match_names("What did ACME say about the deal?", names)
    assert canonical_ids == ["c1"]
    assert person_ids == []

    canonical_ids, person_ids = engine._match_names("What did alice smith think?", names)
    assert canonical_ids == []
    assert person_ids == ["p1"]

    canonical_ids, person_ids = engine._match_names("no matches here", names)
    assert canonical_ids == []
    assert person_ids == []


@pytest.mark.asyncio
async def test_fabricated_citation_ids_are_dropped():
    ft_rows = [{"fragment_id": "f1", "score": 1.0}]
    ctx_rows = [CONTEXT_ROW]
    agent = make_agent(data={"answer": "Answer.", "citations": [{"fragment_id": "made-up"}]})
    patches, session = patch_engine(ft_rows=ft_rows, ctx_rows=ctx_rows)
    apply_all(patches)
    try:
        with patch.object(AskEngine, "_build_agent", return_value=agent):
            result = await AskEngine(config_dict={}).ask(PID, "why?")
    finally:
        stop_all(patches)

    assert result.citations == []
