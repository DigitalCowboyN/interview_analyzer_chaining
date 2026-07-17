"""Ask-the-corpus smoke test (integration, M4.6).

Ingests two small transcripts under one project (Layer 4 smoke's pattern),
runs canned enrichment + resolution with a fake embedder (the e2e smoke's
canned pattern), replays everything through the real handler registry
against real Neo4j, then drives ``AskEngine`` with the same fake embedder
and a mocked synthesis agent. This smoke is about retrieval (hybrid
vector/fulltext/graph fusion + verbatim citation attachment); the e2e smoke
owns the full ingest→enrich→lens→resolve→export chain. Requires
`make test-infra-up`.
"""

import uuid as uuid_mod
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.ask.engine import AskEngine
from src.enrichment.executor import FragmentEnrichment
from src.enrichment.orchestrator import EnrichmentOrchestrator
from src.events.project_events import project_aggregate_id
from src.events.repository import get_repository_factory
from src.events.store import StreamNotFoundError
from src.projections.bootstrap import create_handler_registry
from src.resolution.engine import ResolutionEngine

pytestmark = pytest.mark.integration

TRANSCRIPT_1 = """---
participants: [Jane Doe]
---
Jane: We will go with Acme Corp for the rollout.
Bob: Sounds good to me.
"""

TRANSCRIPT_2 = """---
participants: [Jane Doe]
---
Jane: Let's follow up with the Acme Corp folks again next week.
Alice: Sounds good to me.
"""

QUESTION = "What did they decide about Acme Corp?"

# Fixed-size, deterministic-by-content embedding space: every text that
# mentions "Acme Corp" gets the SAME distinctive one-hot vector (index 0) so
# the vector channel ranks the Acme fragment first; every other distinct text
# gets its own orthogonal unit vector (never colliding with the Acme slot or
# each other) so the vector channel doesn't accidentally rank noise above it.
_VEC_DIM = 8


class _DistinctiveEmbedder:
    """Deterministic, content-keyed vectors shared by enrichment and ask.

    Same embedder instance/class is used to embed fragments/utterances
    during canned enrichment AND to embed the question during ask — the
    "Acme Corp" fragment and the question (which names Acme Corp) resolve
    to the identical one-hot[0] vector, guaranteeing the vector channel
    surfaces that fragment first via cosine similarity.
    """

    model_name = "smoke-ask-embed"
    dim = _VEC_DIM

    def __init__(self):
        self._assigned: dict = {}
        self._next_slot = 1  # slot 0 reserved for "mentions Acme Corp"

    def _vector_for(self, text: str):
        if "Acme Corp" in text:
            key = "__acme__"
        else:
            key = text
        if key not in self._assigned:
            if key == "__acme__":
                slot = 0
            else:
                slot = self._next_slot
                self._next_slot += 1
            vec = [0.0] * _VEC_DIM
            vec[slot % _VEC_DIM] = 1.0
            self._assigned[key] = vec
        return self._assigned[key]

    async def embed(self, texts):
        return [self._vector_for(t) for t in texts]


async def _replay_all(project_id, interview_ids, registry):
    """Replay Interview-, Sentence-, and Project-stream events in commit
    order through the real handler registry (mirrors tests/integration/
    test_end_to_end_smoke.py's convention — no live subscription consumer
    in this environment)."""
    factory = get_repository_factory()
    interview_repo = factory.create_interview_repository()
    fragment_repo = factory.create_fragment_repository()
    project_repo = factory.create_project_repository()

    events = []
    for interview_id, fragment_count in interview_ids:
        events.extend(await interview_repo.event_store.read_stream(f"Interview-{interview_id}"))
        for index in range(fragment_count):
            sid = str(uuid_mod.uuid5(uuid_mod.NAMESPACE_DNS, f"{interview_id}:{index}"))
            events.extend(await fragment_repo.event_store.read_stream(f"Sentence-{sid}"))

    project_stream = f"Project-{project_aggregate_id(project_id)}"
    try:
        events.extend(await project_repo.event_store.read_stream(project_stream))
    except StreamNotFoundError:
        pass  # ResolutionEngine hasn't run yet on this replay pass

    events.sort(key=lambda e: e.occurred_at)
    for event in events:
        handler = registry.get_handler(event.event_type)
        if handler:
            await handler.handle(event)


def _canned_executor(fragment_count, acme_index):
    """MagicMock executor: only `acme_index` yields an "Acme Corp" entity;
    no document-scope specs (no segments/lens needed for this smoke)."""
    executor = MagicMock()
    executor.enrich_fragments = AsyncMock(return_value=[
        FragmentEnrichment(
            index=i,
            classification={"purpose": "Statement"},
            dimension_confidences={"purpose": 0.9},
            entities=(
                [{"text": "Acme Corp", "entity_type": "organization",
                  "start": 16, "end": 25, "confidence": 0.9}]
                if i == acme_index else []
            ),
            provider="anthropic", model="haiku",
        )
        for i in range(fragment_count)
    ])
    executor.enrich_utterances = AsyncMock(return_value=[])
    executor.document_specs = []  # no document-scope (segment) pass
    return executor


async def _ingest_and_enrich(tmp_path, monkeypatch, project_id, filename, text, acme_index):
    from src.ingestion.orchestrator import IngestionOrchestrator

    input_file = tmp_path / filename
    input_file.write_text(text)
    ingest = IngestionOrchestrator(project_id=project_id, map_dir=tmp_path / "maps")
    ingest_result = await ingest.ingest_file(input_file)

    executor = _canned_executor(ingest_result.fragment_count, acme_index)
    embedder = _DistinctiveEmbedder()
    monkeypatch.setattr(EnrichmentOrchestrator, "_build_executor", lambda self: executor)
    monkeypatch.setattr("src.enrichment.orchestrator.get_embedder", lambda cfg=None: embedder)

    enrich_result = await EnrichmentOrchestrator().enrich_interview(ingest_result.interview_id)
    assert enrich_result.entities_extracted == 1
    return ingest_result


@pytest.mark.asyncio
async def test_ask_engine_hybrid_retrieval_and_verbatim_citations(tmp_path, monkeypatch):
    project_id = f"smoke-ask-{uuid_mod.uuid4()}"

    # --- 1/2. Ingest two transcripts, canned enrichment (fake embedder) -----
    result1 = await _ingest_and_enrich(
        tmp_path, monkeypatch, project_id, "smoke_ask_1.txt", TRANSCRIPT_1, acme_index=0
    )
    result2 = await _ingest_and_enrich(
        tmp_path, monkeypatch, project_id, "smoke_ask_2.txt", TRANSCRIPT_2, acme_index=0
    )

    acme_fragment_id = str(uuid_mod.uuid5(uuid_mod.NAMESPACE_DNS, f"{result1.interview_id}:0"))

    # --- Replay, resolve (canonicalizes the overlapping Acme Corp surface) --
    registry = create_handler_registry()
    interview_ids = [
        (result1.interview_id, result1.fragment_count),
        (result2.interview_id, result2.fragment_count),
    ]
    await _replay_all(project_id, interview_ids, registry)

    monkeypatch.setattr(ResolutionEngine, "_build_embedder", lambda self: _DistinctiveEmbedder())
    resolution_result = await ResolutionEngine().apply(project_id)
    assert resolution_result.entities_canonicalized == 1
    assert resolution_result.persons_identified == 1

    await _replay_all(project_id, interview_ids, registry)

    # --- 3/4. AskEngine: same fake embedder, mocked synthesis agent ---------
    ask_embedder = _DistinctiveEmbedder()
    captured_prompt = {}

    def make_agent(citation_fragment_id):
        agent = MagicMock()

        async def _call(prompt, schema=None):
            captured_prompt["text"] = prompt
            return MagicMock(
                data={
                    "answer": "They chose Acme Corp.",
                    "citations": [{"fragment_id": citation_fragment_id}],
                },
                provider="anthropic",
                model="haiku",
            )

        agent.call = AsyncMock(side_effect=_call)
        return agent

    engine = AskEngine(config_dict={})
    monkeypatch.setattr(AskEngine, "_build_embedder", lambda self: ask_embedder)
    monkeypatch.setattr(AskEngine, "_build_agent", lambda self: make_agent(acme_fragment_id))

    result = await engine.ask(project_id, QUESTION)

    assert result.citations == [
        {
            "fragment_id": acme_fragment_id,
            "interview_id": result1.interview_id,
            "quote": "We will go with Acme Corp for the rollout.",
        }
    ]
    assert result.retrieval["channels"]["graph"] >= 1
    assert result.retrieval["channels"]["vector"] >= 1
    assert "vector_unavailable" not in result.retrieval["flags"]
    assert f"[{acme_fragment_id}]" in captured_prompt["text"]
    assert "We will go with Acme Corp for the rollout." in captured_prompt["text"]

    # --- 5. Degradation leg: embedder raises -> answer still produced -------
    class _RaisingEmbedder:
        model_name = "smoke-ask-embed"
        dim = _VEC_DIM

        async def embed(self, texts):
            raise RuntimeError("embedder offline")

    degraded_engine = AskEngine(config_dict={})
    monkeypatch.setattr(AskEngine, "_build_embedder", lambda self: _RaisingEmbedder())
    monkeypatch.setattr(AskEngine, "_build_agent", lambda self: make_agent(acme_fragment_id))

    degraded_result = await degraded_engine.ask(project_id, QUESTION)

    assert degraded_result.answer is not None
    assert degraded_result.retrieval["flags"] == {"vector_unavailable": "RuntimeError"}
    assert degraded_result.retrieval["channels"]["graph"] >= 1
    assert degraded_result.retrieval["channels"]["fulltext"] >= 1

    # --- 6. Unknown project raises ValueError -------------------------------
    unknown_engine = AskEngine(config_dict={})
    monkeypatch.setattr(AskEngine, "_build_embedder", lambda self: ask_embedder)
    with pytest.raises(ValueError):
        await unknown_engine.ask(f"nope-{uuid_mod.uuid4()}", QUESTION)
