"""Layer 5 export smoke (integration).

Front-mattered transcript -> ingest -> canned meeting_minutes lens -> replay
through the real registry into real Neo4j -> OkfExporter -> assert the bundle
is OKF-conformant with links, grounding, and front-matter round-trip.
Requires `make test-infra-up`.
"""

import uuid as uuid_mod

import pytest
import yaml as yaml_mod

from src.enrichment.executor import SpecOutcome
from src.events.repository import get_repository_factory
from src.export.bundler import OkfExporter
from src.lens.engine import LensEngine
from src.projections.bootstrap import create_handler_registry
from src.utils.neo4j_driver import Neo4jConnectionManager

pytestmark = pytest.mark.integration

LABELED = """---
title: Q3 Vendor Selection
project: telemetry
date: 2026-07-01
participants: [Alice Johnson, Bob Reyes]
---
Alice: We will go with vendor X and I'll draft the doc by Friday.
Bob: Sounds good to me.
"""

CANNED = {
    "objectives": {"objectives": [{"text": "Choose a vendor", "confidence": 0.9}]},
    "decisions": {
        "decisions": [
            {"text": "Go with vendor X", "made_by": "Alice Johnson", "confidence": 0.9}
        ]
    },
    "action_items": {
        "action_items": [
            {"text": "Draft the doc", "owner": "SELF", "due": "Friday", "confidence": 0.8}
        ]
    },
    "followups": {"followups": []},
}

EMPTY = {
    "objectives": {"objectives": []},
    "decisions": {"decisions": []},
    "action_items": {"action_items": []},
    "followups": {"followups": []},
}


def canned_outcome(spec, text):
    """Only Alice's utterance (and the document) yields items."""
    source = CANNED if "vendor X" in text else EMPTY
    return SpecOutcome(data=source[spec.name], provider="anthropic", model="haiku")


@pytest.mark.asyncio
async def test_front_matter_ingest_through_okf_bundle(tmp_path, monkeypatch):
    from unittest.mock import AsyncMock, MagicMock

    from src.ingestion.orchestrator import IngestionOrchestrator

    input_file = tmp_path / "smoke_export.txt"
    input_file.write_text(LABELED)

    project_id = f"smoke-{uuid_mod.uuid4()}"
    ingest = IngestionOrchestrator(project_id=project_id, map_dir=tmp_path / "maps")
    ingest_result = await ingest.ingest_file(input_file)
    interview_id = ingest_result.interview_id

    # Mock the executor: canned outcome per lens extractor, no live LLM.
    executor = MagicMock()
    executor.run_spec_on_text = AsyncMock(
        side_effect=lambda spec, text, ctx=None: canned_outcome(spec, text)
    )
    monkeypatch.setattr(LensEngine, "_build_executor", lambda self, lens: executor)

    result = await LensEngine().apply(interview_id, "meeting_minutes")
    assert result.items_extracted == 3  # objective + decision + action item

    # Replay all events in commit order through the real registry.
    factory = get_repository_factory()
    interview_repo = factory.create_interview_repository()
    fragment_repo = factory.create_fragment_repository()
    registry = create_handler_registry()
    events = list(await interview_repo.event_store.read_stream(f"Interview-{interview_id}"))
    for index in range(ingest_result.fragment_count):
        sid = str(uuid_mod.uuid5(uuid_mod.NAMESPACE_DNS, f"{interview_id}:{index}"))
        events.extend(await fragment_repo.event_store.read_stream(f"Sentence-{sid}"))
    events.sort(key=lambda e: e.occurred_at)
    for event in events:
        handler = registry.get_handler(event.event_type)
        if handler:
            await handler.handle(event)

    await OkfExporter().export(interview_id, "meeting_minutes", out_dir=str(tmp_path / "exports"))

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
    assert "Alice Johnson" in interview_md  # front matter round-trip

    decision = next(bundle.glob("decisions/decision-*.md")).read_text()
    assert "DECIDED_BY" in decision and "(/speakers/" in decision
    assert "> " in decision  # verbatim grounding quote

    # Ingest seeded 'Alice' from participants:
    assert (bundle / "speakers" / "alice-johnson.md").exists()

    # Query readers against the same projected graph:
    from src.export import reader

    async with await Neo4jConnectionManager.get_session() as session:
        worklist = await reader.worklist_rows(session, threshold=1.1)  # catches everything
        rollup = await reader.speaker_rollup_rows(session, name="Alice Johnson")
    assert worklist["lens_items"], "worklist should surface items below threshold 1.1"
    assert rollup and rollup[0]["display_name"] == "Alice Johnson"
