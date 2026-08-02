from tools.code.reader import CodeUnit
from tools.code.render import render_index, render_pipeline

UNITS = [
    CodeUnit("ingestion", "pipeline-layer", ["orchestrator"], ["events", "agents"], ["ESDB", "LLM"], "Ingests.", "p"),
    CodeUnit("events", "infrastructure", [], [], ["ESDB"], "Event store.", "p"),
]


def test_render_index_groups_by_role():
    out = render_index(UNITS)
    assert "## pipeline-layer" in out and "ingestion" in out
    assert "ESDB" in out and "events, agents" in out


def test_render_pipeline_is_mermaid():
    out = render_pipeline(UNITS)
    assert out.strip().startswith("```mermaid") or "graph LR" in out
    assert "ingestion --> events" in out and "ingestion --> agents" in out
