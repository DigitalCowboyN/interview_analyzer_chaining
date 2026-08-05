from tools.capability.reader import Capability
from tools.capability.render import render_index

CAPS = [
    Capability("enrich-fragments", "primary", "core", "", ["enrichment", "agents"], "Enrich fragments.", "p"),
    Capability("extract-claims", "child", "", "enrich-fragments", ["enrichment.executor"], "Pull claims.", "p"),
    Capability("project-events-to-graph", "primary", "enabling", "", ["projections"], "Build the read model.", "p"),
]


def test_index_groups_by_tier_and_nests_children():
    out = render_index(CAPS)
    assert "## core" in out and "## enabling" in out
    assert "### enrich-fragments" in out and "enrichment, agents" in out
    assert "extract-claims" in out and "enrichment.executor" in out
    assert out.index("### enrich-fragments") < out.index("## enabling")


def test_index_is_deterministic():
    assert render_index(CAPS) == render_index(list(reversed(CAPS)))
