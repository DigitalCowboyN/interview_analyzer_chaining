from tools.capability.reader import Capability
from tools.capability.render import render_index

CAPS = [
    Capability("enrich-fragments", "primary", "core", "", ["enrichment"], "Enrich fragments.", "p", "product"),
    Capability("extract-claims", "child", "", "enrich-fragments", ["enrichment.executor"], "Pull claims.", "p", ""),
    Capability("map-the-code", "child", "", "maintain-a-guarded-knowledge-graph", ["tools.code"], "Map the code.", "p", ""),
    Capability("maintain-a-guarded-knowledge-graph", "primary", "core", "", [], "Keep the repo honest.", "p", "operations"),
]


def test_index_groups_by_category_then_tier():
    out = render_index(CAPS)
    assert "## product" in out and "## operations" in out
    assert "### core" in out
    assert "#### enrich-fragments" in out and "enrichment" in out
    # product section precedes operations (CATEGORIES order)
    assert out.index("## product") < out.index("## operations")
    # child nested under its operations primary
    assert out.index("## operations") < out.index("map-the-code")


def test_empty_categories_are_omitted():
    out = render_index(CAPS)
    assert "## strategic" not in out and "## supporting" not in out  # reserved, unpopulated


def test_index_is_deterministic():
    assert render_index(CAPS) == render_index(list(reversed(CAPS)))
