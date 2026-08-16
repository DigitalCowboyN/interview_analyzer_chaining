from tools.code.reader import CodeUnit
from tools.code.render import render_index, render_pipeline

UNITS = [
    CodeUnit("api", level="package"),
    CodeUnit("api.main", level="module", depends_on=["events", "api.routers"]),
    CodeUnit("events", level="package"),
]


def test_render_index_groups_by_level():
    out = render_index(UNITS)
    assert "## Packages" in out and "## Modules" in out
    assert "api.main" in out and "events, api.routers" in out


def test_render_pipeline_is_mermaid():
    out = render_pipeline(UNITS)
    assert "graph LR" in out
    assert "api.main --> events" in out and "api.main --> api.routers" in out
