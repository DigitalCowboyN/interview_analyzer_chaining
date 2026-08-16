from tools.code.reader import CodeUnit
from tools.code.render import render_docstring_backlog, render_index, render_pipeline

UNITS = [
    CodeUnit("api", level="package"),
    CodeUnit("api.main", level="module", depends_on=["events", "api.routers"]),
    CodeUnit("events", level="package"),
]


def test_render_index_groups_by_level():
    axes = {"api": ("product", "deterministic")}
    out = render_index(UNITS, axes)
    assert "## Packages" in out and "## Modules" in out
    assert "api.main" in out and "events, api.routers" in out
    assert "| api | product | deterministic |" in out


def test_render_pipeline_is_mermaid():
    out = render_pipeline(UNITS)
    assert "graph LR" in out
    assert "api.main --> events" in out and "api.main --> api.routers" in out


def test_render_docstring_backlog_lists_only_undocumented_modules():
    units = [
        CodeUnit("api", level="package"),                                  # package: never listed
        CodeUnit("api.main", level="module", description=""),              # module, no docstring: listed
        CodeUnit("api.ok", level="module", description="Has one."),        # module w/ docstring: not
        CodeUnit("config", level="module", description=""),                # top-level module: grouped separately
    ]
    out = render_docstring_backlog(units)
    assert "**2 module(s)** remaining." in out
    assert "## api" in out and "- [ ] api.main" in out
    assert "api.ok" not in out                                            # documented → absent
    assert "## (top-level)" in out and "- [ ] config" in out
