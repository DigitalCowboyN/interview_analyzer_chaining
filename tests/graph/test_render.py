# tests/graph/test_render.py
from tools.graph.reader import Edge
from tools.graph.render import render_catalog, render_graph

EDGES = [Edge("implements", "capabilities:x", "code:api"),
         Edge("depends_on", "code:api", "code:ui")]
NODES = {"Capability": {"x"}, "CodeUnit": {"api", "ui"}}


def test_catalog_lists_edge_types_and_counts():
    out = render_catalog(EDGES, NODES)
    assert "implements" in out and "depends_on" in out
    assert "```mermaid" in out            # the meta-schema diagram
    assert "properties" in out            # edge-property capacity is visible


def test_graph_has_per_edge_type_sections():
    out = render_graph(EDGES)
    assert "## implements" in out and "## depends_on" in out
    assert "capabilities:x" in out and "code:api" in out
    assert out.count("```mermaid") >= 2   # one diagram per edge type
