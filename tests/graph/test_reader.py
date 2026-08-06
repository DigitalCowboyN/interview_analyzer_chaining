# tests/graph/test_reader.py
from tools.graph.reader import harvest, nodes, Edge
from tools.graph.registry import EdgeType


def test_harvest_real_edges_typed_and_addressed():
    edges = harvest(".")
    kinds = {e.type for e in edges}
    assert {"implements", "child_of", "depends_on", "governs", "supersedes"} <= kinds
    impl = [e for e in edges if e.type == "implements"][0]
    assert impl.src.startswith("capabilities:") and impl.dst.startswith("code:")


def test_child_of_direction():
    # the field `parent` lives on the child; the edge points child -> parent
    edges = harvest(".")
    co = [e for e in edges if e.type == "child_of"]
    assert co and all(e.src.startswith("capabilities:") and e.dst.startswith("capabilities:") for e in co)


def test_governs_resolves_path_to_units():
    edges = harvest(".")
    gov = [e for e in edges if e.type == "governs"]
    assert gov and all(e.src.startswith("adr:") and e.dst.startswith("code:") for e in gov)


def test_harvest_is_registry_driven_extensible():
    # a NEW authored edge on existing node types, added only to the passed registry,
    # is harvested with NO reader change (proves extensibility)
    extra = [EdgeType("supersedes", "superseded_by", "ADR", "ADR", "authored",
                      field="supersedes", resolve="id")]
    out = harvest(".", edges=extra)
    assert out and all(e.type == "supersedes" for e in out)


def test_nodes_addressable():
    n = nodes(".")
    assert "CodeUnit" in n and "Capability" in n and "ADR" in n
