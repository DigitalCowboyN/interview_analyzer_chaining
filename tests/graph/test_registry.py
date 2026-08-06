from tools.graph.registry import EDGES, NODE_DOMAINS, EdgeType, PropSpec


def test_registry_well_formed():
    names = {e.name for e in EDGES}
    assert {"implements", "child_of", "depends_on", "governs", "supersedes"} <= names
    for e in EDGES:
        assert e.from_type in NODE_DOMAINS
        assert all(t in NODE_DOMAINS for t in e.to_type.split("|"))
        assert e.source in ("authored", "derived")


def test_edge_properties_are_supported():
    # the extensibility capacity: an edge type can carry typed properties (e.g. tests)
    e = EdgeType("verifies", "verified_by", "Capability", "CodeUnit", "authored",
                 field="verifies", properties=[PropSpec("test_type", enum=["unit", "integration"])])
    assert e.properties[0].name == "test_type" and "unit" in e.properties[0].enum
