from tools.graph.reader import nodes, harvest


def test_glossary_terms_are_nodes():
    ns = nodes()
    assert "GlossaryTerm" in ns and len(ns["GlossaryTerm"]) > 50   # ~111 terms


def test_a_defined_in_edge_resolves_to_a_code_unit():
    edges = harvest()
    di = [e for e in edges if e.type == "defined_in"]
    assert di, "expected at least one defined_in edge"
    code_ids = nodes()["CodeUnit"]
    # at least one term's source maps to a real code unit (e.g. events/, projections/)
    assert any(e.dst.split(":", 1)[1] in code_ids for e in di)
