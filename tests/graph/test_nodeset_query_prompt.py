from tools.graph.reader import nodes, harvest


def test_query_and_prompt_are_nodes():
    ns = nodes()
    assert ns.get("GraphQuery") and ns.get("Prompt")


def test_consumed_by_edges_resolve():
    edges = harvest()
    code_ids = nodes()["CodeUnit"]
    cb = [e for e in edges if e.type == "consumed_by"]
    assert cb, "expected consumed_by edges"
    assert all(e.dst.split(":", 1)[1] in code_ids for e in cb)   # all resolve to real units
