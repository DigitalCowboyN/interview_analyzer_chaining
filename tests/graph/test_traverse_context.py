from tools.graph.traverse import walk


def test_walk_fills_context_for_real_nodes():
    # a real capability node exists in the repo; its context should be non-empty
    sg = walk("capabilities:ask-the-corpus", direction="out", depth=1)
    n = sg.nodes["capabilities:ask-the-corpus"]
    assert n.type == "Capability"
    assert n.context.strip() != ""
