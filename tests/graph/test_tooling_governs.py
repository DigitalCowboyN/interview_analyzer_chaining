from tools.graph.reader import harvest, nodes
from tools.graph.check import check_endpoints


def _governs(root="."):
    return {(e.src, e.dst) for e in harvest(root) if e.type == "governs"}


def test_tooling_adrs_now_govern_their_code():
    g = _governs()
    assert ("adr:27", "code:tools.graph.traverse") in g       # lazy walk governs traversal
    assert ("adr:27", "code:tools.code.reader") in g          # + symbol derivation
    assert ("adr:25", "code:tools.graph.neighbors") in g      # ephemeral substrate
    assert ("adr:26", "code:tools.code.reader") in g          # code intake (tools/code/ dir)
    assert ("adr:20", "code:tools.graph.traverse") in g       # graph model (tools/graph/ dir)


def test_no_dangling_after_governs():
    assert check_endpoints(harvest("."), nodes(".")) == []    # every governs endpoint resolves
