from tools.graph.traverse import walk, Subgraph
from tools.graph.reader import Edge


def _fake_harvest(edges):
    return lambda root=".": list(edges)


def test_walk_out_depth_1(monkeypatch):
    import tools.graph.reader as reader
    edges = [Edge("implements", "capabilities:a", "code:x"),
             Edge("depends_on", "code:x", "code:y")]
    monkeypatch.setattr(reader, "harvest", _fake_harvest(edges))
    sg = walk("capabilities:a", direction="out", depth=1)
    assert set(sg.nodes) == {"capabilities:a", "code:x"}          # 1 hop only
    assert [e.dst for e in sg.edges] == ["code:x"]


def test_walk_out_to_exhaustion(monkeypatch):
    import tools.graph.reader as reader
    edges = [Edge("implements", "capabilities:a", "code:x"),
             Edge("depends_on", "code:x", "code:y")]
    monkeypatch.setattr(reader, "harvest", _fake_harvest(edges))
    sg = walk("capabilities:a", direction="out", depth=None)
    assert set(sg.nodes) == {"capabilities:a", "code:x", "code:y"}  # full chain


def test_walk_in_uses_reverse_edges(monkeypatch):
    import tools.graph.reader as reader
    edges = [Edge("implements", "capabilities:a", "code:x")]
    monkeypatch.setattr(reader, "harvest", _fake_harvest(edges))
    sg = walk("code:x", direction="in", depth=1)
    assert set(sg.nodes) == {"code:x", "capabilities:a"}


def test_walk_cycle_terminates(monkeypatch):
    import tools.graph.reader as reader
    edges = [Edge("depends_on", "code:x", "code:y"),
             Edge("depends_on", "code:y", "code:x")]
    monkeypatch.setattr(reader, "harvest", _fake_harvest(edges))
    sg = walk("code:x", direction="out", depth=None)
    assert set(sg.nodes) == {"code:x", "code:y"}


def test_walk_unknown_entry_is_singleton(monkeypatch):
    import tools.graph.reader as reader
    monkeypatch.setattr(reader, "harvest", _fake_harvest([]))
    sg = walk("code:nope", depth=1)
    assert set(sg.nodes) == {"code:nope"} and sg.edges == []
