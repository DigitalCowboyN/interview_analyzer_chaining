import tools.graph.check as gc
from tools.graph.check import check_reachability
from tools.graph.traverse import Subgraph, Node


def _sg(addresses):
    return Subgraph(nodes={a: Node(address=a, type="", context="") for a in addresses}, edges=[])


def test_unreached_code_unit_is_flagged(monkeypatch):
    monkeypatch.setattr(gc, "nodes", lambda root=".": {
        "Capability": {"cap"}, "UseCase": set(), "ADR": set(), "CodeUnit": {"reached", "orphan"}})
    # walk from the intents reaches only code:reached
    monkeypatch.setattr(gc, "walk",
                        lambda entry, direction="both", depth=None, root=".": _sg(["capabilities:cap", "code:reached"]))
    msgs = [f.message for f in check_reachability()]
    assert any("code:orphan" in m for m in msgs)
    assert not any("code:reached" in m for m in msgs)


def test_all_reached_is_clean(monkeypatch):
    monkeypatch.setattr(gc, "nodes", lambda root=".": {
        "Capability": {"cap"}, "UseCase": set(), "ADR": set(), "CodeUnit": {"reached"}})
    monkeypatch.setattr(gc, "walk",
                        lambda entry, direction="both", depth=None, root=".": _sg(["capabilities:cap", "code:reached"]))
    assert check_reachability() == []
