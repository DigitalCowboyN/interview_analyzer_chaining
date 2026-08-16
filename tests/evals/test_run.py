import json
import os

from tools.graph.traverse import Node, Subgraph
from evals.graph.run import load_scenarios, score, substantive


def _sg(nodes):
    # nodes: {address: context}
    return Subgraph(nodes={a: Node(address=a, type="", context=c) for a, c in nodes.items()}, edges=[])


def test_substantive_requires_real_words():
    assert not substantive("")
    assert not substantive("x y")               # under the word floor
    assert substantive("This module walks the graph.")


def test_score_context_coverage_and_recall():
    scenario = {"gold_context": ["code:a", "code:b", "capabilities:c"]}
    # a is reached WITH context, b reached WITHOUT, c not reached at all
    sg = _sg({"code:a": "Does a real thing here.", "code:b": "", "code:x": "extra reached node."})
    s = score(sg, scenario)
    assert s["coverage"] == 0.5                  # of {a,b} code gold, only a has substantive context
    assert round(s["recall"], 3) == round(2 / 3, 3)  # a,b reached of {a,b,c}
    assert s["overfetch"] == 1                   # code:x reached but not gold
    assert s["missing"] == ["capabilities:c"]


def test_load_scenarios_reads_all():
    # load_scenarios just parses the scenario dir; validity-on-real-graph is checked separately
    ids = {s["id"] for s in load_scenarios()}
    assert {"explore-tools-graph", "spec-code-intake", "trace-classify-obligation"} <= ids


def test_gold_addresses_resolve_on_the_real_graph():
    # every entry + gold_context address must be a real node — no dangling gold
    from tools.graph.reader import nodes
    from tools.graph.registry import NODE_DOMAINS
    ns = nodes(".")
    real = {f"{NODE_DOMAINS[t]}:{i}" for t, ids in ns.items() for i in ids}
    for s in load_scenarios():
        for addr in list(s["entry"]) + list(s["gold_context"]):
            assert addr in real, f"{s['id']}: gold address {addr} does not resolve"
