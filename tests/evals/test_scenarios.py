from evals.graph.run import load_scenarios
from tools.graph.reader import nodes
from tools.graph.registry import NODE_DOMAINS

_VALID_CATEGORY = {"bug-fix", "new-component", "refactor", "governance",
                   "pipeline", "deployment", "exploration", "meta", "spec", "implement"}
_VALID_EXPECTED = {"solvable", "partial", "gap"}


def _real_addresses():
    ns = nodes(".")
    return {f"{NODE_DOMAINS[t]}:{i}" for t, ids in ns.items() for i in ids}


def test_every_scenario_has_required_fields_and_valid_enums():
    for s in load_scenarios():
        assert s["id"] and s["task"] and s["entry"] and s["gold_context"]
        assert s["category"] in _VALID_CATEGORY, s["id"]
        assert s["expected"] in _VALID_EXPECTED, s["id"]
        if s["expected"] in ("partial", "gap"):
            assert s.get("gap_note"), f"{s['id']} missing gap_note"


def test_no_dangling_gold_addresses():
    real = _real_addresses()
    for s in load_scenarios():
        for addr in list(s["entry"]) + list(s["gold_context"]):
            assert addr in real, f"{s['id']}: gold address {addr} does not resolve"


def test_corpus_is_broad():
    cats = {s["category"] for s in load_scenarios()}
    assert {"bug-fix", "refactor", "governance", "pipeline", "deployment"} <= cats
