# tests/graph/test_check.py
from tools.graph.reader import Edge
from tools.graph.check import check_endpoints, check_registry, run_all
from tools.graph.registry import EdgeType


def test_endpoints_flag_dangling():
    edges = [Edge("implements", "capabilities:x", "code:gone")]
    node_ids = {"Capability": {"x"}, "CodeUnit": {"api"}, "ADR": set()}
    msgs = " ".join(f.message for f in check_endpoints(edges, node_ids))
    assert "code:gone" in msgs


def test_endpoints_clean_when_resolvable():
    edges = [Edge("implements", "capabilities:x", "code:api")]
    node_ids = {"Capability": {"x"}, "CodeUnit": {"api"}, "ADR": set()}
    assert check_endpoints(edges, node_ids) == []


def test_registry_flags_unknown_node_type():
    bad = [EdgeType("weird", "", "Nope", "CodeUnit", "authored")]
    msgs = " ".join(f.message for f in check_registry(bad, {"CodeUnit": "code"}))
    assert "Nope" in msgs


def test_run_all_returns_list_never_raises(tmp_path):
    assert isinstance(run_all(str(tmp_path)), list)
