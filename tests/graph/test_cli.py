import subprocess, sys


def test_check_exits_zero():
    p = subprocess.run([sys.executable, "-m", "tools.graph", "check"], capture_output=True, text=True)
    assert p.returncode == 0 and "graph-check" in p.stdout


def test_neighbors_reports_edges():
    p = subprocess.run([sys.executable, "-m", "tools.graph", "neighbors", "code:api"],
                       capture_output=True, text=True)
    assert p.returncode == 0


def test_graph_in_knowledge_registry():
    from tools.knowledge.check import DOMAINS
    assert "graph" in {slug for slug, _ in DOMAINS}
