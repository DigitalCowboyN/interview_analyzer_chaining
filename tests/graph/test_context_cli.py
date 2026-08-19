import subprocess
import sys


def test_walk_cli_level_symbol():
    p = subprocess.run([sys.executable, "-m", "tools.graph", "walk",
                        "code:tools.graph.traverse", "--level", "symbol", "--depth", "1"],
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert "code:tools.graph.traverse.walk" in p.stdout        # symbol reachable via CLI


def test_context_cli_returns_minimal_context_with_intent():
    p = subprocess.run([sys.executable, "-m", "tools.graph", "context",
                        "code:tools.graph.traverse"], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert "adr:" in p.stdout                                  # walked up to the governing ADR
