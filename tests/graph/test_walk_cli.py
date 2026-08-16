import subprocess
import sys


def test_walk_cli_runs():
    p = subprocess.run([sys.executable, "-m", "tools.graph", "walk", "capabilities:ask-the-corpus",
                        "--dir", "out", "--depth", "1"], capture_output=True, text=True)
    assert p.returncode == 0
    assert "capabilities:ask-the-corpus" in p.stdout
