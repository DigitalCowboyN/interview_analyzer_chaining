import subprocess
import sys


def test_cli_check_exits_zero():
    proc = subprocess.run([sys.executable, "-m", "tools.capability", "check"],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "capability-check" in proc.stdout


def test_capabilities_in_knowledge_registry():
    from tools.knowledge.check import DOMAINS
    slugs = {slug for slug, _ in DOMAINS}
    assert "capabilities" in slugs
