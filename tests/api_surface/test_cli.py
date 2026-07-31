import subprocess
import sys


def test_cli_check_exits_zero():
    proc = subprocess.run([sys.executable, "-m", "tools.api", "check"], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "api-check" in proc.stdout
