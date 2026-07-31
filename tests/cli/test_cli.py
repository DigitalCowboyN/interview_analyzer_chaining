import glob
import subprocess
import sys


def test_cli_help_and_check_exit_zero():
    for cmd in ("help", "check"):
        proc = subprocess.run([sys.executable, "-m", "tools.cli", cmd], capture_output=True, text=True)
        assert proc.returncode == 0, (cmd, proc.stderr)


def test_tools_cli_is_stdlib_only():
    banned = ("import yaml", "from src", "import pydantic", "import requests")
    for path in glob.glob("tools/cli/*.py"):
        src = open(path, encoding="utf-8").read()
        for b in banned:
            assert b not in src, f"{path} contains banned import: {b}"
