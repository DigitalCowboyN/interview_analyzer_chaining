import json
import subprocess
import sys


def _run(args, stdin=""):
    return subprocess.run([sys.executable, "-m", "tools.knowledge", *args],
                          input=stdin, capture_output=True, text=True)


def test_check_exits_zero():
    proc = _run(["check"])
    assert proc.returncode == 0, proc.stderr
    assert "knowledge-check" in proc.stdout


def test_nudge_fires_on_spec_path():
    proc = _run(["nudge"], stdin=json.dumps(
        {"tool_input": {"file_path": "docs/superpowers/specs/2026-08-05-x-design.md"}}))
    assert proc.returncode == 0
    assert "docs/index.md" in proc.stdout


def test_nudge_fires_on_plan_path():
    proc = _run(["nudge"], stdin=json.dumps(
        {"tool_input": {"file_path": "docs/superpowers/plans/2026-08-05-x.md"}}))
    assert "Knowledge-graph check" in proc.stdout


def test_nudge_silent_on_other_path():
    proc = _run(["nudge"], stdin=json.dumps({"tool_input": {"file_path": "src/api/main.py"}}))
    assert proc.returncode == 0
    assert proc.stdout.strip() == ""


def test_nudge_survives_null_tool_input():
    # a JSON null tool_input must not raise (non-blocking hook invariant)
    proc = _run(["nudge"], stdin=json.dumps({"tool_input": None}))
    assert proc.returncode == 0
    assert proc.stdout.strip() == ""
