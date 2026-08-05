import json
import subprocess
import sys


def _context(prompt):
    return subprocess.run([sys.executable, "-m", "tools.adr", "context"],
                          input=json.dumps({"prompt": prompt}),
                          capture_output=True, text=True)


def test_architectural_prompt_gets_pointer_not_table():
    out = _context("should we change this architecture decision?").stdout
    assert "docs/adr/index.md" in out
    assert "| 0001 |" not in out  # the full table is gone


def test_non_architectural_prompt_is_silent():
    out = _context("fix this typo in the readme").stdout
    assert out.strip() == ""
