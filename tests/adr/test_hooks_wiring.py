import json
import os


def test_settings_registers_both_hooks():
    cfg = json.load(open(".claude/settings.json", encoding="utf-8"))
    hooks = cfg.get("hooks", {})
    ups = json.dumps(hooks.get("UserPromptSubmit", []))
    ptu = json.dumps(hooks.get("PostToolUse", []))
    # read side: slimmed ADR pointer, still via the shared interpreter resolver
    assert "with-project-py.sh tools.adr context" in ups
    # capture side: knowledge-graph honesty-check nudge on spec/plan writes
    assert "with-project-py.sh tools.knowledge nudge" in ptu


def test_precommit_hook_is_executable_and_nonblocking():
    path = ".githooks/pre-commit"
    assert os.path.exists(path)
    assert os.access(path, os.X_OK)
    body = open(path, encoding="utf-8").read()
    # resolver-driven: runs the changed-domain checks (+ graph), and never blocks the commit
    assert "changed-domains" in body and "graph" in body and "exit 0" in body
