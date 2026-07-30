import json
import os


def test_settings_registers_both_hooks():
    cfg = json.load(open(".claude/settings.json", encoding="utf-8"))
    hooks = cfg.get("hooks", {})
    ups = json.dumps(hooks.get("UserPromptSubmit", []))
    ptu = json.dumps(hooks.get("PostToolUse", []))
    assert "tools.adr context" in ups        # read side
    assert "tools.adr nudge" in ptu           # capture side


def test_precommit_hook_is_executable_and_nonblocking():
    path = ".githooks/pre-commit"
    assert os.path.exists(path)
    assert os.access(path, os.X_OK)
    body = open(path, encoding="utf-8").read()
    assert "adr" in body and "exit 0" in body   # never blocks the commit
