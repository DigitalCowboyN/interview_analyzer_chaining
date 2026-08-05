from __future__ import annotations

import argparse
import json
import sys

from tools.knowledge.check import run_all

_SPEC_PLAN_DIRS = ("docs/superpowers/specs/", "docs/superpowers/plans/")


def _read_stdin_json() -> dict:
    try:
        return json.loads(sys.stdin.read() or "{}")
    except Exception:
        return {}


def cmd_check(args) -> int:
    findings = run_all()
    if findings:
        print(f"knowledge-check: {len(findings)} warning(s):")
        for f in findings:
            print(f"  - {f.message}")
    else:
        print("knowledge-check: clean")
    return 0  # NON-BLOCKING


def cmd_nudge(args) -> int:
    # PostToolUse(Write) hook: honesty-check reminder when a spec/plan lands.
    path = _read_stdin_json().get("tool_input", {}).get("file_path", "").replace("\\", "/")
    if any(d in path for d in _SPEC_PLAN_DIRS):
        print("This spec/plan likely touches the knowledge graph. Review it against "
              "docs/index.md: for each domain it affects, consult the bundle and run "
              "its `make <domain>-check`; record a '## Knowledge-graph check' addendum "
              "(per-domain touched/consulted + verdict). If it locks architectural "
              "decisions, also capture ADR(s).")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="tools.knowledge")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    sub.add_parser("nudge")
    args = parser.parse_args(argv)
    return {"check": cmd_check, "nudge": cmd_nudge}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
