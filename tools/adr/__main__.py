from __future__ import annotations

import argparse
import json
import sys

from tools.adr.check import run_all
from tools.adr.index import write_generated
from tools.adr.intent import is_architectural
from tools.adr.scaffold import new_adr

DEFAULT_ADR_DIR = "docs/adr"
DEFAULT_SPECS_DIR = "docs/superpowers/specs"


def _read_stdin_json() -> dict:
    try:
        return json.loads(sys.stdin.read() or "{}")
    except Exception:
        return {}


def cmd_index(args) -> int:
    write_generated(args.adr_dir)
    print(f"regenerated {args.adr_dir}/index.md and log.md")
    return 0


def cmd_check(args) -> int:
    findings = run_all(args.adr_dir, args.specs_dir)
    if findings:
        print(f"adr-check: {len(findings)} warning(s):")
        for f in findings:
            print(f"  - {f.message}")
    else:
        print("adr-check: clean")
    return 0  # NON-BLOCKING: always 0


def cmd_new(args) -> int:
    path = new_adr(args.adr_dir, args.title)
    print(f"created {path} — fill it in, set `source:` and `date:`, then `make adr-index`")
    return 0


def cmd_context(args) -> int:
    # UserPromptSubmit hook: stdout is injected as context. Quiet unless architectural.
    prompt = _read_stdin_json().get("prompt", "")
    if is_architectural(prompt):
        try:
            print(open(f"{args.adr_dir}/index.md", encoding="utf-8").read())
            print("(Before locking a decision, consult these ADRs; supersede rather than silently override.)")
        except FileNotFoundError:
            pass
    return 0


def cmd_nudge(args) -> int:
    # PostToolUse(Write) hook: remind to capture decisions when a spec lands.
    path = _read_stdin_json().get("tool_input", {}).get("file_path", "")
    if "docs/superpowers/specs/" in path.replace("\\", "/"):
        print("This spec may lock decisions — capture them as ADR(s) "
              "(`python -m tools.adr new \"<title>\"`) and set `source:` to this spec.")
    return 0


def main(argv=None) -> int:
    # Shared options live on a parent parser so they are valid AFTER the
    # subcommand (e.g. `tools.adr check --adr-dir X --specs-dir Y`).
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--adr-dir", default=DEFAULT_ADR_DIR)
    common.add_argument("--specs-dir", default=DEFAULT_SPECS_DIR)

    parser = argparse.ArgumentParser(prog="tools.adr")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("index", parents=[common])
    sub.add_parser("check", parents=[common])
    p_new = sub.add_parser("new", parents=[common]); p_new.add_argument("title")
    sub.add_parser("context", parents=[common])
    sub.add_parser("nudge", parents=[common])
    args = parser.parse_args(argv)
    return {
        "index": cmd_index, "check": cmd_check, "new": cmd_new,
        "context": cmd_context, "nudge": cmd_nudge,
    }[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
