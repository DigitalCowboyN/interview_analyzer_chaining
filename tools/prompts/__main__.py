from __future__ import annotations

import argparse
import os
import sys

from tools.prompts.check import run_all
from tools.prompts.reader import load_prompt_entries
from tools.prompts.render import render_catalog

CATALOG = "docs/prompts/index.md"


def cmd_index(args) -> int:
    os.makedirs(os.path.dirname(CATALOG), exist_ok=True)
    with open(CATALOG, "w", encoding="utf-8") as fh:
        fh.write(render_catalog(load_prompt_entries()))
    print(f"wrote {CATALOG}")
    return 0


def cmd_check(args) -> int:
    findings = run_all()
    if findings:
        print(f"prompt-check: {len(findings)} warning(s):")
        for f in findings:
            print(f"  - {f.message}")
    else:
        print("prompt-check: clean")
    return 0  # NON-BLOCKING


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="tools.prompts")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("index")
    sub.add_parser("check")
    args = parser.parse_args(argv)
    return {"index": cmd_index, "check": cmd_check}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
