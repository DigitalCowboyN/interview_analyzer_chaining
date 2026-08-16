"""CLI entry point for the `tools.capability` domain: `index` renders
`docs/capabilities/index.md` from the authored capability docs, `check` runs the
non-blocking capability findings."""

from __future__ import annotations

import argparse
import os
import sys

from tools.capability.check import run_all
from tools.capability.reader import load_capabilities
from tools.capability.render import render_index

INDEX = "docs/capabilities/index.md"


def cmd_index(args) -> int:
    os.makedirs(os.path.dirname(INDEX), exist_ok=True)
    with open(INDEX, "w", encoding="utf-8") as fh:
        fh.write(render_index(load_capabilities()))
    print(f"wrote {INDEX}")
    return 0


def cmd_check(args) -> int:
    findings = run_all()
    if findings:
        print(f"capability-check: {len(findings)} warning(s):")
        for f in findings:
            print(f"  - {f.message}")
    else:
        print("capability-check: clean")
    return 0  # NON-BLOCKING


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="tools.capability")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("index")
    sub.add_parser("check")
    args = parser.parse_args(argv)
    return {"index": cmd_index, "check": cmd_check}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
