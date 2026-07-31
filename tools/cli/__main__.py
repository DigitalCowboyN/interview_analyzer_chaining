from __future__ import annotations

import argparse
import os
import sys

from tools.cli.check import run_all
from tools.cli.reader import module_entrypoints, parse_makefile
from tools.cli.render import render_catalog, render_help

CATALOG = "docs/cli/index.md"


def _commands(root: str = "."):
    return parse_makefile(os.path.join(root, "Makefile")) + module_entrypoints(root)


def cmd_help(args) -> int:
    print(render_help(_commands()), end="")
    return 0


def cmd_index(args) -> int:
    os.makedirs(os.path.dirname(CATALOG), exist_ok=True)
    with open(CATALOG, "w", encoding="utf-8") as fh:
        fh.write(render_catalog(_commands()))
    print(f"wrote {CATALOG}")
    return 0


def cmd_check(args) -> int:
    findings = run_all()
    if findings:
        print(f"cli-check: {len(findings)} warning(s):")
        for f in findings:
            print(f"  - {f.message}")
    else:
        print("cli-check: clean")
    return 0  # NON-BLOCKING: always 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="tools.cli")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("help")
    sub.add_parser("index")
    sub.add_parser("check")
    args = parser.parse_args(argv)
    return {"help": cmd_help, "index": cmd_index, "check": cmd_check}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
