from __future__ import annotations

import argparse
import os
import sys

from tools.capability.reader import load_capabilities, real_code_units
from tools.usecase.check import run_all
from tools.usecase.coverage import coverage
from tools.usecase.reader import load_use_cases
from tools.usecase.render import render_index

INDEX = "docs/use-cases/index.md"


def _coverage(root: str = "."):
    return coverage(
        load_use_cases(root), load_capabilities(root), real_code_units(root)
    )


def cmd_index(args) -> int:
    os.makedirs(os.path.dirname(INDEX), exist_ok=True)
    with open(INDEX, "w", encoding="utf-8") as fh:
        fh.write(render_index(load_use_cases(), _coverage()))
    print(f"wrote {INDEX}")
    return 0


def cmd_check(args) -> int:
    findings = run_all()
    if findings:
        print(f"usecase-check: {len(findings)} warning(s):")
        for f in findings:
            print(f"  - {f.message}")
    else:
        print("usecase-check: clean")
    return 0  # NON-BLOCKING


def cmd_coverage(args) -> int:
    cov = _coverage()
    for slug in sorted(cov):
        print(f"{cov[slug]:18} {slug}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="tools.usecase")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("index")
    sub.add_parser("check")
    sub.add_parser("coverage")
    args = parser.parse_args(argv)
    return {"index": cmd_index, "check": cmd_check, "coverage": cmd_coverage}[args.cmd](
        args
    )


if __name__ == "__main__":
    sys.exit(main())
