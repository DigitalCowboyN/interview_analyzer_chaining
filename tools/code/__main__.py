from __future__ import annotations

import argparse
import os
import sys

from tools.code.check import run_all
from tools.code.reader import load_units
from tools.code.render import render_index, render_pipeline
from tools.graph.classify import derive_axes

CODE_DIR = "docs/code"
INDEX = f"{CODE_DIR}/index.md"
PIPELINE = f"{CODE_DIR}/pipeline.md"


def cmd_index(args) -> int:
    os.makedirs(CODE_DIR, exist_ok=True)
    units = load_units()
    with open(INDEX, "w", encoding="utf-8") as fh:
        fh.write(render_index(units, derive_axes()))
    with open(PIPELINE, "w", encoding="utf-8") as fh:
        fh.write(render_pipeline(units))
    print(f"wrote {INDEX} + {PIPELINE}")
    return 0


def cmd_check(args) -> int:
    findings = run_all()
    if findings:
        print(f"code-check: {len(findings)} warning(s):")
        for f in findings:
            print(f"  - {f.message}")
    else:
        print("code-check: clean")
    return 0  # NON-BLOCKING


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="tools.code")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("index")
    sub.add_parser("check")
    args = parser.parse_args(argv)
    return {"index": cmd_index, "check": cmd_check}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
