from __future__ import annotations

import argparse
import os
import sys

from tools.graphq.check import run_all
from tools.graphq.reader import load_queries
from tools.graphq.render import render_catalog

CATALOG = "docs/graph-queries/index.md"


def cmd_index(args) -> int:
    os.makedirs(os.path.dirname(CATALOG), exist_ok=True)
    with open(CATALOG, "w", encoding="utf-8") as fh:
        fh.write(render_catalog(load_queries()))
    print(f"wrote {CATALOG}")
    return 0


def cmd_check(args) -> int:
    findings = run_all()
    if findings:
        print(f"graphq-check: {len(findings)} warning(s):")
        for f in findings:
            print(f"  - {f.message}")
    else:
        print("graphq-check: clean")
    return 0  # NON-BLOCKING


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="tools.graphq")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("index")
    sub.add_parser("check")
    args = parser.parse_args(argv)
    return {"index": cmd_index, "check": cmd_check}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
