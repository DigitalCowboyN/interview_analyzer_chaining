"""CLI entry point for the `tools.api` domain: `index` renders the live FastAPI route table
into `docs/api/index.md`, `check` runs the non-blocking api-surface findings."""

from __future__ import annotations

import argparse
import os
import sys

from tools.api.check import run_all
from tools.api.reader import live_endpoints, load_app
from tools.api.render import render_catalog

CATALOG = "docs/api/index.md"


def cmd_index(args) -> int:
    app = load_app()
    os.makedirs(os.path.dirname(CATALOG), exist_ok=True)
    with open(CATALOG, "w", encoding="utf-8") as fh:
        fh.write(render_catalog(live_endpoints(app)))
    print(f"wrote {CATALOG}")
    return 0


def cmd_check(args) -> int:
    findings = run_all()
    if findings:
        print(f"api-check: {len(findings)} warning(s):")
        for f in findings:
            print(f"  - {f.message}")
    else:
        print("api-check: clean")
    return 0  # NON-BLOCKING: always 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="tools.api")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("index")
    sub.add_parser("check")
    args = parser.parse_args(argv)
    return {"index": cmd_index, "check": cmd_check}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
