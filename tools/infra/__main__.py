# tools/infra/__main__.py
"""CLI for the tools.infra domain: `index` renders docs/infra/index.md; `check` runs the
non-blocking findings; `list` prints the services."""
from __future__ import annotations

import argparse
import os
import sys

from tools.infra.check import run_all
from tools.infra.reader import (
    configured_by_pairs, load_env_vars, load_services, requires_pairs, runs_pairs, talks_to_pairs,
)
from tools.infra.render import render_index

INFRA_DIR = "docs/infra"
INDEX = f"{INFRA_DIR}/index.md"


def cmd_index(args) -> int:
    os.makedirs(INFRA_DIR, exist_ok=True)
    with open(INDEX, "w", encoding="utf-8") as fh:
        fh.write(render_index(load_services(), load_env_vars(), runs_pairs(),
                              talks_to_pairs(), requires_pairs(), configured_by_pairs()))
    print(f"wrote {INDEX}")
    return 0


def cmd_check(args) -> int:
    findings = run_all()
    if findings:
        print(f"infra-check: {len(findings)} warning(s):")
        for f in findings:
            print(f"  - {f.message}")
    else:
        print("infra-check: clean")
    return 0  # NON-BLOCKING


def cmd_list(args) -> int:
    for s in load_services():
        print(f"{s.kind:8} {s.id}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="tools.infra")
    sub = parser.add_subparsers(dest="cmd", required=True)
    for c in ("index", "check", "list"):
        sub.add_parser(c)
    args = parser.parse_args(argv)
    return {"index": cmd_index, "check": cmd_check, "list": cmd_list}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
