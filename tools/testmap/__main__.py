"""CLI entry point for the `tools.testmap` domain: `index` renders `docs/tests/index.md`
(the test suite as a graph node set plus its derived capability/use-case verification
rollup), `check` runs the non-blocking findings, and `verification` prints the rollup."""

from __future__ import annotations

import argparse
import os
import sys

from tools.capability.reader import load_capabilities
from tools.usecase.reader import load_use_cases
from tools.testmap.check import run_all
from tools.testmap.reader import load_tests
from tools.testmap.render import render_index
from tools.testmap.verification import verify_capabilities, verify_use_cases

INDEX = "docs/tests/index.md"


def _rollups(root: str = "."):
    tests = load_tests(root)
    caps = load_capabilities(root)
    ucs = load_use_cases(root)
    return tests, verify_capabilities(caps, tests), verify_use_cases(ucs, caps, tests)


def cmd_index(args) -> int:
    os.makedirs(os.path.dirname(INDEX), exist_ok=True)
    tests, cap_ver, uc_ver = _rollups()
    with open(INDEX, "w", encoding="utf-8") as fh:
        fh.write(render_index(tests, cap_ver, uc_ver))
    print(f"wrote {INDEX}")
    return 0


def cmd_check(args) -> int:
    findings = run_all()
    if findings:
        print(f"testmap-check: {len(findings)} warning(s):")
        for f in findings:
            print(f"  - {f.message}")
    else:
        print("testmap-check: clean")
    return 0  # NON-BLOCKING


def cmd_verification(args) -> int:
    _, cap_ver, uc_ver = _rollups()
    for slug in sorted(uc_ver):
        print(f"use-case      {uc_ver[slug]:18} {slug}")
    for slug in sorted(cap_ver):
        print(f"capability    {cap_ver[slug]:18} {slug}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="tools.testmap")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("index")
    sub.add_parser("check")
    sub.add_parser("verification")
    args = parser.parse_args(argv)
    return {"index": cmd_index, "check": cmd_check, "verification": cmd_verification}[
        args.cmd
    ](args)


if __name__ == "__main__":
    sys.exit(main())
