from __future__ import annotations

import argparse
import sys

from tools.corpus.check import run_all
from tools.corpus.reader import okf_records


def cmd_check(args) -> int:
    findings = run_all()
    if findings:
        print(f"corpus-check: {len(findings)} warning(s):")
        for f in findings:
            print(f"  - {f.message}")
    else:
        print("corpus-check: clean")
    return 0  # NON-BLOCKING


def cmd_list(args) -> int:
    for r in okf_records():
        print(f"{r.type}\t{r.id}\t{r.path}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="tools.corpus")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    sub.add_parser("list")
    args = p.parse_args(argv)
    return {"check": cmd_check, "list": cmd_list}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
