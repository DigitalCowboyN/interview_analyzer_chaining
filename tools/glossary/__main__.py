"""CLI entry point for the glossary domain (`python -m tools.glossary`): regenerate
docs/glossary/index.md, run the non-blocking coverage/consistency check, and scaffold
new enum/dimension term files.
"""
from __future__ import annotations

import argparse
import os
import sys

from tools.glossary.check import run_all
from tools.glossary.model import load_glossary
from tools.glossary.render import render_index
from tools.glossary.scaffold import new_term

GLOSSARY = "docs/glossary"


def cmd_index(args) -> int:
    os.makedirs(GLOSSARY, exist_ok=True)
    with open(os.path.join(GLOSSARY, "index.md"), "w", encoding="utf-8") as fh:
        fh.write(render_index(load_glossary(GLOSSARY)))
    print(f"wrote {GLOSSARY}/index.md")
    return 0


def cmd_check(args) -> int:
    findings = run_all()
    if findings:
        print(f"glossary-check: {len(findings)} warning(s):")
        for f in findings:
            print(f"  - {f.message}")
    else:
        print("glossary-check: clean")
    return 0  # NON-BLOCKING


def cmd_scaffold(args) -> int:
    path = new_term(args.name, args.kind)
    print(f"created {path} — fill in the definition" + (" and values" if args.kind == "dimension" else ""))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="tools.glossary")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("index")
    sub.add_parser("check")
    p_s = sub.add_parser("scaffold"); p_s.add_argument("name"); p_s.add_argument("kind", choices=["enum", "dimension"])
    args = parser.parse_args(argv)
    return {"index": cmd_index, "check": cmd_check, "scaffold": cmd_scaffold}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
