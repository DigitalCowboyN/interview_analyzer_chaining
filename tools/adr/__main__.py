"""CLI entry point for the ADR domain (`python -m tools.adr <cmd>`): `index` regenerates
the generated files, `check` runs the non-blocking drift checks, `new` scaffolds an ADR,
`context` is the UserPromptSubmit hook nudging toward the ADR index on architectural
prompts, and `where` looks up which ADR governs a given path."""
from __future__ import annotations

import argparse
import json
import sys

from tools.adr.check import run_all
from tools.adr.index import write_generated
from tools.adr.intent import is_architectural
from tools.adr.scaffold import new_adr

DEFAULT_ADR_DIR = "docs/adr"
DEFAULT_SPECS_DIR = "docs/superpowers/specs"


def _read_stdin_json() -> dict:
    try:
        return json.loads(sys.stdin.read() or "{}")
    except Exception:
        return {}


def cmd_index(args) -> int:
    write_generated(args.adr_dir)
    print(f"regenerated {args.adr_dir}/index.md, log.md, by-code.md")
    return 0


def cmd_check(args) -> int:
    findings = run_all(args.adr_dir, args.specs_dir)
    if findings:
        print(f"adr-check: {len(findings)} warning(s):")
        for f in findings:
            print(f"  - {f.message}")
    else:
        print("adr-check: clean")
    return 0  # NON-BLOCKING: always 0


def cmd_new(args) -> int:
    path = new_adr(args.adr_dir, args.title)
    print(f"created {path} — fill it in, set `source:` and `date:`, then `make adr-index`")
    return 0


def cmd_context(args) -> int:
    # UserPromptSubmit hook: a PROVISIONAL, lean before-signal. Emits a pointer (not
    # the full ADR table) only on architectural prompts. Retire once the cascade
    # reliably gets ADRs consulted without it (see the 2026-08-05 knowledge-cascade
    # spec's retirement criterion).
    prompt = _read_stdin_json().get("prompt", "")
    if is_architectural(prompt):
        print("Locking an architectural decision? Consult docs/adr/index.md before "
              "you do (and docs/index.md for the wider knowledge map). "
              "Supersede rather than silently override.")
    return 0


def cmd_where(args) -> int:
    from tools.adr.index import load_bundle
    from tools.adr.check import _path_covered_by
    adrs = load_bundle(args.adr_dir)
    hits = [a for a in adrs if _path_covered_by(args.path, a.governs)]
    if hits:
        for a in hits:
            print(f"ADR-{a.id:04d} {a.title}")
    else:
        print(f"no ADR governs {args.path}")
    return 0


def main(argv=None) -> int:
    # Shared options live on a parent parser so they are valid AFTER the
    # subcommand (e.g. `tools.adr check --adr-dir X --specs-dir Y`).
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--adr-dir", default=DEFAULT_ADR_DIR)
    common.add_argument("--specs-dir", default=DEFAULT_SPECS_DIR)

    parser = argparse.ArgumentParser(prog="tools.adr")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("index", parents=[common])
    sub.add_parser("check", parents=[common])
    p_new = sub.add_parser("new", parents=[common])
    p_new.add_argument("title")
    sub.add_parser("context", parents=[common])
    p_where = sub.add_parser("where", parents=[common])
    p_where.add_argument("path")
    args = parser.parse_args(argv)
    return {
        "index": cmd_index, "check": cmd_check, "new": cmd_new,
        "context": cmd_context, "where": cmd_where,
    }[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
