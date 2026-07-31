from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List

from tools.cli.reader import Command, module_entrypoints, parse_makefile
from tools.cli.render import render_catalog

# command mentions are only counted inside inline code spans (leading backtick),
# so prose like "make sure" is not a false positive.
_MAKE_MENTION = re.compile(r"`make\s+([a-zA-Z][\w-]*)")
_MODULE_MENTION = re.compile(r"`python\s+-m\s+([\w.]+)")


@dataclass
class Finding:
    message: str


def _module_resolves(root: str, dotted: str) -> bool:
    # `python -m a.b.c` is valid if a/b/c.py, a/b/c/__main__.py, or the package exists.
    base = os.path.join(root, *dotted.split("."))
    return os.path.isfile(base + ".py") or os.path.isdir(base)


def check_docs_reference_real(commands: List[Command], doc_paths: List[str],
                              root: str = ".") -> List[Finding]:
    real_make = {c.name for c in commands if c.kind == "make"}
    findings: List[Finding] = []
    for dp in doc_paths:
        if not os.path.exists(dp):
            continue
        text = open(dp, encoding="utf-8").read()
        base = os.path.basename(dp)
        for m in _MAKE_MENTION.finditer(text):
            if m.group(1) not in real_make:
                findings.append(Finding(f"{base} references `make {m.group(1)}` which is not a real target"))
        for m in _MODULE_MENTION.finditer(text):
            if not _module_resolves(root, m.group(1)):
                findings.append(Finding(f"{base} references `python -m {m.group(1)}` which does not resolve to a module"))
    return findings


def check_catalog_in_sync(catalog_path: str, commands: List[Command]) -> List[Finding]:
    want = render_catalog(commands)
    have = open(catalog_path, encoding="utf-8").read() if os.path.exists(catalog_path) else ""
    if want != have:
        return [Finding("docs/cli/index.md out of sync — run `make cli-index`")]
    return []


def check_undocumented(commands: List[Command]) -> List[Finding]:
    return [Finding(f"make target `{c.name}` has no ## description")
            for c in commands if c.kind == "make" and c.visibility == "undocumented"]


def run_all(root: str = ".", docs=("CLAUDE.md", "README.md"),
            catalog: str = "docs/cli/index.md") -> List[Finding]:
    commands = parse_makefile(os.path.join(root, "Makefile")) + module_entrypoints(root)
    findings: List[Finding] = []
    findings += check_docs_reference_real(commands, [os.path.join(root, d) for d in docs], root)
    findings += check_catalog_in_sync(os.path.join(root, catalog), commands)
    findings += check_undocumented(commands)
    return findings
