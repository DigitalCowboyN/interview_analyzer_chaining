"""Glossary domain guards: compare code-derived enums, dimensions, and graph
vocabulary against docs/glossary/*.md terms to flag missing coverage, value drift,
stale entries, and an out-of-sync index. Non-blocking — findings are warnings, not
build failures.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List

from tools.glossary.reader import (
    CodeTerm, code_dimensions, code_enums, code_literals, graph_vocabulary,
)
from tools.glossary.model import Term, load_glossary
from tools.glossary.render import render_index


@dataclass
class Finding:
    message: str


def check_coverage(code: Dict[str, CodeTerm], terms: List[Term]) -> List[Finding]:
    have = {t.term for t in terms}
    findings: List[Finding] = []
    for name in sorted(code):
        if name not in have:
            ct = code[name]
            findings.append(Finding(f"code defines {ct.kind} {name} ({ct.source}) with no glossary term"))
    return findings


def check_enum_values(code: Dict[str, CodeTerm], terms: List[Term]) -> List[Finding]:
    findings: List[Finding] = []
    for t in terms:
        key = getattr(t, "code_symbol", None) or (t.term if t.kind == "enum" else None)
        if key is None:
            continue
        ct = code.get(key)
        if ct is None:
            continue
        if set(t.values) != set(ct.values):
            missing = sorted(set(ct.values) - set(t.values))
            extra = sorted(set(t.values) - set(ct.values))
            findings.append(Finding(f"glossary term {t.term} values differ from code (missing: {missing}, extra: {extra})"))
    return findings


def check_stale_source(code: Dict[str, CodeTerm], terms: List[Term]) -> List[Finding]:
    findings: List[Finding] = []
    for t in terms:
        symbol = getattr(t, "code_symbol", None)
        if symbol:
            if symbol not in code:
                findings.append(Finding(f"glossary term {t.term}: code_symbol {symbol} no longer defined in code"))
        elif t.kind in ("enum", "dimension", "graph-label", "rel-type", "graph-property"):
            if t.term not in code:
                findings.append(Finding(f"glossary term {t.term}: no longer defined in code (source {t.source})"))
        # registry-pinned kinds (entity-type, etc.) are not code-backed -> skip
    return findings


def check_index_in_sync(index_path: str, terms: List[Term]) -> List[Finding]:
    want = render_index(terms)
    have = open(index_path, encoding="utf-8").read() if os.path.exists(index_path) else ""
    if want != have:
        return [Finding("docs/glossary/index.md out of sync — run make glossary-index")]
    return []


def run_all(root: str = ".") -> List[Finding]:
    enums = code_enums(root)
    dims = code_dimensions(root)
    lits = code_literals(root)
    gv = graph_vocabulary(root)
    terms = load_glossary(os.path.join(root, "docs/glossary"))
    findings: List[Finding] = []
    findings += check_coverage({**enums, **dims, **gv}, terms)    # NOT lits
    findings += check_enum_values({**enums, **dims, **lits}, terms)
    findings += check_stale_source({**enums, **dims, **lits, **gv}, terms)
    findings += check_index_in_sync(os.path.join(root, "docs/glossary/index.md"), terms)
    return findings
