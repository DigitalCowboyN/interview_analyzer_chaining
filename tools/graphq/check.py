# tools/graphq/check.py
from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass
from typing import Dict, List

from tools.graphq.reader import QueryEntry, load_queries
from tools.graphq.render import render_catalog
from tools.glossary.reader import graph_vocabulary


@dataclass
class Finding:
    message: str


def check_schema_drift(entries: List[QueryEntry], vocab: Dict) -> List[Finding]:
    known = set(vocab)
    findings: List[Finding] = []
    for e in entries:
        for label in e.labels:
            if label not in known:
                findings.append(Finding(f"{e.bundle}:{e.name} references label :{label} not produced by any projection"))
        for rel in e.rels:
            if rel not in known:
                findings.append(Finding(f"{e.bundle}:{e.name} references rel [:{rel}] not produced by any projection"))
    return findings


def check_output_contract(entries: List[QueryEntry], root: str = ".") -> List[Finding]:
    findings: List[Finding] = []
    src = {os.path.relpath(f, root).replace(os.sep, "/"): open(f, encoding="utf-8", errors="ignore").read()
           for f in glob.glob(os.path.join(root, "src", "**", "*.py"), recursive=True)}
    for e in entries:
        if not e.returns:
            continue
        returned = set(e.returns)
        call = re.compile(rf"\b{re.escape(e.name)}\s*\(")
        access = re.compile(r"""(?:\[["']|\.get\(["'])(\w+)["']""")
        for rel, text in src.items():
            if rel.endswith("reader.py") or not call.search(text):
                continue
            for m in access.finditer(text):
                fld = m.group(1)
                # only flag fields that look like query outputs (appear in some query's returns)
                if fld not in returned and any(fld in x.returns for x in entries):
                    findings.append(Finding(f"{rel} reads field '{fld}' not returned by {e.bundle}:{e.name}"))
    return findings


def check_missing_marker(entries: List[QueryEntry]) -> List[Finding]:
    return [Finding(f"{e.bundle}:{e.name} has no graphq: marker (purpose/scope/audience)")
            for e in entries if not e.purpose]


def check_catalog_in_sync(catalog_path: str, entries: List[QueryEntry]) -> List[Finding]:
    want = render_catalog(entries)
    have = open(catalog_path, encoding="utf-8").read() if os.path.exists(catalog_path) else ""
    return [Finding("docs/graph-queries/index.md out of sync — run make graphq-index")] if want != have else []


def run_all(root: str = ".") -> List[Finding]:
    entries = load_queries(root)
    vocab = graph_vocabulary(root)
    findings: List[Finding] = []
    findings += check_schema_drift(entries, vocab)
    findings += check_output_contract(entries, root)
    findings += check_missing_marker(entries)
    findings += check_catalog_in_sync(os.path.join(root, "docs/graph-queries/index.md"), entries)
    return findings
