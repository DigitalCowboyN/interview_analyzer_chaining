from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

from tools.api.reader import (
    Endpoint, committed_pairs, live_endpoints, live_openapi_pairs, load_app,
)
from tools.api.render import render_catalog

OPENAPI = "frontend/openapi.json"
_ENDPOINT_MENTION = re.compile(r"`(GET|POST|PUT|DELETE|PATCH)\s+(/[\w{}./-]*)`")


@dataclass
class Finding:
    message: str


def _shape(path: str) -> str:
    return re.sub(r"\{[^}]+\}", "{}", path)


def check_openapi_fresh(committed: Optional[Set[Tuple[str, str]]],
                        live: Set[Tuple[str, str]],
                        openapi_path: str = OPENAPI) -> List[Finding]:
    if committed is None:
        return [Finding(f"{openapi_path} is missing — run make ui-typegen")]
    findings: List[Finding] = []
    for m, p in sorted(live - committed):
        findings.append(Finding(f"{m} {p} exists in the app but not in {openapi_path} — run make ui-typegen"))
    for m, p in sorted(committed - live):
        findings.append(Finding(f"{m} {p} is in {openapi_path} but not in the app — run make ui-typegen"))
    return findings


def check_docs_reference_real(endpoints: List[Endpoint], doc_paths: List[str]) -> List[Finding]:
    real = {(e.method, _shape(e.path)) for e in endpoints}
    findings: List[Finding] = []
    for dp in doc_paths:
        if not os.path.exists(dp):
            continue
        text = open(dp, encoding="utf-8").read()
        base = os.path.basename(dp)
        for mo in _ENDPOINT_MENTION.finditer(text):
            method, path = mo.group(1), mo.group(2)
            if (method, _shape(path)) not in real:
                findings.append(Finding(f"{base} references `{method} {path}` which is not a real endpoint"))
    return findings


def check_catalog_in_sync(catalog_path: str, endpoints: List[Endpoint]) -> List[Finding]:
    want = render_catalog(endpoints)
    have = open(catalog_path, encoding="utf-8").read() if os.path.exists(catalog_path) else ""
    if want != have:
        return [Finding("docs/api/index.md out of sync — run make api-index")]
    return []


def run_all(root: str = ".") -> List[Finding]:
    try:
        app = load_app()
    except Exception as e:  # non-blocking: degrade to a warning
        return [Finding(f"could not import src.main:app ({e}) — api-check skipped")]
    endpoints = live_endpoints(app)
    findings: List[Finding] = []
    findings += check_openapi_fresh(committed_pairs(os.path.join(root, OPENAPI)), live_openapi_pairs(app),
                                    os.path.join(root, OPENAPI))
    findings += check_catalog_in_sync(os.path.join(root, "docs/api/index.md"), endpoints)
    findings += check_docs_reference_real(endpoints, [os.path.join(root, d) for d in ("CLAUDE.md", "README.md")])
    return findings
