# tools/graphq/check.py
from __future__ import annotations

import ast
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
    by_name = {e.name: e for e in entries}

    def _query_of_call(node):
        if isinstance(node, ast.Call):
            fname = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
            return by_name.get(fname)
        return None

    def _query_in(expr):
        for n in ast.walk(expr):
            q = _query_of_call(n)
            if q:
                return q
        return None

    for f in glob.glob(os.path.join(root, "src", "**", "*.py"), recursive=True):
        rel = os.path.relpath(f, root).replace(os.sep, "/")
        if rel.endswith("reader.py"):
            continue
        try:
            tree = ast.parse(open(f, encoding="utf-8", errors="ignore").read())
        except Exception:
            continue
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            var_query = {}  # varname -> QueryEntry
            for node in ast.walk(fn):
                if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                    q = _query_in(node.value)
                    if q:
                        var_query[node.targets[0].id] = q
                if isinstance(node, ast.For) and isinstance(node.target, ast.Name):
                    q = _query_in(node.iter)
                    if q:
                        var_query[node.target.id] = q
            # one hop: `for row in <var-bound-to-a-query>`
            for node in ast.walk(fn):
                if (isinstance(node, ast.For) and isinstance(node.target, ast.Name)
                        and isinstance(node.iter, ast.Name) and node.iter.id in var_query):
                    var_query.setdefault(node.target.id, var_query[node.iter.id])
            if not var_query:
                continue
            for node in ast.walk(fn):
                var = field = None
                if (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
                        and isinstance(node.ctx, ast.Load)):
                    key = node.slice
                    if isinstance(key, ast.Constant) and isinstance(key.value, str):
                        var, field = node.value.id, key.value
                elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                      and node.func.attr == "get" and isinstance(node.func.value, ast.Name)
                      and node.args and isinstance(node.args[0], ast.Constant)
                      and isinstance(node.args[0].value, str)):
                    var, field = node.func.value.id, node.args[0].value
                if var in var_query and field and field not in var_query[var].returns:
                    q = var_query[var]
                    findings.append(Finding(f"{rel} reads field '{field}' not returned by {q.bundle}:{q.name}"))
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
