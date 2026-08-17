# tools/graphq/reader.py
"""Discovers Cypher-emitting functions under `src/**/reader.py` and `src/api/routers/*.py`
by walking their AST for string constants containing Cypher keywords, then parses each
query's labels/rels/props/returns and its `graphq:` docstring marker (purpose, scope,
audience) into a `QueryEntry`, deriving consumers by scanning for call sites."""

from __future__ import annotations

import ast
import glob
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

_LABEL = re.compile(r"[(\[]\s*\w*:([A-Z][A-Za-z]+)")
_REL = re.compile(r"\[\s*\w*:([A-Z_]{3,})")
_PROP = re.compile(r"\b[a-z_]\w*\.(\w+)")
_RETURN = re.compile(r"\bRETURN\b(.+?)(?:\bORDER\b|\bLIMIT\b|\bSKIP\b|$)", re.I | re.S)
_ALIAS = re.compile(r"\bAS\s+(\w+)")
_MARKER = re.compile(r"graphq:\s*(.+)")

READ_GLOBS = ("src/**/reader.py", "src/api/routers/*.py")


@dataclass
class QueryEntry:
    bundle: str
    name: str
    purpose: str = ""
    scope: str = ""
    audience: List[str] = field(default_factory=list)
    labels: List[str] = field(default_factory=list)
    rels: List[str] = field(default_factory=list)
    props: List[str] = field(default_factory=list)
    returns: List[str] = field(default_factory=list)
    consumers: List[str] = field(default_factory=list)

    @property
    def graph_id(self) -> str:
        # bundle-stem-qualified so queries sharing a function name across files stay distinct
        return f"{os.path.splitext(os.path.basename(self.bundle))[0]}.{self.name}"


def parse_cypher(text: str) -> Tuple[List[str], List[str], List[str], List[str]]:
    labels = sorted({m.group(1) for m in _LABEL.finditer(text) if not m.group(1).isupper()})
    rels = sorted({m.group(1) for m in _REL.finditer(text)})
    props = sorted({m.group(1) for m in _PROP.finditer(text)})
    returns: List[str] = []
    rm = _RETURN.search(text)
    if rm:
        returns = [a for a in _ALIAS.findall(rm.group(1))]
    return labels, rels, props, returns


def parse_graphq_marker(docstring: str) -> Dict:
    out: Dict = {}
    m = _MARKER.search(docstring or "")
    if not m:
        return out
    body = m.group(1)
    for key in ("purpose", "scope"):
        km = re.search(rf"{key}=([\w\-]+)", body)
        if km:
            out[key] = km.group(1)
    am = re.search(r"audience=\[([^\]]*)\]", body)
    if am:
        out["audience"] = [x.strip() for x in am.group(1).split(",") if x.strip()]
    return out


def _cypher_of(fn: ast.AST) -> str:
    parts = []
    for node in ast.walk(fn):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if re.search(r"\b(MATCH|MERGE|RETURN|CALL|CREATE)\b", node.value):
                parts.append(node.value)
    return "\n".join(parts)


def load_queries(root: str = ".", read_globs=READ_GLOBS) -> List[QueryEntry]:
    seen = set()
    files = []
    for g in read_globs:
        for f in glob.glob(os.path.join(root, g), recursive=True):
            if f not in seen:
                seen.add(f); files.append(f)
    entries: List[QueryEntry] = []
    for f in sorted(files):
        rel = os.path.relpath(f, root).replace(os.sep, "/")
        try:
            tree = ast.parse(open(f, encoding="utf-8").read())
        except Exception:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            cypher = _cypher_of(node)
            if not cypher:
                continue
            labels, rels, props, returns = parse_cypher(cypher)
            meta = parse_graphq_marker(ast.get_docstring(node) or "")
            entries.append(QueryEntry(
                bundle=rel, name=node.name,
                purpose=meta.get("purpose", ""), scope=meta.get("scope", ""),
                audience=meta.get("audience", []),
                labels=labels, rels=rels, props=props, returns=returns,
                consumers=derive_consumers(node.name, root),
            ))
    return entries


def derive_consumers(fn_name: str, root: str = ".") -> List[str]:
    roles = set()
    call = re.compile(rf"\b{re.escape(fn_name)}\s*\(")
    for f in glob.glob(os.path.join(root, "src", "**", "*.py"), recursive=True):
        rel = os.path.relpath(f, root).replace(os.sep, "/")
        if rel.endswith("reader.py"):
            continue  # skip the definition sites
        try:
            if call.search(open(f, encoding="utf-8", errors="ignore").read()):
                roles.add(rel.split("/")[1] if rel.startswith("src/") else rel)
        except Exception:
            continue
    return sorted(roles)
