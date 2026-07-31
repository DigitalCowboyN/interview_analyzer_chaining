from __future__ import annotations

import ast
import os
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class CodeTerm:
    name: str
    kind: str            # "enum" | "dimension"
    source: str          # repo-relative path
    values: List[str] = field(default_factory=list)


def _is_enum(classdef: ast.ClassDef) -> bool:
    for b in classdef.bases:
        name = b.id if isinstance(b, ast.Name) else getattr(b, "attr", "")
        if name == "Enum" or name.endswith("Enum"):
            return True
    return False


def code_enums(root: str = ".", subdirs=("src",)) -> Dict[str, CodeTerm]:
    out: Dict[str, CodeTerm] = {}
    for base in subdirs:
        start = os.path.join(root, base)
        if not os.path.isdir(start):
            continue
        for dirpath, _dirs, files in os.walk(start):
            for fn in files:
                if not fn.endswith(".py"):
                    continue
                full = os.path.join(dirpath, fn)
                try:
                    tree = ast.parse(open(full, encoding="utf-8").read())
                except Exception:
                    continue
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef) and _is_enum(node):
                        members = [
                            t.targets[0].id for t in node.body
                            if isinstance(t, ast.Assign) and len(t.targets) == 1
                            and isinstance(t.targets[0], ast.Name)
                        ]
                        rel = os.path.relpath(full, root).replace(os.sep, "/")
                        out[node.name] = CodeTerm(node.name, "enum", rel, members)
    return out


def code_dimensions(root: str = ".",
                    model_path: str = "src/models/analysis_result.py",
                    class_name: str = "AnalysisResult") -> Dict[str, CodeTerm]:
    out: Dict[str, CodeTerm] = {}
    p = os.path.join(root, model_path)
    if not os.path.exists(p):
        return out
    try:
        tree = ast.parse(open(p, encoding="utf-8").read())
    except Exception:
        return out
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for b in node.body:
                if isinstance(b, ast.AnnAssign) and isinstance(b.target, ast.Name):
                    out[b.target.id] = CodeTerm(b.target.id, "dimension", model_path, [])
    return out


def code_literals(root: str = ".", subdirs=("src",)) -> Dict[str, CodeTerm]:
    out: Dict[str, CodeTerm] = {}
    for base in subdirs:
        start = os.path.join(root, base)
        if not os.path.isdir(start):
            continue
        for dirpath, _dirs, files in os.walk(start):
            for fn in files:
                if not fn.endswith(".py"):
                    continue
                full = os.path.join(dirpath, fn)
                try:
                    tree = ast.parse(open(full, encoding="utf-8").read())
                except Exception:
                    continue
                rel = os.path.relpath(full, root).replace(os.sep, "/")
                for node in ast.walk(tree):
                    if not isinstance(node, ast.ClassDef):
                        continue
                    for b in node.body:
                        if isinstance(b, ast.AnnAssign) and isinstance(b.target, ast.Name):
                            ann = b.annotation
                            if isinstance(ann, ast.Subscript) and _name_of(ann.value) == "Literal":
                                vals = [e.value for e in _literal_elts(ann.slice)
                                        if isinstance(e, ast.Constant) and isinstance(e.value, str)]
                                if vals:
                                    out[f"{node.name}.{b.target.id}"] = CodeTerm(
                                        f"{node.name}.{b.target.id}", "literal", rel, vals)
    return out


def _name_of(n):
    return n.id if isinstance(n, ast.Name) else getattr(n, "attr", "")


def _literal_elts(sl):
    node = sl.value if isinstance(sl, ast.Index) else sl  # py<3.9 compat
    return node.elts if isinstance(node, ast.Tuple) else [node]
