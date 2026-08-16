# tools/code/reader.py
from __future__ import annotations

import ast
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class CodeUnit:
    unit: str
    level: str = ""                                   # "package" | "module"
    depends_on: List[str] = field(default_factory=list)
    description: str = ""                             # docstring
    path: str = ""


_SRC_TREES = (("src", ""), ("tools", "tools."))
_IMPORT_DOTTED = re.compile(r"^\s*(?:from|import)\s+((?:src|tools)\.[\w.]+)", re.M)


def _docstring(path: str) -> str:
    try:
        tree = ast.parse(open(path, encoding="utf-8", errors="ignore").read())
    except (OSError, SyntaxError):
        return ""
    return (ast.get_docstring(tree) or "").strip()


def _dotted(prefix: str, parts: List[str]) -> str:
    return prefix + ".".join(parts)


def _longest_node_prefix(cand: str, ids: set) -> str:
    parts = cand.split(".")
    for i in range(len(parts), 0, -1):
        pref = ".".join(parts[:i])
        if pref in ids:
            return pref
    return ""


def _module_deps(path: str, self_id: str, ids: set) -> List[str]:
    try:
        text = open(path, encoding="utf-8", errors="ignore").read()
    except OSError:
        return []
    deps = set()
    for m in _IMPORT_DOTTED.finditer(text):
        dotted = m.group(1)
        cand = dotted[len("src."):] if dotted.startswith("src.") else dotted  # keep 'tools.'
        dep = _longest_node_prefix(cand, ids)
        # skip self and own ancestors — the parent chain is the `contains` edge, not a dependency
        if dep and dep != self_id and not self_id.startswith(dep + "."):
            deps.add(dep)
    return sorted(deps)


def discover_units(root: str = ".") -> List[CodeUnit]:
    """Derive package + module CodeUnits from source (src/, tools/).

    A directory that directly contains a .py file is a package; every non-__init__ .py is a
    module. Ids are dotted paths (src/ stripped, tools. kept). Context = the docstring."""
    units: Dict[str, CodeUnit] = {}
    for tree, prefix in _SRC_TREES:
        base = os.path.join(root, tree)
        if not os.path.isdir(base):
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
            pyfiles = sorted(f for f in filenames if f.endswith(".py"))
            if not pyfiles:
                continue
            rel = os.path.relpath(dirpath, base)
            dir_parts = [] if rel == "." else rel.split(os.sep)
            if dir_parts:                                   # package node (root itself is not one)
                pid = _dotted(prefix, dir_parts)
                init = os.path.join(dirpath, "__init__.py")
                units[pid] = CodeUnit(
                    unit=pid, level="package",
                    description=_docstring(init) if os.path.exists(init) else "",
                    path=dirpath + os.sep)
            for f in pyfiles:                               # module nodes
                if f == "__init__.py":
                    continue
                mpath = os.path.join(dirpath, f)
                mid = _dotted(prefix, dir_parts + [f[:-3]])
                units[mid] = CodeUnit(
                    unit=mid, level="module",
                    description=_docstring(mpath), path=mpath)
    ids = set(units)
    for u in units.values():
        if u.level == "module":
            u.depends_on = _module_deps(u.path, u.unit, ids)
    return [units[k] for k in sorted(units)]


def contains_edges(root: str = ".") -> List["tuple"]:
    """(parent, child) hierarchy pairs — a package contains its sub-packages and modules."""
    ids = {u.unit for u in discover_units(root)}
    out = []
    for uid in sorted(ids):
        parent = uid.rsplit(".", 1)[0] if "." in uid else ""
        if parent and parent in ids:
            out.append((parent, uid))
    return out


def load_units(root: str = ".") -> List[CodeUnit]:
    """The code node registry — derived from source (packages + modules)."""
    return discover_units(root)


def dep_edges(root: str = ".") -> Dict[str, List[str]]:
    """unit id -> module-granular depends_on targets (modules only carry deps)."""
    return {u.unit: u.depends_on for u in load_units(root) if u.depends_on}
