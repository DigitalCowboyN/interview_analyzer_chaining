# tools/code/reader.py
"""Derives the code node registry — `CodeUnit` packages and modules — directly from source
under `src/` and `tools/`: a directory holding `.py` files is a package, each non-`__init__`
file a module, its docstring becomes the description, and its `from`/`import` statements
resolve to `depends_on` edges. The heart of the derived code graph; `load_units` is the
registry every other `tools.*` domain reads."""

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
_CALLS_MARKER = re.compile(r"#\s*calls:\s*code:([\w.]+)")


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


@dataclass
class Symbol:
    id: str                                  # dotted, e.g. "api.main.Router.add"
    kind: str                                # function | class | method
    signature: str = ""
    docstring: str = ""
    parent: str = ""                         # module id, or the class id for a method
    calls: List[str] = field(default_factory=list)   # filled by Task 3


def render_signature(node) -> str:
    a = node.args
    parts = [arg.arg for arg in a.args]
    if a.vararg:
        parts.append("*" + a.vararg.arg)
    if a.kwarg:
        parts.append("**" + a.kwarg.arg)
    # attach simple defaults to the trailing positional args
    defaults = list(a.defaults)
    if defaults:
        base = len(a.args) - len(defaults)
        for i, d in enumerate(defaults):
            try:
                parts[base + i] = f"{a.args[base + i].arg}={ast.unparse(d)}"
            except Exception:
                pass
    ret = f" -> {ast.unparse(node.returns)}" if getattr(node, "returns", None) else ""
    return f"{node.name}({', '.join(parts)}){ret}"


def _module_path(module_id: str, root: str) -> str:
    if module_id.startswith("tools."):
        return os.path.join(root, "tools", *module_id.split(".")[1:]) + ".py"
    return os.path.join(root, "src", *module_id.split(".")) + ".py"


def _name_index(tree, module_id: str) -> Dict[str, str]:
    """local name -> target symbol/module id, from imports (absolute + relative) + top-level defs."""
    idx: Dict[str, str] = {}
    pkg = module_id.split(".")[:-1]                 # this module's package parts
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level and node.level > 0:
            # relative import: `from . / .. import x` — resolve against this module's package
            base = pkg[: len(pkg) - (node.level - 1)]
            prefix = ".".join(base + ([node.module] if node.module else []))
            if prefix:
                for alias in node.names:
                    idx[alias.asname or alias.name] = f"{prefix}.{alias.name}"
        elif isinstance(node, ast.ImportFrom) and node.module and \
                (node.module.startswith("src.") or node.module.startswith("tools.")):
            base = node.module[4:] if node.module.startswith("src.") else node.module
            for alias in node.names:
                idx[alias.asname or alias.name] = f"{base}.{alias.name}"
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(("src.", "tools.")):
                    tgt = alias.name[4:] if alias.name.startswith("src.") else alias.name
                    idx[alias.asname or alias.name] = tgt
    for node in tree.body:                       # local top-level defs
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            idx[node.name] = f"{module_id}.{node.name}"
    return idx


def calls_of(func_node, name_index: Dict[str, str], marker_text: str = "") -> List[str]:
    """Resolve a function/method body's calls via the module name index, plus `# calls:` markers.

    Ceiling, not floor: `foo()` and `mod.foo()` resolve when `foo`/`mod` are in the name index
    (imports + local top-level defs); a call on an unrecognized name (e.g. `obj.method()`) is
    skipped rather than guessed."""
    out = set(_CALLS_MARKER.findall(marker_text))         # explicit markers
    for n in ast.walk(func_node):
        if isinstance(n, ast.Call):
            f = n.func
            if isinstance(f, ast.Name) and f.id in name_index:          # foo()
                out.add(name_index[f.id])
            elif isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) \
                    and f.value.id in name_index:                       # mod.foo()
                out.add(f"{name_index[f.value.id]}.{f.attr}")
            # obj.method() on an unknown Name -> not in name_index -> skipped (ceiling)
    return sorted(out)


def symbols_of(module_id: str, root: str = ".") -> List[Symbol]:
    """Top-level functions/classes of a module, and a class's methods (one level of nesting).

    Each function/method Symbol's `.calls` is resolved pragmatically from this file's own
    imports + local defs (see `_name_index`/`calls_of`) — a same-file ceiling, not a full
    cross-repo call graph."""
    path = _module_path(module_id, root)
    try:
        source = open(path, encoding="utf-8", errors="ignore").read()
        tree = ast.parse(source)
    except (OSError, SyntaxError):
        return []
    nidx = _name_index(tree, module_id)

    def _marker(n):                                   # a `# calls:` marker applies to its own def
        return ast.get_source_segment(source, n) or ""

    out: List[Symbol] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            sym = Symbol(f"{module_id}.{node.name}", "function",
                         render_signature(node), (ast.get_docstring(node) or "").strip(),
                         module_id)
            sym.calls = calls_of(node, nidx, marker_text=_marker(node))
            out.append(sym)
        elif isinstance(node, ast.ClassDef):
            cid = f"{module_id}.{node.name}"
            out.append(Symbol(cid, "class", f"class {node.name}",
                              (ast.get_docstring(node) or "").strip(), module_id))
            for m in node.body:
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    msym = Symbol(f"{cid}.{m.name}", "method",
                                  render_signature(m), (ast.get_docstring(m) or "").strip(), cid)
                    msym.calls = calls_of(m, nidx, marker_text=_marker(m))
                    out.append(msym)
    return out
