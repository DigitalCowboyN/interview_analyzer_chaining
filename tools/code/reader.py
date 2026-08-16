# tools/code/reader.py
from __future__ import annotations

import ast
import glob
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List

from src.ingestion.front_matter import parse_front_matter

# curated load-bearing modules (dotted, relative to src/)
KEY_MODULES = [
    "ingestion.orchestrator", "ingestion.stitcher", "ingestion.speaker_inference",
    "enrichment.orchestrator", "enrichment.executor",
    "lens.engine", "export.reader", "export.renderer", "export.bundler",
    "ui.reader", "ask.reader", "ask.engine",
    "resolution.engine", "agents.agent_factory",
    # curated top-level src/*.py modules (resolved by _files_of to src/<name>.py)
    "config", "celery_app", "tasks", "main", "run_projection_service",
]

_IMPORT = re.compile(r"(?:from|import)\s+(src|tools)\.(\w+)")


def _dep_slug(match) -> str:
    tree, name = match.group(1), match.group(2)
    return name if tree == "src" else f"tools.{name}"


@dataclass
class CodeUnit:
    unit: str
    role: str = ""
    key_modules: List[str] = field(default_factory=list)
    depends_on: List[str] = field(default_factory=list)
    io: List[str] = field(default_factory=list)
    description: str = ""
    path: str = ""
    level: str = ""


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


def packages(root: str = ".") -> List[str]:
    out = []
    for tree, prefix in (("src", ""), ("tools", "tools.")):
        base = os.path.join(root, tree)
        if os.path.isdir(base):
            for name in sorted(os.listdir(base)):
                if os.path.isdir(os.path.join(base, name)) and name != "__pycache__":
                    out.append(f"{prefix}{name}")
    return out


def _files_of(unit: str, root: str) -> List[str]:
    # tools package -> all its .py; src package -> all its .py; src dotted module -> that one file
    if unit.startswith("tools."):
        pkg = unit.split(".", 1)[1]
        return glob.glob(os.path.join(root, "tools", pkg, "**", "*.py"), recursive=True)
    if "." in unit:
        return [os.path.join(root, "src", *unit.split(".")) + ".py"]
    # bare: a src package dir, else a top-level src module (src/<unit>.py)
    pkg_dir = os.path.join(root, "src", unit)
    if os.path.isdir(pkg_dir):
        return glob.glob(os.path.join(pkg_dir, "**", "*.py"), recursive=True)
    mod = os.path.join(root, "src", unit + ".py")
    return [mod] if os.path.exists(mod) else []


def dep_edges(root: str = ".") -> Dict[str, List[str]]:
    pkgs = packages(root)
    edges: Dict[str, set] = {p: set() for p in pkgs}
    for pkg in pkgs:
        for f in _files_of(pkg, root):
            try:
                text = open(f, encoding="utf-8", errors="ignore").read()
            except Exception:
                continue
            for m in _IMPORT.finditer(text):
                dep = _dep_slug(m)
                if dep != pkg and dep in edges:
                    edges[pkg].add(dep)
    return {p: sorted(s) for p, s in edges.items()}


def io_of(unit: str, root: str = ".") -> List[str]:
    io = set()
    for f in _files_of(unit, root):
        try:
            t = open(f, encoding="utf-8", errors="ignore").read()
        except Exception:
            continue
        if re.search(r"from src\.events|import src\.events|EventStore|esdb", t, re.I):
            io.add("ESDB")
        if re.search(r"neo4j|GraphDatabase|from src\.persistence", t, re.I):
            io.add("Neo4j")
        if re.search(r"from src\.agents|AgentFactory|openai|anthropic", t, re.I):
            io.add("LLM")
        if re.search(r"FastAPI|APIRouter|uvicorn", t):
            io.add("HTTP")
        if re.search(r"open\(|Path\(|\.read_text|glob\.", t):
            io.add("files")
    return sorted(io)


def load_units(root: str = ".", code_dir: str = "docs/code") -> List[CodeUnit]:
    edges = dep_edges(root)
    units: List[CodeUnit] = []
    for path in sorted(glob.glob(os.path.join(root, code_dir, "*.md"))):
        if os.path.basename(path) in ("index.md", "pipeline.md"):
            continue
        text = open(path, encoding="utf-8", errors="ignore").read()
        fm, offset = parse_front_matter(text)
        if not fm or "unit" not in fm:
            continue
        unit = str(fm["unit"])
        units.append(CodeUnit(
            unit=unit, role=str(fm.get("role", "")),
            key_modules=list(fm.get("key_modules") or []),
            depends_on=edges.get(unit, []) or dep_edges_for_module(unit, root),
            io=io_of(unit, root), description=text[offset:], path=path,
        ))
    return units


def _dep_targets(root: str) -> set:
    # valid depends_on targets: src packages + curated top-level src modules (config, etc.)
    return set(packages(root)) | {m for m in KEY_MODULES if "." not in m}


def dep_edges_for_module(unit: str, root: str) -> List[str]:
    if unit.startswith("tools."):
        return []
    # a bare unit that is a package is already covered by dep_edges(); only a top-level
    # module (src/<unit>.py) or a dotted key-module needs per-file derivation here.
    if "." not in unit and os.path.isdir(os.path.join(root, "src", unit)):
        return []
    targets = _dep_targets(root)
    parent = unit.split(".")[0]
    deps = set()
    for f in _files_of(unit, root):
        try:
            t = open(f, encoding="utf-8", errors="ignore").read()
        except Exception:
            continue
        for m in _IMPORT.finditer(t):
            dep = _dep_slug(m)
            if dep != parent and dep in targets:
                deps.add(dep)
    return sorted(deps)
