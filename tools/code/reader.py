# tools/code/reader.py
from __future__ import annotations

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
