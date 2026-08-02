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
]

_IMPORT = re.compile(r"(?:from|import)\s+src\.(\w+)")


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
    src = os.path.join(root, "src")
    out = []
    if os.path.isdir(src):
        for name in sorted(os.listdir(src)):
            p = os.path.join(src, name)
            if os.path.isdir(p) and name != "__pycache__":
                out.append(name)
    return out


def _files_of(unit: str, root: str) -> List[str]:
    # package -> all its .py; dotted module -> that one file
    if "." in unit:
        return [os.path.join(root, "src", *unit.split(".")) + ".py"]
    return glob.glob(os.path.join(root, "src", unit, "**", "*.py"), recursive=True)


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
                dep = m.group(1)
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


def dep_edges_for_module(unit: str, root: str) -> List[str]:
    if "." not in unit:
        return []
    valid = set(packages(root))  # only edges to documented packages (mirrors dep_edges)
    deps = set()
    for f in _files_of(unit, root):
        try:
            t = open(f, encoding="utf-8", errors="ignore").read()
        except Exception:
            continue
        pkg = unit.split(".")[0]
        for m in _IMPORT.finditer(t):
            if m.group(1) != pkg and m.group(1) in valid:
                deps.add(m.group(1))
    return sorted(deps)
