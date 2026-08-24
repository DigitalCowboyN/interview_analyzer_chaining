# tools/infra/reader.py
"""KG-3 infra overlay: Service/EnvVar node data and the four infra edge-pair sets, all derived
from docker-compose.yml (+ a client-lib map and `# talks-to:` markers). Consumed by
tools.graph.reader via _ADAPTERS + _DERIVED. Must not import tools.graph.reader (layering)."""
# governed-by: ADR-0029
from __future__ import annotations

import ast
import os
import re
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple

import yaml

from tools.code.reader import load_units

COMPOSE = "docker-compose.yml"

# backing-service client library (top-level import) -> the compose service it connects to
_CLIENT_LIBS = {"neo4j": "neo4j", "esdbclient": "eventstore", "celery": "redis"}
_TALKS_MARKER = re.compile(r"#\s*talks-to:\s*([\w-]+)")
_SRC_TOKEN = re.compile(r"^src\.([\w.]+?)(?::\w+)?$")   # "src.main:app" -> "main"


@dataclass
class Service:
    id: str
    kind: str                                  # "code" | "backing"
    image: str = ""
    command: List[str] = field(default_factory=list)
    ports: List[str] = field(default_factory=list)
    requires: List[str] = field(default_factory=list)   # service names (compose depends_on)
    env: List[str] = field(default_factory=list)         # inline environment var names
    loads_env_file: bool = False


@dataclass
class EnvVar:
    name: str


def _compose(root: str) -> dict:
    with open(os.path.join(root, COMPOSE), encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _depends(spec: dict) -> List[str]:
    d = spec.get("depends_on")
    if isinstance(d, dict):
        return list(d)
    return list(d or [])


def _env_names(spec: dict) -> List[str]:
    e = spec.get("environment")
    if isinstance(e, dict):
        return list(e)
    return [str(x).split("=", 1)[0] for x in (e or [])]


def load_services(root: str = ".") -> List[Service]:
    out: List[Service] = []
    for name, spec in (_compose(root).get("services") or {}).items():
        spec = spec or {}
        out.append(Service(
            id=name,
            kind="code" if "build" in spec else "backing",
            image=spec.get("image", "") or "",
            command=list(spec.get("command") or []),
            ports=[str(p) for p in (spec.get("ports") or [])],
            requires=_depends(spec),
            env=_env_names(spec),
            loads_env_file=bool(spec.get("env_file")),
        ))
    return out


def load_env_vars(root: str = ".") -> List[EnvVar]:
    names = sorted({v for s in load_services(root) for v in s.env})
    return [EnvVar(n) for n in names]


def requires_pairs(root: str = ".") -> List[Tuple[str, str]]:
    ids = {s.id for s in load_services(root)}
    return [(s.id, dep) for s in load_services(root) for dep in s.requires if dep in ids]


def configured_by_pairs(root: str = ".") -> List[Tuple[str, str]]:
    return [(s.id, var) for s in load_services(root) for var in s.env]


def _entrypoint_module(command: List[str], code_ids: Set[str]) -> Optional[str]:
    for tok in command:
        m = _SRC_TOKEN.match(str(tok))
        if m and m.group(1) in code_ids:
            return m.group(1)
    return None


def runs_pairs(root: str = ".") -> List[Tuple[str, str]]:
    code_ids = {u.unit for u in load_units(root)}
    out: List[Tuple[str, str]] = []
    for s in load_services(root):
        mod = _entrypoint_module(s.command, code_ids)
        if mod:
            out.append((s.id, mod))
    return out


def _module_imports(src: str) -> Set[str]:
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return set()
    libs: Set[str] = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                libs.add(a.name.split(".")[0])
        elif isinstance(n, ast.ImportFrom) and n.module:
            libs.add(n.module.split(".")[0])
    return libs


def talks_to_pairs(root: str = ".") -> List[Tuple[str, str]]:
    service_ids = {s.id for s in load_services(root)}
    out: List[Tuple[str, str]] = []
    for u in load_units(root):
        if getattr(u, "level", "") != "module" or not str(u.path).endswith(".py"):
            continue
        try:
            src = open(u.path, encoding="utf-8", errors="ignore").read()
        except OSError:
            continue
        targets: Set[str] = set()
        for lib in _module_imports(src):                       # derived: client-lib import
            svc = _CLIENT_LIBS.get(lib)
            if svc in service_ids:
                targets.add(svc)
        for svc in _TALKS_MARKER.findall(src):                 # marker fallback
            if svc in service_ids:
                targets.add(svc)
        out += [(u.unit, svc) for svc in sorted(targets)]
    return out
