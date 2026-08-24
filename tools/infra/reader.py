# tools/infra/reader.py
"""KG-3 infra overlay: Service/EnvVar node data and the four infra edge-pair sets, all derived
from docker-compose.yml (+ a client-lib map and `# talks-to:` markers). Consumed by
tools.graph.reader via _ADAPTERS + _DERIVED. Must not import tools.graph.reader (layering)."""
# governed-by: ADR-0029
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import List

import yaml

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
