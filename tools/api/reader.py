from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

_METHODS = ("GET", "POST", "PUT", "DELETE", "PATCH")


@dataclass
class Endpoint:
    method: str
    path: str
    router: str
    summary: str


def load_app():
    import logging
    os.environ.setdefault("ANTHROPIC_API_KEY", "api-surface-placeholder")
    os.environ.setdefault("OPENAI_API_KEY", "api-surface-placeholder")
    logging.disable(logging.CRITICAL)
    from src.main import app
    return app


def live_endpoints(app) -> List[Endpoint]:
    from fastapi.routing import APIRoute
    eps: List[Endpoint] = []
    for r in app.routes:
        if isinstance(r, APIRoute) and r.endpoint.__module__.startswith("src"):
            for m in sorted(r.methods):
                if m in _METHODS:
                    eps.append(Endpoint(m, r.path, r.endpoint.__module__, r.summary or r.name or ""))
    return sorted(eps, key=lambda e: (e.router, e.path, e.method))


def live_openapi_pairs(app) -> Set[Tuple[str, str]]:
    return {(m.upper(), p) for p, ops in app.openapi()["paths"].items() for m in ops}


def committed_pairs(openapi_path: str) -> Optional[Set[Tuple[str, str]]]:
    if not os.path.exists(openapi_path):
        return None
    data = json.load(open(openapi_path, encoding="utf-8"))
    return {(m.upper(), p) for p, ops in data.get("paths", {}).items() for m in ops}
