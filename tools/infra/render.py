# tools/infra/render.py
"""Renders docs/infra/index.md — the deployment topology catalog: services by kind, the requires
DAG, each code service's entrypoint + talks_to, and the inline env vars. Pure function of the
compose-derived data (freshness-gated)."""
from __future__ import annotations

from typing import List, Tuple

from tools.infra.reader import EnvVar, Service


def render_index(services: List[Service], env_vars: List[EnvVar],
                 runs: List[Tuple[str, str]], talks_to: List[Tuple[str, str]],
                 requires: List[Tuple[str, str]], configured_by: List[Tuple[str, str]]) -> str:
    lines = ["# Infra / deployment topology",
             "",
             "> Generated from `docker-compose.yml` by `make infra-index`. Do not edit by hand.",
             "",
             "## Services", ""]
    for s in sorted(services, key=lambda x: (x.kind, x.id)):
        detail = f"image `{s.image}`" if s.kind == "backing" else f"runs `{' '.join(s.command)}`"
        reqs = ", ".join(sorted(d for a, d in requires if a == s.id)) or "—"
        lines.append(f"- **{s.id}** ({s.kind}) — {detail}; requires: {reqs}")
    lines += ["", "## Code → backing service (`talks_to`)", ""]
    for code, svc in sorted(talks_to):
        lines.append(f"- `{code}` → **{svc}**")
    lines += ["", "## Environment variables (inline; `.env` excluded)", ""]
    for v in sorted(env_vars, key=lambda x: x.name):
        svcs = ", ".join(sorted(s for s, name in configured_by if name == v.name))
        lines.append(f"- `{v.name}` — {svcs}")
    return "\n".join(lines) + "\n"
