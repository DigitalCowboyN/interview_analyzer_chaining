# tools/infra/check.py
"""Non-blocking drift checks for the infra overlay: a code-service whose compose command resolves
to no code module (so `runs` silently drops it), and a `# talks-to:` marker naming an unknown
service. Returns List[Finding]; the CLI returns 0."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from tools.code.reader import load_units
from tools.infra.reader import _entrypoint_module, _marker_services, load_services


@dataclass
class Finding:
    message: str


def check_infra(root: str = ".") -> List[Finding]:
    findings: List[Finding] = []
    code_ids = {u.unit for u in load_units(root)}
    for s in load_services(root):
        if s.kind == "code" and s.command and _entrypoint_module(s.command, code_ids) is None:
            findings.append(Finding(
                f"infra: service {s.id} command {s.command!r} resolves to no code module "
                f"— `runs` will drop it (use exec-form src.* or a marker)"))
    service_ids = {s.id for s in load_services(root)}
    for u in load_units(root):
        if getattr(u, "level", "") != "module" or not str(u.path).endswith(".py"):
            continue
        try:
            src = open(u.path, encoding="utf-8", errors="ignore").read()
        except OSError:
            continue
        for svc in _marker_services(src):
            if svc not in service_ids:
                findings.append(Finding(
                    f"infra: {u.unit} has `# talks-to: {svc}` but no compose service named {svc}"))
    return findings


def run_all(root: str = ".") -> List[Finding]:
    return check_infra(root)
