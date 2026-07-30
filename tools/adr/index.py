from __future__ import annotations

import glob
import os
from typing import List

from tools.adr.model import Adr, parse_adr

RESERVED = {"index.md", "log.md"}


def load_bundle(adr_dir: str) -> List[Adr]:
    adrs: List[Adr] = []
    for path in sorted(glob.glob(os.path.join(adr_dir, "*.md"))):
        if os.path.basename(path) in RESERVED:
            continue
        with open(path, encoding="utf-8") as fh:
            adrs.append(parse_adr(fh.read(), path=path))
    return sorted(adrs, key=lambda a: a.id)


def render_index(adrs: List[Adr]) -> str:
    lines = ["# ADR Index", "", "| id | title | status |", "| --- | --- | --- |"]
    for a in adrs:
        lines.append(f"| {a.id:04d} | {a.title} | {a.status} |")
    return "\n".join(lines) + "\n"


def render_log(adrs: List[Adr]) -> str:
    lines = ["# Decision Log", ""]
    for a in sorted(adrs, key=lambda a: (a.date, a.id)):
        sup = ""
        if a.supersedes:
            sup = " (supersedes " + ", ".join(f"{i:04d}" for i in a.supersedes) + ")"
        lines.append(f"- {a.date} — **{a.id:04d}** {a.title} · _{a.status}_{sup}")
    return "\n".join(lines) + "\n"


def write_generated(adr_dir: str) -> None:
    adrs = load_bundle(adr_dir)
    with open(os.path.join(adr_dir, "index.md"), "w", encoding="utf-8") as fh:
        fh.write(render_index(adrs))
    with open(os.path.join(adr_dir, "log.md"), "w", encoding="utf-8") as fh:
        fh.write(render_log(adrs))
