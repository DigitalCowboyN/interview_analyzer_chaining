from __future__ import annotations

from typing import List

from tools.cli.reader import Command


def render_help(commands: List[Command]) -> str:
    everyday = [c for c in commands if c.kind == "make" and c.visibility == "everyday"]
    lines = ["Usage: make [target]", ""]
    for c in sorted(everyday, key=lambda c: c.name):
        lines.append(f"  {c.name:24s} {c.description}")
    return "\n".join(lines) + "\n"


def render_catalog(commands: List[Command]) -> str:
    makes = sorted((c for c in commands if c.kind == "make"), key=lambda c: c.name)
    mods = sorted((c for c in commands if c.kind == "module"), key=lambda c: c.name)
    lines = ["# CLI surface", "", "## Make targets", "",
             "| command | visibility | description |", "| --- | --- | --- |"]
    for c in makes:
        lines.append(f"| {c.name} | {c.visibility} | {c.description} |")
    lines += ["", "## Module entry points", "", "| command | description |", "| --- | --- |"]
    for c in mods:
        lines.append(f"| {c.name} | {c.description} |")
    return "\n".join(lines) + "\n"
