"""Renders loaded `PromptEntry` records into `docs/prompts/index.md`, one table per
source YAML file listing each prompt's classification, metadata, and enumerated values."""

from __future__ import annotations

from typing import List

from tools.prompts.reader import PromptEntry


def render_catalog(entries: List[PromptEntry]) -> str:
    by_file: dict = {}
    for e in entries:
        by_file.setdefault(e.file, []).append(e)
    lines = ["# Prompt registry (probabilistic components)", ""]
    for file in sorted(by_file):
        lines.append(f"## {file}")
        lines.append("")
        lines.append("| key | classification | used_for | audience | consumers | values |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for e in sorted(by_file[file], key=lambda e: e.key):
            vals = ", ".join(e.values) if e.values else ""
            lines.append(f"| {e.key} | probabilistic | {', '.join(e.used_for)} | "
                         f"{', '.join(e.audience)} | {', '.join(e.consumers)} | {vals} |")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
