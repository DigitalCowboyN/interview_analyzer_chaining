from __future__ import annotations

from typing import List

from tools.glossary.model import Term


def render_index(terms: List[Term]) -> str:
    lines = ["# Glossary", "", "| term | kind | source |", "| --- | --- | --- |"]
    for t in sorted(terms, key=lambda t: (t.kind, t.term)):
        lines.append(f"| {t.term} | {t.kind} | {t.source} |")
    return "\n".join(lines) + "\n"
