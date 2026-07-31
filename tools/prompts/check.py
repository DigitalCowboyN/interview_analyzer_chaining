from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List

from tools.prompts.reader import PromptEntry, load_prompt_entries
from tools.prompts.render import render_catalog
from tools.glossary.model import load_glossary

# prompt key -> glossary term
KEY_TO_TERM = {
    "function_type": "function_type", "structure_type": "structure_type",
    "purpose": "purpose", "topic_level_1": "topic_level_1", "topic_level_3": "topic_level_3",
    "entity_mentions": "entity-type", "claims": "claim-kind",
}
INTERNAL_ROLES = {"enrichment", "ingestion", "ask", "lens", "api", "agent"}


@dataclass
class Finding:
    message: str


def check_values_vs_glossary(entries: List[PromptEntry], terms: List) -> List[Finding]:
    by_term = {t.term: t for t in terms}
    findings: List[Finding] = []
    for e in entries:
        term_name = KEY_TO_TERM.get(e.key)
        if not term_name or not e.values:
            continue
        t = by_term.get(term_name)
        if t is None:
            findings.append(Finding(f"prompt {e.file}:{e.key} enumerates values but glossary has no term {term_name}"))
            continue
        if set(e.values) != set(t.values):
            missing = sorted(set(e.values) - set(t.values))
            extra = sorted(set(t.values) - set(e.values))
            findings.append(Finding(
                f"glossary term {term_name} out of sync with the registry ({e.file}:{e.key}) — "
                f"missing: {missing}, extra: {extra} — update the glossary"))
    return findings


def check_audience_vs_consumers(entries: List[PromptEntry]) -> List[Finding]:
    findings: List[Finding] = []
    for e in entries:
        declared_internal = {a for a in e.audience if a in INTERNAL_ROLES}
        for role in sorted(declared_internal):
            if role not in e.consumers:
                findings.append(Finding(f"{e.file}:{e.key} declares audience {role} but no code consumes it"))
        for role in sorted(set(e.consumers) - set(e.audience)):
            findings.append(Finding(f"{e.file}:{e.key} is consumed by {role} but audience does not list it"))
    return findings


def check_orphan(entries: List[PromptEntry]) -> List[Finding]:
    findings: List[Finding] = []
    seen_files = set()
    for e in entries:
        external = [a for a in e.audience if a not in INTERNAL_ROLES]
        if not e.consumers and not external and e.file not in seen_files:
            findings.append(Finding(f"{e.file} appears unused (no code consumer)"))
            seen_files.add(e.file)
    return findings


def check_missing_metadata(entries: List[PromptEntry]) -> List[Finding]:
    return [Finding(f"{e.file}:{e.key} has no used_for/audience metadata")
            for e in entries if e.consumers and not (e.used_for or e.audience)]


def check_catalog_in_sync(catalog_path: str, entries: List[PromptEntry]) -> List[Finding]:
    want = render_catalog(entries)
    have = open(catalog_path, encoding="utf-8").read() if os.path.exists(catalog_path) else ""
    if want != have:
        return [Finding("docs/prompts/index.md out of sync — run make prompt-index")]
    return []


def run_all(root: str = ".") -> List[Finding]:
    entries = load_prompt_entries(root)
    terms = load_glossary(os.path.join(root, "docs/glossary"))
    findings: List[Finding] = []
    findings += check_values_vs_glossary(entries, terms)
    findings += check_audience_vs_consumers(entries)
    findings += check_orphan(entries)
    findings += check_missing_metadata(entries)
    findings += check_catalog_in_sync(os.path.join(root, "docs/prompts/index.md"), entries)
    return findings
