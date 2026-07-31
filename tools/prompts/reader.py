from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass, field
from typing import List

import yaml

_FMT = re.compile(r'"[a-z_]+":\s*"([a-z][a-z|_\-]+)"')
STAGE_BY_PREFIX = {
    "src/enrichment": "enrichment", "src/ingestion": "ingestion",
    "src/ask": "ask", "src/lens": "lens", "src/api": "api",
}


@dataclass
class PromptEntry:
    file: str
    key: str
    used_for: List[str] = field(default_factory=list)
    audience: List[str] = field(default_factory=list)
    values: List[str] = field(default_factory=list)
    consumers: List[str] = field(default_factory=list)


def extract_values(text: str) -> List[str]:
    for m in _FMT.finditer(text):
        if "|" in m.group(1):
            return m.group(1).split("|")
    bullets = re.findall(r'^\s*-\s*([A-Za-z][\w\- ]*?)\s*$', text, re.M)
    if len(bullets) >= 2:
        return [b.strip() for b in bullets]
    return []


def derive_consumers(prompt_filename: str, root: str = ".") -> List[str]:
    base = os.path.basename(prompt_filename)
    if base.startswith("lens_"):
        return ["lens"]                       # loaded dynamically via lens.prompts_file
    roles = set()
    needle = f"prompts/{base}"
    for f in glob.glob(os.path.join(root, "src", "**", "*.py"), recursive=True):
        try:
            if needle in open(f, encoding="utf-8", errors="ignore").read():
                rel = os.path.relpath(f, root).replace(os.sep, "/")
                for prefix, role in STAGE_BY_PREFIX.items():
                    if rel.startswith(prefix):
                        roles.add(role)
        except Exception:
            continue
    return sorted(roles)


def load_prompt_entries(root: str = ".") -> List[PromptEntry]:
    entries: List[PromptEntry] = []
    for path in sorted(glob.glob(os.path.join(root, "prompts", "*.yaml"))):
        base = os.path.basename(path)
        try:
            data = yaml.safe_load(open(path, encoding="utf-8")) or {}
        except Exception:
            continue
        consumers = derive_consumers(base, root)
        for key, v in data.items():
            if not (isinstance(v, dict) and "prompt" in v):
                continue
            entries.append(PromptEntry(
                file=base, key=key,
                used_for=list(v.get("used_for") or []),
                audience=list(v.get("audience") or []),
                values=extract_values(str(v.get("prompt", ""))),
                consumers=consumers,
            ))
    return entries
