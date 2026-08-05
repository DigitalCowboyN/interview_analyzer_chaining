# tools/capability/reader.py
from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import List

from src.ingestion.front_matter import parse_front_matter
from tools.code.reader import KEY_MODULES, load_units, packages


@dataclass
class Capability:
    slug: str
    kind: str            # primary | child | variant
    tier: str            # core | enabling  ("" on children/variants — inherited)
    parent: str          # "" on primaries
    implemented_by: List[str]
    statement: str
    path: str


def load_capabilities(root: str = ".", cap_dir: str = "docs/capabilities") -> List[Capability]:
    caps: List[Capability] = []
    for path in sorted(glob.glob(os.path.join(root, cap_dir, "*.md"))):
        if os.path.basename(path) == "index.md":
            continue
        try:
            text = open(path, encoding="utf-8", errors="ignore").read()
        except OSError:
            continue
        fm, offset = parse_front_matter(text)
        if not fm or fm.get("type") != "Capability":
            continue
        caps.append(Capability(
            slug=os.path.splitext(os.path.basename(path))[0],
            kind=str(fm.get("kind", "")),
            tier=str(fm.get("tier", "")),
            parent=str(fm.get("parent", "")),
            implemented_by=list(fm.get("implemented_by") or []),
            statement=text[offset:].strip(),
            path=path,
        ))
    return caps


def real_code_units(root: str = ".") -> set:
    """Valid implemented_by targets — the code map's unit registry (single source)."""
    return set(packages(root)) | set(KEY_MODULES)


def code_nodes(root: str = "."):
    """CodeUnit nodes (with .unit + .role) for the coverage check."""
    return load_units(root)
