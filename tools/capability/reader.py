# tools/capability/reader.py
from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import List

from src.ingestion.front_matter import parse_front_matter
from tools.code.reader import load_units

# The capability category axis — an open, ordered set carrying each value's definition.
# A non-empty definition = defined & in use; "" = reserved (declared; define on first use).
# Adding/promoting a value = one edit here; knowledge-check flags a used-but-undefined value.
# Kept dict-shaped so `x in CATEGORIES` / `for c in CATEGORIES` behave as before (keys).
CATEGORIES = {
    "product": "the product itself — the capability a customer directly uses",
    "operations": "running and maintaining the system — CI, infra, projections, the guarded knowledge graph",
    "supporting": (
        "customer-facing but around the product, not the product itself — "
        "self-help, notifications, getting output out"
    ),
    "strategic": "",  # reserved: direction-setting; define on first use
}


def category_defined(name: str) -> bool:
    """True when `name` is a category with a real definition (not reserved / unknown)."""
    return bool(CATEGORIES.get(name))


@dataclass
class Capability:
    slug: str
    kind: str            # primary | child | variant
    tier: str            # core | enabling  ("" on children/variants — inherited)
    parent: str          # "" on primaries
    implemented_by: List[str]
    statement: str
    path: str
    category: str = ""   # product | operations | … (primaries; children inherit)


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
            category=str(fm.get("category", "")),
        ))
    return caps


def real_code_units(root: str = ".") -> set:
    """Valid implemented_by / verifies targets — the derived code node registry (packages + modules)."""
    return {u.unit for u in load_units(root)}
