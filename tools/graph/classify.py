# tools/graph/classify.py
from __future__ import annotations

from typing import Dict, Tuple

from tools.capability.reader import load_capabilities
from tools.code.reader import load_units
from tools.graph.reader import harvest


def _id(addr: str) -> str:
    return addr.split(":", 1)[1]


def derive_axes(root: str = ".") -> Dict[str, Tuple[str, str]]:
    """code unit id -> (category, determinism), computed from the assembled cross-domain edges.

    category: the category of a capability that `implements` the unit (direct only; a unit no
              capability implements has no category — the reachability signal, not a gap).
    determinism: probabilistic if the unit is consumed_by a Prompt, or depends_on the `agents`
                 package/module; else deterministic."""
    edges = harvest(root)
    cap_category = {c.slug: c.category for c in load_capabilities(root)}

    category: Dict[str, str] = {}
    probabilistic = set()
    for e in edges:
        if e.type == "implements":
            cat = cap_category.get(_id(e.src))
            if cat:
                category.setdefault(_id(e.dst), cat)
        elif e.type == "consumed_by" and e.src.startswith("prompts:"):
            probabilistic.add(_id(e.dst))
        elif e.type == "depends_on" and _id(e.dst).split(".")[0] == "agents":
            probabilistic.add(_id(e.src))

    axes: Dict[str, Tuple[str, str]] = {}
    for u in load_units(root):
        det = "probabilistic" if u.unit in probabilistic else "deterministic"
        axes[u.unit] = (category.get(u.unit, ""), det)
    return axes
