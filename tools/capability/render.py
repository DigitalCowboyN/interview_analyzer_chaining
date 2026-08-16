"""Renders the `tools.capability` `Capability` list into the `docs/capabilities/index.md`
catalog, grouped by category and tier with children/variants nested under their primary."""

from __future__ import annotations

from typing import Dict, List

from tools.capability.reader import CATEGORIES, Capability

_TIERS = ["core", "enabling"]


def render_index(caps: List[Capability]) -> str:
    primaries = [c for c in caps if c.kind == "primary"]
    children_of: Dict[str, List[Capability]] = {}
    for c in caps:
        if c.parent:
            children_of.setdefault(c.parent, []).append(c)
    lines = ["# Capabilities", "",
             "What the system can do, linked to the code map (`../code/`).", ""]
    for category in CATEGORIES:
        cat_primaries = [p for p in primaries if p.category == category]
        if not cat_primaries:
            continue  # reserved/empty category — omit
        lines.append(f"## {category}")
        lines.append("")
        for tier in _TIERS:
            tier_primaries = sorted((p for p in cat_primaries if p.tier == tier), key=lambda c: c.slug)
            if not tier_primaries:
                continue
            lines.append(f"### {tier}")
            lines.append("")
            for p in tier_primaries:
                lines.append(f"#### {p.slug}")
                lines.append(p.statement)
                lines.append("")
                lines.append(f"- **implemented_by:** {', '.join(p.implemented_by) or '—'}")
                for k in sorted(children_of.get(p.slug, []), key=lambda c: c.slug):
                    tag = " _(variant)_" if k.kind == "variant" else ""
                    impl = ', '.join(k.implemented_by) or '—'
                    lines.append(f"- {k.slug}{tag} — {k.statement} ({impl})")
                lines.append("")
    return "\n".join(lines).rstrip() + "\n"
