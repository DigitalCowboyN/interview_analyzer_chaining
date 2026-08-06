from __future__ import annotations

from typing import Dict, List

from tools.capability.reader import CATEGORIES
from tools.usecase.coverage import NOT_COVERED
from tools.usecase.reader import FORMS, UseCase


def render_index(use_cases: List[UseCase], coverage: Dict[str, str]) -> str:
    lines = [
        "# Use-Cases",
        "",
        'The user-centered intents this system serves — the "why" above the '
        "capabilities (`../capabilities/`). Coverage is derived from `fulfilled_by`, "
        "never stored.",
        "",
    ]
    for category in CATEGORIES:
        cat = [u for u in use_cases if u.category == category]
        if not cat:
            continue  # reserved/empty category — omit
        lines.append(f"## {category}")
        lines.append("")
        for form in FORMS:
            form_ucs = sorted((u for u in cat if u.form == form), key=lambda u: u.slug)
            if not form_ucs:
                continue
            lines.append(f"### {form}")
            lines.append("")
            for u in form_ucs:
                state = coverage.get(u.slug, NOT_COVERED)
                lines.append(f"#### {u.slug} — {state}")
                lines.append(u.statement)
                lines.append("")
                lines.append(f"- **actor:** {u.actor or '—'}")
                lines.append(f"- **fulfilled_by:** {', '.join(u.fulfilled_by) or '—'}")
                ac = u.acceptance_criteria
                lines.append(
                    f"- **acceptance_criteria:** {len(ac) if ac else '— none yet'}"
                )
                lines.append("")
    return "\n".join(lines).rstrip() + "\n"
