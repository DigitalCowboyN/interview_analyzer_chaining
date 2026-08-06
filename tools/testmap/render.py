from __future__ import annotations

from typing import Dict, List

from tools.testmap.reader import TEST_TYPES, Test


def render_index(
    tests: List[Test], cap_ver: Dict[str, str], uc_ver: Dict[str, str]
) -> str:
    lines = [
        "# Tests",
        "",
        "The test suite as a graph node set, and what it verifies (`../code/`, "
        "`../capabilities/`, `../use-cases/`). Verification is derived, orthogonal to "
        "implementation coverage.",
        "",
    ]
    for tt in TEST_TYPES:
        group = sorted((t for t in tests if t.test_type == tt), key=lambda t: t.slug)
        if not group:
            continue
        lines.append(f"## {tt}")
        lines.append("")
        for t in group:
            target = t.target or "—"
            verifies = ", ".join(t.verifies) or "—"
            lines.append(
                f"- `{t.slug}` ({t.n_tests}) → {target}  ·  verifies: {verifies}"
            )
        lines.append("")

    lines.append("## Verification rollup")
    lines.append("")
    lines.append("Use-cases:")
    for slug in sorted(uc_ver):
        lines.append(f"- {slug}: {uc_ver[slug]}")
    lines.append("")
    lines.append("Capabilities:")
    for slug in sorted(cap_ver):
        lines.append(f"- {slug}: {cap_ver[slug]}")

    return "\n".join(lines).rstrip() + "\n"
