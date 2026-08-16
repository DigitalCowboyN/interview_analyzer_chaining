from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

# OKF document types → their expected home directory (repo-relative). A record of type X
# found outside its home is "misfiled". Code is NOT a document type — it is derived from
# source (src/, tools/) by tools/code/reader.py, not authored as markdown (see the
# hierarchical-code-intake ADR). These four all carry `type:` frontmatter today.
# Code-DERIVED nodes (Test, GraphQuery, Prompt) are NOT documents — they self-declare via
# `# okf:` markers in code / YAML keys, handled in a later phase (ADR-0024), not here.
OKF_HOMES: Dict[str, str] = {
    "ADR": "docs/adr",
    "Capability": "docs/capabilities",
    "UseCase": "docs/use-cases",
    "Term": "docs/glossary",
}


@dataclass
class Record:
    type: str            # the file's OWN frontmatter `type:` — an OKF document type
    id: str              # local id: the file stem
    path: str            # provenance: repo-relative path the record was found at
    frontmatter: dict    # parsed top-of-file frontmatter
    body: str            # content after the frontmatter (the record's claim + context)
