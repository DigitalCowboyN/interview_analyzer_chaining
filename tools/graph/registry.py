from __future__ import annotations

from dataclasses import dataclass, field as _field
from typing import List


@dataclass
class PropSpec:
    name: str
    enum: List[str] = _field(default_factory=list)


@dataclass
class EdgeType:
    name: str                       # verb (traceability vocabulary)
    inverse: str                    # "" if none
    from_type: str
    to_type: str
    source: str                     # "authored" | "derived"
    field: str = ""                 # authored: frontmatter key on the from-node; derived: origin tag
    resolve: str = "id"             # id | path  (how a target string maps to a to-node id)
    properties: List[PropSpec] = _field(default_factory=list)
    description: str = ""


# Node type -> cascade domain slug (for <domain>:<id> addressing).
# Add a row (+ a reader adapter in reader.py) for a genuinely new node type.
NODE_DOMAINS = {
    "CodeUnit": "code",
    "Capability": "capabilities",
    "ADR": "adr",
    # reserved: GlossaryTerm→glossary, Prompt→prompts, GraphQuery→graph-queries,
    # Spec→spec, Test→test, UseCase→usecase
}

# The extensible edge registry. Adding an authored edge on existing node types is a
# one-entry change here (harvest is registry-driven). Reserved edges (verifies/fulfills)
# are added the same way in their rounds.
EDGES: List[EdgeType] = [
    EdgeType("implements", "implemented_by", "Capability", "CodeUnit", "authored",
             field="implemented_by", resolve="id",
             description="A capability's current implementation reaches toward its intent."),
    EdgeType("child_of", "parent_of", "Capability", "Capability", "authored",
             field="parent", resolve="id",
             description="Decomposition: a narrower intent sits under a broader one."),
    EdgeType("depends_on", "depended_on_by", "CodeUnit", "CodeUnit", "derived",
             field="dep_edges", resolve="id",
             description="Static import dependency between code units."),
    EdgeType("governs", "governed_by", "ADR", "CodeUnit", "authored",
             field="governs", resolve="path",
             description="An architectural decision constrains code under a path."),
    EdgeType("supersedes", "superseded_by", "ADR", "ADR", "authored",
             field="supersedes", resolve="id",
             description="A decision replaces an earlier one."),
]
