"""The cross-domain graph's schema: `NODE_DOMAINS` maps each node type to its `<domain>:<id>`
address slug, and `EDGES` is the extensible registry of `EdgeType`s (authored, from a
frontmatter field, or derived, from a named builder in `tools.graph.reader`) that
`tools.graph.reader.harvest` walks to assemble the graph — adding an edge on existing node
types is a one-entry change here."""
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
    "UseCase": "use-cases",
    "Test": "tests",
    "GlossaryTerm": "glossary",
    "GraphQuery": "graph-queries",
    "Prompt": "prompts",
    "Service": "service",
    "EnvVar": "env",
    # reserved: Spec→spec
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
    EdgeType("contains", "contained_by", "CodeUnit", "CodeUnit", "derived",
             field="contains_edges", resolve="id",
             description="Hierarchy: a package contains its sub-packages and modules."),
    EdgeType("governs", "governed_by", "ADR", "CodeUnit", "authored",
             field="governs", resolve="path",
             description="An architectural decision constrains code under a path."),
    EdgeType("supersedes", "superseded_by", "ADR", "ADR", "authored",
             field="supersedes", resolve="id",
             description="A decision replaces an earlier one."),
    EdgeType("fulfilled_by", "fulfills", "UseCase", "Capability", "authored",
             field="fulfilled_by", resolve="id",
             description="A use-case's intent is reached toward by a capability's implementation."),
    EdgeType("verifies", "verified_by", "Test", "CodeUnit|UseCase|Capability", "derived",
             field="verifies_edges", resolve="id",
             properties=[PropSpec("test_type", enum=["unit", "integration", "e2e"])],
             description="A test proves a code unit works, or an acceptance test proves an intent."),
    EdgeType("defined_in", "defines", "GlossaryTerm", "CodeUnit", "authored",
             field="source", resolve="file",
             description="A glossary term is defined in the code unit that owns its source file."),
    EdgeType("consumed_by", "consumes", "GraphQuery", "CodeUnit", "derived",
             field="gq_consumed_by", resolve="id",
             description="A graph query is consumed by the code units that call it."),
    EdgeType("consumed_by", "consumes", "Prompt", "CodeUnit", "derived",
             field="prompt_consumed_by", resolve="id",
             description="A prompt is consumed by the code units that use it."),
    EdgeType("reads", "read_by", "GraphQuery", "GlossaryTerm", "derived",
             field="reads_edges", resolve="id",
             description="A graph query reads nodes of a Neo4j label (a glossary term)."),
    EdgeType("writes", "written_by", "CodeUnit", "GlossaryTerm", "derived",
             field="writes_edges", resolve="id",
             description="A projection-handler module writes nodes of a Neo4j label (glossary term)."),
    EdgeType("requires", "required_by", "Service", "Service", "derived",
             field="requires_edges", resolve="id",
             description="A compose service must be up before this one (compose depends_on)."),
    EdgeType("configured_by", "configures", "Service", "EnvVar", "derived",
             field="configured_by_edges", resolve="id",
             description="A compose service is configured by an inline environment variable."),
]
