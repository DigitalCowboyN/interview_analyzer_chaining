---
type: Capability
kind: child
parent: maintain-a-guarded-knowledge-graph
implemented_by: [tools.graph.traverse, tools.graph.neighbors]
---
Walk the ephemeral code knowledge graph on demand — at package, module, or symbol grain — expanding the frontier lazily so an agent can disclose detail vertically (progressive disclosure) and discover relationships horizontally (progressive discovery), using the returned subgraph as working context.
