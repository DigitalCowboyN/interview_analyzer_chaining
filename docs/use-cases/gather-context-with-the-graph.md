---
type: UseCase
form: use-case
category: operations
actor: maintainer
acceptance_criteria:
  - "Given a task that targets some code, when the agent walks the graph from it, then it can progressively walk up to the governing intent (capability/use-case/ADR) and out to related code, without loading the whole graph"
  - "Given a question the graph cannot answer (e.g. no decision links to the code), when the agent exhausts the reachable set, then it reports the absence honestly rather than inferring relevance from proximity"
  - "Given a coarse question, when the agent walks at module grain, then no symbol bodies are parsed; symbol detail (signatures, calls) is disclosed only when it descends to symbol grain"
fulfilled_by: [walk-the-graph-for-context, link-the-domains, map-the-code]
level: summary
---
As a maintainer (often working through an AI agent that carries no memory forward), I want to walk the codebase's own knowledge graph to gather the correct, minimal context at the right layer for whatever task I'm on — tracing code up to the intent that governs it and out to what it relates to — so I spend effort on the task, not on re-reading the whole system.
