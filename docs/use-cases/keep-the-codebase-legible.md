---
type: UseCase
form: use-case
category: operations
actor: maintainer
acceptance_criteria:
  - "Given a change to a surface the knowledge graph covers, when the relevant check runs, then drift between documented and actual knowledge is reported, not silently accepted"
fulfilled_by: [maintain-a-guarded-knowledge-graph]
level: summary
---
As a maintainer inheriting a system built and extended across many sessions — including by AI agents that don't carry memory forward — I want the codebase's own knowledge to explain itself, so work compounds instead of every session rediscovering the same ground.
