---
type: UseCase
form: feature
category: operations
actor: maintainer
acceptance_criteria:
  - "Given a surface the system exposes (API, CLI, graph queries, prompts, code, capabilities, use-cases), when its catalog check runs, then it's guarded against drift from what's actually live"
fulfilled_by: [catalog-the-api-surface, catalog-the-cli-surface, catalog-the-graph-queries, catalog-the-prompt-registry, map-the-code, map-capabilities, map-use-cases]
---
As a maintainer who can't trust documentation to describe reality on its own, I want every surface the system exposes — its endpoints, commands, queries, prompts, capabilities — cataloged and checked against what's actually running, so stale docs get caught before they mislead the next person.
