---
type: UseCase
form: requirement
category: operations
actor: operator
acceptance_criteria:
  - "Given any change to an interview's analysis, when it's applied, then it is appended to the event log, never rewritten in place"
  - "Given the event log, when the read model needs rebuilding, then replaying it in causal order reproduces the same Neo4j state"
fulfilled_by: [maintain-event-source-of-truth, project-events-to-graph]
---
As an operator responsible for the system's integrity, I want every change captured as an immutable, replayable event and the read model rebuilt deterministically from it, so nothing is ever silently lost, and the graph can always be reconstructed from the one source of truth.
