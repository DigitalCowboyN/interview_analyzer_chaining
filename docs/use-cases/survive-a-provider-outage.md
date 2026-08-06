---
type: UseCase
form: requirement
category: operations
actor: operator
acceptance_criteria:
  - "Given a configured chat provider becomes unavailable, when a call fails, then it transparently fails over to the next configured provider"
  - "Given an embedding call, when a provider becomes unavailable, then it does not fail over, since vectors from different models aren't comparable"
fulfilled_by: [provider-strategy-and-focused-calls, chat-failover, pinned-embeddings]
---
As an operator running analysis against external LLM providers I don't control, I want chat calls to fail over automatically to another provider when one goes down, so a single vendor outage doesn't stall the whole pipeline — while embeddings stay pinned, since mixing vector spaces would corrupt retrieval.
