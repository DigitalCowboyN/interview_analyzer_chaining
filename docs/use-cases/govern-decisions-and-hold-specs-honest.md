---
type: UseCase
form: feature
category: operations
actor: maintainer
acceptance_criteria:
  - "Given a decision that changes an earlier one, when it's captured, then it supersedes the old record explicitly rather than silently overriding it in prose"
  - "Given a new spec or plan, when it's written, then it's nudged to reconcile against the knowledge domains it touches"
fulfilled_by: [govern-architectural-decisions, disclose-knowledge-and-check-specs, link-the-domains, maintain-the-glossary]
---
As a maintainer trying to keep a fast-moving system's decisions honest, I want architectural choices captured durably with explicit supersession, and every new spec held accountable to the domains it touches, so decisions don't get silently overridden or forgotten as the system evolves.
