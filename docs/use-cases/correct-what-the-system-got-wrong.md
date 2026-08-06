---
type: UseCase
form: feature
category: product
actor: analyst
acceptance_criteria:
  - "Given a wrong speaker attribution, wrong resolution, or a hallucinated lens item, when I correct it, then the correction is appended as a new event, not a silent rewrite"
fulfilled_by: [correct-the-analysis, correct-resolution, edit-text, override-lens-items, remove-segments, rename-reattribute-speakers]
---
As an analyst who spots something the AI got wrong — a misattributed line, a bad resolution, an invented action item — I want to correct it directly and trust that correction is recorded as history, not quietly overwritten, so I can rely on the output without losing an audit trail.
