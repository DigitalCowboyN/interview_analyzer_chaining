---
type: UseCase
form: feature
category: product
actor: analyst
acceptance_criteria:
  - "Given a correction made in one browser tab, when another analyst is viewing the same interview, then their view updates without a manual refresh"
fulfilled_by: [serve-workbench-and-gallery, gallery-read, live-notifications, run-read-queries, workbench-write]
---
As an analyst working across many interviews at once alongside others touching the same corpus, I want a workbench and gallery that update themselves the moment something changes, so I'm never staring at stale analysis while I work.
