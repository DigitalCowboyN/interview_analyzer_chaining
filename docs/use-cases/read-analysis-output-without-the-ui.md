---
type: UseCase
form: user-story
category: supporting
actor: downstream tool integrator
acceptance_criteria:
  - "Given a completed interview analysis, when a downstream tool requests it by file or by sentence, then it can retrieve the output directly without going through the workbench UI"
fulfilled_by: [access-analysis-output-files]
---
As someone building a downstream tool or script that needs to consume a finished interview analysis, I want to list and fetch the output files directly, so I can integrate without reimplementing the workbench.
