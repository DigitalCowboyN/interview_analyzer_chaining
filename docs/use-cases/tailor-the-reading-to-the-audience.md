---
type: UseCase
form: use-case
category: product
actor: team lead
acceptance_criteria:
  - "Given a new lens profile (YAML + prompts) for a purpose the corpus hasn't been read for before, when it's run, then the engine extracts that lens's structured items without any lens-specific code being written"
fulfilled_by: [run-lens-engine, per-lens-extractors]
level: user-goal
---
As a team lead who needs a specific shape of output — meeting minutes for a standup, a persona profile for a research readout — I want to apply a purpose-built lens and get exactly that structure back, without waiting for engineering to hand-write a new extractor every time I need a new reading.
