---
type: CodeUnit
unit: enrichment.executor
role: pipeline-layer
key_modules: []
---
Runs registered extractors: one focused, schema-enforced, Pydantic-validated LLM call per dimension per unit; invalid responses degrade to a review flag, never a failed run.
