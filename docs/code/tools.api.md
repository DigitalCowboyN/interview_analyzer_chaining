---
type: CodeUnit
unit: tools.api
role: tooling
key_modules: [reader, render, check]
---
The HTTP-surface domain: catalogues FastAPI routes and reconciles them against the running app's `openapi.json` so the documented surface never drifts from the served one.
