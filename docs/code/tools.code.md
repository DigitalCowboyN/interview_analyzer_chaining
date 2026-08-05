---
type: CodeUnit
unit: tools.code
role: tooling
key_modules: [reader, render, check]
---
The code map itself: enumerates every src/ and tools/ package as a node, derives its dependency + I/O edges, and reconciles the authored map against the real tree.
