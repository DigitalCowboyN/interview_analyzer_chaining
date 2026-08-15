---
type: CodeUnit
unit: tools.corpus
role: tooling
key_modules: [reader, check, model]
---
The corpus substrate (ADR-0024): a single type-primary, repo-wide intake that discovers every
OKF document by its own top-of-file `type:` frontmatter — never by folder or a body match —
plus a non-blocking misfiled check. The foundation the domains project over.
