# Knowledge map

Guarded knowledge domains over this codebase. Land here, then follow the one you're
working in — read its `index.md`, and run its `make <domain>-check` when you change a
surface it covers. All checks are non-blocking (visibility, not gates).

| domain | what it holds | reconcile with |
| --- | --- | --- |
| [adr/](adr/index.md) | architectural decisions (what & why) — consult before locking one | `make adr-check` |
| [glossary/](glossary/index.md) | canonical vocabulary (nodes, lenses, dimensions, graph labels) pinned to code enums | `make glossary-check` |
| [code/](code/index.md) | package/module map: roles, derived deps + I/O, Mermaid pipeline | `make code-check` |
| [capabilities/](capabilities/README.md) | what the system can do (value-framed intent), linked to the code map | `make capability-check` |
| [api/](api/index.md) | HTTP surface vs. committed `openapi.json` | `make api-check` |
| [cli/](cli/index.md) | command surface (CLI + make targets) | `make cli-check` |
| [prompts/](prompts/index.md) | probabilistic components — the LLM prompts the agents use | `make prompt-check` |
| [graph-queries/](graph-queries/index.md) | Neo4j read-query registry (schema + output contract) | `make graphq-check` |
| [graph/](graph/index.md) | cross-domain edge graph (typed links between all domains) | `make graph-check` |

**Writing a spec or plan?** Record a `## Knowledge-graph check` addendum — the
per-domain review of what it touched and what you reconciled (`make knowledge-check`
flags a new one that skipped it). Verdict is one of: **clean** (every touched domain
consulted) · **reconciled** (gaps found and fixed) · **overridden** (a design-affecting
gap the owner accepted, rationale recorded). If the check surfaces a gap a domain
should have caught, don't silently pass: fix mechanical gaps directly; for
design-affecting ones, loop back (change the design) or record an owner override.

Other docs (not knowledge domains): `architecture/` (system overview, data flow),
`product/`, `superpowers/{specs,plans}/` (design specs + implementation plans).
