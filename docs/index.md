# Knowledge map — the guarded knowledge graph

This repo keeps a **guarded knowledge graph over its own codebase**: a set of small
Markdown "domains" under `docs/`, each cataloging one facet of the system (its decisions,
its vocabulary, its code map, what it can do, who it's for, what proves it works) and
**reconciling itself against the real code** so drift gets caught instead of rotting
silently. Every domain is an OKF-conformant bundle plus a generated index plus a
**non-blocking `make <domain>-check`** — visibility, never a gate.

**You are here.** Land on this page, then follow the domain you're working in: read its
`index.md` / `README.md`, and run its check when you change a surface it covers.

## How it's organized — the traceability spine

The domains aren't a flat list; they form a **Requirements Traceability Matrix** — a spine
from *why* down to *proof*:

```
use-case  ──fulfilled_by──▶  capability  ──implemented_by──▶  code  ◀──verifies──  test
  (the user problem)          (durable intent)               (the how)         (the proof)
```

- **use-cases/** — the user-centered *why* (requirements, stories, features), the source layer.
- **capabilities/** — durable *intent*: what the system is expected to do, never "built," only currently implemented by a replaceable iteration.
- **code/** — the *how*: the package/module map, roles and dependencies derived from the import graph.
- **tests/** — the *proof*: the test suite as nodes, and what each test verifies.

This spine carries **two orthogonal, derived coverage axes** — a node is described on both,
independently:

- **Implementation coverage** (`NOT_COVERED / PARTIALLY_COVERED / FULLY_COVERED`) — is the intent built? Derived from `fulfilled_by` / `implemented_by` links down to code.
- **Verification coverage** (`UNVERIFIED / PARTIALLY_VERIFIED / VERIFIED`) — is it proven? Derived from `verifies` edges. A use-case can honestly read *`FULLY_COVERED` + `UNVERIFIED`* — built but not yet proven.

Around the spine sit **supporting registries** — `adr/` (the decisions), `glossary/` (the
vocabulary), `api/`, `cli/`, `prompts/`, `graph-queries/` — and **`graph/`**, the
cross-domain edge layer that stitches every domain's typed links into one traversable
graph you can query (`python -m tools.graph neighbors <domain>:<id>`).

## The pattern every domain follows

One shape, repeated — so a new domain is a *registry addition, not a redesign*:

- **`tools/<domain>/`** = `reader → render → check → CLI`. The reader loads the nodes; render writes the generated `index.md`; check reconciles; the CLI exposes `index | check`.
- **Non-blocking** — every check returns findings and never fails a build; it informs.
- **Authored intent, derived facts.** *Intent* is authored by a human (a capability's value, a use-case's problem, a `# verifies:` acceptance marker). *Facts about the code* — dependencies, test→code links, the coverage states — are **derived from the source**, so they can't drift from it silently. What's derived is never hand-maintained.
- **Progressive disclosure** — this page points to domains, domains point to nodes, nodes point to code. Nothing is dumped into context all at once; you pull the level you need.

## The domains

| domain | what it holds | reconcile with |
| --- | --- | --- |
| [adr/](adr/index.md) | architectural decisions (what & why) — consult before locking one | `make adr-check` |
| [glossary/](glossary/index.md) | canonical vocabulary (nodes, lenses, dimensions, graph labels) pinned to code enums | `make glossary-check` |
| [code/](code/index.md) | package/module map: roles, derived deps + I/O, Mermaid pipeline | `make code-check` |
| [capabilities/](capabilities/README.md) | what the system can do (value-framed intent), linked to the code map | `make capability-check` |
| [use-cases/](use-cases/README.md) | user-centered intents (requirements/stories/features) with derived coverage over capabilities | `make usecase-check` |
| [tests/](tests/README.md) | the test suite as nodes + what it verifies (derived verification axis) | `make testmap-check` |
| [api/](api/index.md) | HTTP surface vs. committed `openapi.json` | `make api-check` |
| [cli/](cli/index.md) | command surface (CLI + make targets) | `make cli-check` |
| [prompts/](prompts/index.md) | probabilistic components — the LLM prompts the agents use | `make prompt-check` |
| [graph-queries/](graph-queries/index.md) | Neo4j read-query registry (schema + output contract) | `make graphq-check` |
| [graph/](graph/index.md) | cross-domain edge graph (typed links between all domains) | `make graph-check` |

Run everything at once with **`make health`**. The whole model is recorded in ADRs
**0016–0022** (cascade, capabilities-as-intent, graph-links, use-cases, tests) — see
[adr/](adr/index.md).

**Writing a spec or plan?** Record a `## Knowledge-graph check` addendum — the
per-domain review of what it touched and what you reconciled (`make knowledge-check`
flags a new one that skipped it). Verdict is one of: **clean** (every touched domain
consulted) · **reconciled** (gaps found and fixed) · **overridden** (a design-affecting
gap the owner accepted, rationale recorded). If the check surfaces a gap a domain
should have caught, don't silently pass: fix mechanical gaps directly; for
design-affecting ones, loop back (change the design) or record an owner override.

Other docs (not knowledge domains): `architecture/` (system overview, data flow),
`product/`, `superpowers/{specs,plans}/` (design specs + implementation plans).
