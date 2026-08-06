# Capabilities — how to think about this domain

This bundle catalogs **what the system can do**, as value-framed intent, linked to the
code that reaches toward it. The live map is **[index.md](index.md)** (generated —
grouped by category → tier → primary). This page is the mental model; read it before
authoring a node.

## A capability is durable intent — never "built"

A capability names an **expectation** of the product (or repo). It is never "built" —
only *currently implemented*, by an iteration that can be replaced wholesale later while
the capability stands unchanged. So there is **no status / maturity / "built" field**.
How far current code reaches toward the intent is **derived from the links**, and an
**empty or partial `implemented_by` is legitimate** — an intent that today's
implementation only partly reaches (or hasn't reached at all). `import-transcripts` is
exactly that: a real product intent with `implemented_by: []`, because no import feature
exists yet.

## The tree is all intent; code is the only "how"

Going *down* the tree makes the intent **narrower, not more implementation-y**:

- **primary** — a broad intent.
- **child** — a *narrower* intent (a sub-ability). Not "how you do it" — a smaller *what*.
- **variant** — an *alternative form* of the same intent (e.g. per-lens extractors;
  `chat-failover` vs `pinned-embeddings`).
- **code** (`implemented_by`) — the **only** "how": the current mechanism reaching toward
  the intent.

`parent` is **decomposition** (a smaller *what* inside a bigger *what*), not a how-chain.

## Three artifacts, three questions

There is **no middle "how-definition" capability layer** — that already exists, as its
own domains:

| artifact | answers |
| --- | --- |
| **capability** (this bundle) | *what / why* the product does it (durable intent) |
| **ADR / spec** (`../adr/`, `../superpowers/`) | *how we decided* to do it |
| **code** (`implemented_by` → `../code/`) | the *current implementation* of that decision |

Putting "how" decisions into capability children would just duplicate the ADR/spec
corpus and drift from both it and the code.

## Frontmatter

```yaml
---
type: Capability
kind: primary | child | variant
tier: core | enabling            # primaries carry tier; children/variants inherit
category: product | operations   # primaries carry category (industry axis; strategic/supporting reserved)
parent: <primary-slug>           # children/variants only
implemented_by: [<code-unit slugs from ../code/>]   # may be [] — legitimate
---
One or two lines of value-framed intent (what/why, for whom). Do NOT state "not
implemented" — that is derived from the links.
```

The slug is the filename. `implemented_by` targets are CodeUnit slugs from the
[code map](../code/) — `src` units are bare (`api`); `tools` units are `tools.<pkg>`.

## Links, categories, use-cases

- **`implements` is the inverse of `implemented_by`, derived** — read the edges backward,
  never authored as markers in code (`src/`/`tools/` stay untouched). Surfacing that
  inverse is the job of the forthcoming graph-links work.
- **Categories** follow industry and are an open set. `product` and `operations` are
  populated; `strategic` (direction-setting) and `supporting` are reserved — add a value
  only when a concrete capability forces it.
- **Capability ↔ use-case is indirect and many-to-many** — a use-case may inform several
  capabilities; a capability may fulfill several (or none, for operational/support
  ones). They are **not** linked here (that is a later round, and even then loosely).

## Reconciling

`make capability-check` (non-blocking) reconciles: every `implemented_by` slug resolves
to a real code unit; every pipeline/surface/tooling code unit is claimed by some
capability (coverage — infra/model advisory); every primary is classified (kind + tier +
category); the committed `index.md` matches a fresh render. Run `make capability-index`
after adding or editing a node.
