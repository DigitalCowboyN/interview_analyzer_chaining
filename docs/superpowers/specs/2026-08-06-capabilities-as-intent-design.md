# Capabilities as Intent — model clarification

**Status:** approved by owner 2026-08-06 (brainstorm dialogue).
**Program:** a refinement of the capability domain (ADR-0017 / ADR-0018). No schema or
code change — the guard already permits it. This captures *what a capability is* so the
model doesn't drift back into conflating a capability with its implementation.

## Framing (locked in brainstorm)

A capability is **durable intent** — an expectation of what the product (or repo) does.
It is **never "built."** It is only *currently implemented*, by an iteration that can be
replaced wholesale in two years while the capability stands unchanged. So there is **no
maturity / status / "built" dimension** to add; that framing is wrong.

The **degree of implementation lives entirely in the links** (`implemented_by`), and is
**derived**, never authored as an attribute. A capability with **empty or partial**
`implemented_by` is legitimate — an intent that current implementations only partly
reach toward — not a defect. (The operations primary `maintain-a-guarded-knowledge-graph`
already ships with `implemented_by: []` and the guard is clean.)

### The layers (the knot this resolves)

Going *down* the capability tree makes the intent **narrower, not more implementation-y.**
`primary`, `child`, and `variant` are **all intent** (the WHAT), differing only in grain
and shape:

- **primary** — a broad intent.
- **child** — a *narrower* intent (a sub-ability). NOT "how you do it" — a smaller *what*.
- **variant** — an *alternative form* of the same intent (e.g. per-lens extractors;
  `chat-failover` vs `pinned-embeddings`).
- **code** (`implemented_by`) — the **only** HOW: the current mechanism reaching toward
  the intent.

`parent` is **decomposition** (a smaller WHAT inside a bigger WHAT), not a how-chain.
There is **no middle "how-definition" capability layer** — because the how-*decisions*
already live in their own domains: **ADRs and specs**. ("We infer speakers with focused
LLM calls, not one mega-call" is ADR-0007, not a capability child.) Division of labor:

| artifact | answers |
| --- | --- |
| capability (intent tree) | *what / why* the product does it (durable) |
| ADR / spec | *how we decided* to do it (the "how-definition") |
| code (`implemented_by`) | the *current implementation* of that decision |

A third capability layer would duplicate the ADR/spec corpus and drift from both it and
the code.

### The inverse, categories, and use-cases (settled)

- **`implements` inverse = derived**, not authored in code. `implements(unit)` is just
  the `implemented_by` edges read backward — one authored direction, so the inverse is
  free and can never disagree with itself. No `implements:` markers in `src/`/`tools/`
  (keeps "capabilities point at the code map; code untouched"). Surfacing the derived
  inverse (e.g. in the code map) is **deferred to the queued graph-links topic** — that
  edge traversal is exactly what it will render.
- **Categories follow industry** and stay an open set. `product` + `operations` are
  populated; `strategic` (direction-setting) + `supporting` remain reserved and likely
  thin — "operations" (our dev-tooling) and "supporting" may ultimately be the same
  bucket. No change now; the open set means we adjust only when a concrete capability
  forces it (do not pre-commit a five-category taxonomy we can't fill).
- **Capability ↔ use-case is indirect / many-to-many** — a use-case may inform several
  capabilities; a capability may fulfill several use-cases; operational/support
  capabilities may fulfill none. The two must **not** be tied together (round 3, and even
  then loosely).

## Deliverables

1. **ADR-0019** — "Capabilities are durable intent; implementation is a derived,
   replaceable link." The durable record of the model above. Refines ADR-0017/0018;
   supersedes nothing.
2. **A capability-domain concept doc** — `docs/capabilities/README.md` — the
   human/agent-facing "how to think about capabilities here": the intent tree, the three
   artifacts table, empty/partial links are legitimate, derived inverse, categories,
   use-case indirection. So a future author doesn't re-conflate intent with code. Linked
   from `docs/capabilities/index.md`'s header (or the cascade).
3. **(Owner-flagged) a concrete aspirational capability** demonstrating the principle —
   e.g. `import-transcripts` (product/core): the product *should* let an analyst bring
   source transcripts in; today there is **no** implementation (`implemented_by: []`) —
   ingestion reads from a directory, there is no import/upload feature. This is the pure
   intent-outruns-implementation case. **This one node is the single judgment call for
   owner review** — include it, or capture the principle only and add concrete
   aspirational capabilities as real intents are named.

## Non-goals (this round)

- **A `status`/`built`/maturity field** — rejected; degree is derived from links.
- **A derived "intent vs. implementations" report / the `implements` inverse in the code
  map** — deferred to the graph-links topic (it is the same edge set).
- **Renaming `operations` → `supporting` or reworking the category set** — no concrete
  case forces it yet.
- **Any capability ↔ use-case link** — round 3.
- **Code / schema change** — none; the guard already permits empty/partial links.

## Testing

- **Smoke** — if the `import-transcripts` node is included: `make capability-index` then
  `~/.pyenv/shims/python -m tools.capability check` → `capability-check: clean` (an
  `implemented_by: []` primary with `kind/tier/category` set passes; it adds no coverage
  obligation — coverage is code→capability). `make knowledge-check` clean. No unit tests
  (authored docs; the guard behaviour is unchanged and already tested).

## Knowledge-graph check

Reviewed against `docs/index.md` on 2026-08-06.

| domain | touched? | consulted / reconciled | notes |
| --- | --- | --- | --- |
| capabilities | yes | concept README + (optional) one intent node; no schema change | the subject |
| adr | yes | ADR-0019 (the principle); refines 0017/0018 | — |
| code | no | — | `implements` inverse deferred to graph-links; code untouched |
| cli / glossary / api / prompts / graph-queries | no | — | no target/vocabulary/surface/prompt/query change |

**Verdict:** reconciled — capabilities (subject) + adr consulted; the derived-inverse /
code-map surfacing is explicitly deferred to the graph-links topic.
