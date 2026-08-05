# Operations Capabilities + `category` Axis — design

**Status:** approved by owner 2026-08-05 (brainstorm dialogue).
**Program:** Round B of the operations-capabilities work (Round A extended the code map
to `tools/`, shipped PR #29). This round adds a **`category: product | operations`**
axis to the capability domain and authors the **operations** capabilities — the
guarded-knowledge-graph program itself, captured as first-class capabilities linked to
the `tools/` code units. Plus a small product capability (file management) the owner
called out.

## Framing (locked in brainstorm + research)

Capabilities so far have all been *product* — what the Interview Analyzer does for
analysts, implemented by `src/`. But the repo also has a substantial **operations**
capability: the knowledge graph, guards, ADRs, drift detection, cascade, honesty check
— implemented by `tools/`. Capturing it as capabilities is how the work compounds
(self-documenting), and it is categorically different from product value.

Industry capability maps "group capabilities into categories, such as operational,
customer-facing, or strategic." We adopt **`category`** as the axis word (the industry
term) with two values: **`product`** (what the app does) and **`operations`** (what the
repo does to stay correct and advanceable). `category` is **orthogonal to `tier`**
(core/enabling still applies within each category).

Two principles the owner locked, carried into this domain:
- **`implemented_by` = "helps fulfill," not "shares code."** A child/variant earns its
  place by contributing to the capability, regardless of code overlap with siblings.
- **Small is still a capability.** File management is a real (small) capability.

## The `category` axis

New authored frontmatter key on **primaries**: `category: <value>` (children/variants
inherit their primary's category, exactly like `tier`).

`category` (not `class` — a Python reserved word; not `realm` — non-standard) is the
industry axis term and a valid dataclass field name.

**The axis is an OPEN, ordered set — not a fixed pair.** Industry capability maps use
categories such as *strategic, customer-facing/product, operational, supporting/
enabling*. We define the recognized set as an extensible constant:

```
CATEGORIES = ["product", "operations", "strategic", "supporting"]
```

- **product** — what the app does for analysts (the 11 existing primaries + file
  management). *Populated this round.*
- **operations** — what the repo does to stay correct and advanceable (the
  knowledge-graph program). *Populated this round.*
- **strategic** — direction-setting capabilities. *Reserved; none yet.*
- **supporting** — support tools / systems that enable the above (e.g. dev
  infrastructure, ops systems we may build later). *Reserved; none yet — but the axis
  must accept a `category: supporting` node the day we make one, with no code change.*

The reader stores `category` as a free string; the **guard validates it against
`CATEGORIES`** (adding a value = one edit to the list); the **renderer groups by
`CATEGORIES` order, skipping empty categories** (exactly as it already skips empty
tiers). So `strategic`/`supporting` cost nothing until used, and adding them later is a
one-line change — do not hardcode a two-value assumption anywhere.

## Nodes added

### Operations tree (`category: operations`)

One primary + nine children, each child mapping 1:1 to a `tools/` package (so the
tightened `tooling` coverage comes out clean):

| node | kind | implemented_by |
| --- | --- | --- |
| maintain-a-guarded-knowledge-graph | primary (operations, core) | — |
| govern-architectural-decisions | child | `tools.adr` |
| map-the-code | child | `tools.code` |
| map-capabilities | child | `tools.capability` |
| catalog-the-cli-surface | child | `tools.cli` |
| catalog-the-api-surface | child | `tools.api` |
| maintain-the-glossary | child | `tools.glossary` |
| catalog-the-graph-queries | child | `tools.graphq` |
| catalog-the-prompt-registry | child | `tools.prompts` |
| disclose-knowledge-and-check-specs | child | `tools.knowledge` |

### File management (`category: product`)

| node | kind | implemented_by |
| --- | --- | --- |
| manage-transcript-files | primary (product, core) | `api` |

A small, distinct product capability — the pre-ingest step of getting source
transcripts into the system (`src/api/routers/files`). Its own primary per "small is
still a capability."

## Coverage tightens — `tooling` becomes mandatory

The capability guard's coverage check currently requires only `pipeline-layer` and
`surface` code units to be claimed. Now that operations capabilities exist, **`tooling`
joins the mandatory set**: every `tools/` package must be claimed by some capability.
The 9 operations children claim all 9 `tools.*` units → clean. A 10th tool added later
without a capability → `capability-check` flags it (the operations map stays as
drift-guarded as the product map). Infrastructure/model units remain advisory.

## Module changes — `tools/capability/`

- **`reader.py`** — `Capability` gains a `category: str` field; `load_capabilities`
  reads the `category:` key (default `""`).
- **`render.py`** — `render_index` groups **by category → tier → primary → children**,
  iterating the `CATEGORIES` order and **skipping empty categories** (so `strategic`/
  `supporting` simply don't render until populated). A `## product` / `## operations`
  top level, then `### core`/`### enabling`, then primaries. Deterministic.
- **`check.py`** — `CATEGORIES` constant lives here (the single source of truth reader/
  render/check share). `check_classification` additionally requires a **primary** to
  carry `category in CATEGORIES`. `check_coverage` adds `tooling` to `_MANDATORY_ROLES`.
  `check_links` unchanged (the code registry already includes `tools.*` from Round A).
  Non-blocking throughout.
- **Backfill** — stamp `category: product` on the 11 existing primaries; author the 10
  new nodes (1 operations primary + 9 operations children + 1 file-management primary).
  Regenerate `docs/capabilities/index.md`.

`load_units` (code reader) already returns the 9 `tools.*` nodes with `role: tooling`,
so coverage and link-resolution work with no code-map change this round.

## The guard after this round — `make capability-check` (non-blocking)

Unchanged shape, wider scope: **link resolution** (every `implemented_by` resolves —
now spanning `src` and `tools`), **coverage** (pipeline-layer + surface + **tooling**
mandatory; infra/model advisory), **classification** (kind; primary has `tier` **and
`category`**; child/variant parent resolves), **index-sync**.

## Testing

- **Unit** — `load_capabilities` parses `category`; `render_index` produces a
  `## product` then `## operations` section in `CATEGORIES` order and **omits an empty
  category** (a `strategic`/`supporting` section must NOT appear when unpopulated);
  `check_classification` flags a primary missing `category` **and one whose `category`
  is not in `CATEGORIES`**;
  `check_coverage` now flags an unclaimed `tooling` unit (and still not an unclaimed
  infra unit); `check_links` resolves a `tools.*` slug. Assert no check raises.
- **Smoke** — `make capability-index` writes the catalogue with both categories;
  `make capability-check` clean after backfill (all 9 `tools.*` claimed, all primaries
  categorized); `make cli-check` + `make knowledge-check` clean.

## Capture as ADR

Capture **ADR-0018**: adopt the `category` axis — an **open, ordered set**
(`product, operations, strategic, supporting`; `product`/`operations` populated,
`strategic`/`supporting` reserved) — and the operations capabilities. `source:` = this
spec. Refines ADR-0017 (the capabilities domain) rather than superseding it — it adds a
classification axis and the operations tree. Record the reserved values so the decision
to leave the axis extensible is durable, not just implied by a constant.

## Non-goals (this round)

- **The `fulfills` / use-case edge** — round 3 (use-cases), still deferred.
- **Modeling non-package operations plumbing** (the interpreter script, hook wiring)
  as separate capabilities — they are folded into `disclose-knowledge-and-check-specs`
  / the relevant tool; coverage is package-level.
- **A capability→code Mermaid map / the general graph-links renderer** — still the
  owner-queued next topic after this.
- **Reverse `capability:` markers in code** — capabilities point at the code map.
- **Blocking** on any finding.

## Knowledge-graph check

Reviewed against `docs/index.md` on 2026-08-05.

| domain | touched? | consulted / reconciled | notes |
| --- | --- | --- | --- |
| capabilities | yes | the domain being extended — `category` axis + 11 new nodes + coverage change | — |
| code | yes (read-only) | operations `implemented_by` → `tools.*` units (added in Round A); `load_units` is the registry | no code-map change |
| adr | yes (at impl) | ADR-0018 captured; refines ADR-0017 | deferred to implementation |
| cli | no | — | `capability-*` targets already catalogued (from the capabilities round) |
| glossary / api / prompts / graph-queries | no | — | no vocabulary/surface/prompt/query change |

**Verdict:** reconciled — capabilities (subject) + code (read-only, via Round A's
`tools.*` nodes) consulted; adr reconciled in the plan.
