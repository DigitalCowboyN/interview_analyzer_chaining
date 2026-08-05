# Capabilities Domain (#8) — design

**Status:** approved by owner 2026-08-05 (brainstorm dialogue).
**Program:** round 2 of the vertical stack over the *guarded knowledge graph* —
the **what** layer. Round 1 (`docs/code/`, the code map) is the *how*; this round
names **what the system can do**, in value terms, and links each capability to the
CodeUnit nodes that implement it. Use-cases (round 3) sit above/beside capabilities
and are **deferred**.

## Framing (locked in brainstorm + research)

A capability is a **stable statement of what the system can do, independent of how**
— the counterpart to the code map's packages/roles. Research grounding: capabilities
are "functional abstractions… a high-level perspective of system functionality" that
stay stable while implementations churn, align with bounded contexts (not technical
packages), and decompose L1 primary → L2 sub-capabilities (start with 5–10 primaries,
drill down only where justified). OKF models them as a concept type linked to
implementing units via an `IMP_BY`-style edge.

Three rules for *our* capabilities:
1. **Value-framed, verb-object names** (`ingest-transcripts`, `ask-the-corpus`) — not
   package names. If the list ends up 1:1 with `src/`, it's just the code map re-skinned.
2. **Capabilities may cross packages, and a package may serve many** — e.g.
   `correct-the-analysis` spans `api/edits` + `commands` + `events`.
3. **Authored definition, guarded links** — the *what/for-whom* is human judgment
   (uncomputable from code); the check only *reconciles* the `implemented_by` links
   and coverage. `src/` is never touched — capabilities point **at** the code map.

## Nodes

One `docs/capabilities/<slug>.md` per capability (primary, child, or variant). Flat
files linked by `parent:` — OKF-idiomatic, matching `adr/` and `glossary/`. Frontmatter:

```yaml
---
type: Capability
kind: primary | child | variant
tier: core | enabling            # primaries carry tier; children/variants inherit
parent: enrich-fragments         # children/variants only; omit on primaries
implemented_by: [enrichment.executor, agents, models]
---
Classify each fragment's function, structure, and purpose — the analytic backbone
the workbench and lenses read.
```

- `kind`: **primary** (an L1 capability), **child** (a sub-capability — decomposition),
  **variant** (an alternative form of one capability, e.g. a provider failover chain or
  a per-lens extractor). One hierarchy (`parent:`), not two.
- `tier`: **core** (analyst-facing value) vs **enabling** (substrate). Primaries only;
  children inherit their parent's tier.
- `implemented_by`: CodeUnit slugs from the code map (`docs/code/`) — the traceability
  edge. Authored, reconciled by the check.
- Body: a one/two-line value statement (what it does, for whom).

**No use-case / `fulfills` edge this round** (round 3).

## The inventory (backfill target)

**tier: core**

| primary | children | implemented_by |
| --- | --- | --- |
| ingest-transcripts | parse-fragments · infer-speakers · stitch-utterances · segment-conversation | `ingestion`, `ingestion.orchestrator`, `ingestion.speaker_inference`, `ingestion.stitcher`, `agents` |
| enrich-fragments | extract-claims · classify-dimensions · tag-topics-keywords | `enrichment`, `enrichment.orchestrator`, `enrichment.executor`, `agents`, `models` |
| extract-insights-via-lenses | run-lens-engine · *(variant: per-lens extractors)* | `lens`, `lens.engine`, `agents` |
| resolve-entities-and-people | canonicalize-entities · resolve-persons · merge-split-link-alias | `resolution`, `resolution.engine` |
| correct-the-analysis | edit-text · rename-reattribute-speakers · remove-segments · override-lens-items · correct-resolution | `api`, `commands`, `events` |
| ask-the-corpus | hybrid-retrieval · cited-synthesis | `ask`, `ask.engine`, `ask.reader` |
| export-a-portable-bundle | assemble-bundle · render-bundle | `export`, `export.reader`, `export.renderer`, `export.bundler` |
| serve-workbench-and-gallery | run-read-queries · workbench-write · gallery-read · live-notifications | `api`, `ui`, `ui.reader` |

**tier: enabling**

| primary | note | implemented_by |
| --- | --- | --- |
| maintain-event-source-of-truth | append-only truth; frozen wire format | `commands`, `events` |
| project-events-to-graph | sole Neo4j writer; causal-order replay | `projections` |
| provider-strategy-and-focused-calls | *variants: chat-failover · pinned-embeddings* | `agents` |

~11 primaries + ~25 children/variants (comparable to the code map's 30 nodes). The
owner will review for gaps once it is in place.

## Generated artifacts

- **`docs/capabilities/index.md`** — the catalogue: grouped by `tier`, then primary,
  with children nested and each capability's `implemented_by` shown. Generated, never
  hand-edited.

(No separate Mermaid map this round — the capability→code edges are shown in the
index; a capability→code graph is a possible later drill-down.)

## The guard — `make capability-check` (non-blocking, exit 0)

1. **link resolution** — every `implemented_by` slug is a real CodeUnit (packages +
   key modules from `tools.code.reader`). An unknown slug → finding. (Reuses the code
   map's registry, exactly as ADR↔code reuses the scanner.)
2. **coverage** — a **pipeline-layer or surface** CodeUnit claimed by *no* capability
   → finding (an undocumented capability or dead code). **Infrastructure and model**
   units (`utils`, `models`, `io`, `persistence`) are **advisory-only** — listed if
   unclaimed, never flagged (they are shared substrate; `io`/`persistence` are known
   M3.0 legacy). Enabling capabilities cover `events`/`commands`/`projections`/`agents`.
3. **classification** — a capability with no `kind` or a primary with no `tier` →
   finding; a child/variant with no resolvable `parent:` → finding.
4. **index-sync** — committed `docs/capabilities/index.md` matches a fresh regeneration.

All non-blocking, `return 0`.

## Module design — new `tools/capability/`

Mirrors the established reader → render → check → CLI split. Module dir
`tools/capability/` (singular); make targets `capability-index` / `capability-check`.

- `reader.py` — `@dataclass Capability(slug, kind, tier, parent, implemented_by,
  statement, path)`; `load_capabilities(root, cap_dir="docs/capabilities")` (parse the
  authored node files); `real_code_units(root)` (packages + KEY_MODULES via
  `tools.code.reader`, the single source of truth for valid `implemented_by` targets).
- `render.py` — `render_index(caps)` (grouped tier → primary → children, with
  `implemented_by`). Pure.
- `check.py` — `Finding`; `check_links`, `check_coverage`, `check_classification`,
  `check_index_sync`, `run_all(root=".")`. Non-blocking.
- `__main__.py` — `python -m tools.capability {index|check}`.
- **Makefile** — `capability-index`, `capability-check` (self-documented `##`).
- **Registry + cascade** — add `("capabilities", "capability")` to
  `tools/knowledge/check.py`'s `DOMAINS`, and a row to `docs/index.md`. (The
  knowledge-check will flag both until done — dogfooding.)

`Capability` / `Finding` local to `tools/capability`.

## Backfill

Author the ~11 primaries + children/variants per the inventory: pick `kind`/`tier`,
set `parent:` and `implemented_by:` (slugs from `docs/code/`), write a one/two-line
value statement (mine `system-overview.md`, `data-flow.md`, and the code nodes for
accuracy). Then `make capability-index`; iterate `make capability-check` to clean.

## Testing

- **Unit** — `load_capabilities` parses a node + its links; `check_links` (unknown
  `implemented_by` slug → finding; valid → none); `check_coverage` (a pipeline/surface
  unit in no capability → finding; an unclaimed infra unit → NOT flagged);
  `check_classification` (missing `kind`/`tier`, unresolved `parent`); `check_index_sync`
  (stale index → finding). Assert **no check raises**. `render_index` groups by tier.
- **Smoke** — `make capability-index` writes the catalogue for the real inventory;
  `make capability-check` clean after backfill; `make knowledge-check` clean (cascade
  row + registry entry added); `make cli-check` clean after `cli-index` picks up the
  new `capability-*` targets.

## Non-goals (this round)

- **The `fulfills` / use-case edge** — round 3 (use-cases).
- **Reverse markers in `src/`** (a `capability:` docstring tag) — capabilities point
  at the code map; code stays untouched. (A reverse overlay is a possible later round.)
- **A capability→code Mermaid graph** — the index shows the edges; a graph is a later
  drill-down.
- **Forcing infra/model units into capabilities** — coverage is advisory for them.
- **Blocking** on any finding.

## Knowledge-graph check

Reviewed against `docs/index.md` on 2026-08-05.

| domain | touched? | consulted / reconciled | notes |
| --- | --- | --- | --- |
| code | yes (read-only) | `implemented_by` slugs drawn from `docs/code/` units; `tools.code.reader` is the link registry | no `src/` or code-node change |
| cli | yes (at impl) | new `capability-*` make targets → `cli-index` in the plan's backfill; `cli-check` clean | deferred to implementation |
| adr | yes (at impl) | capture ADR-0017 (adopt capabilities domain), `source:` = this spec | deferred to implementation |
| glossary | no | — | "capability" is a knowledge-domain concept, not code vocabulary pinned to an enum |
| api / prompts / graph-queries | no | — | no surface/prompt/query change |

**Verdict:** reconciled — the one live touch (code map, read-only) is consulted;
cli + adr reconciliation is carried into the implementation plan.
