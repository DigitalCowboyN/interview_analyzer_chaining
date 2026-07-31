# Probabilistic-Components Registry (F) + Glossary Made Living & Extended (design)

**Status:** approved by owner 2026-07-31 (brainstorm dialogue, two rounds).
**Program:** sub-project "F" of the *guarded knowledge graph over the codebase*, done
together with a revision + extension of the glossary (A) it reconciles against.

## The reframe: prompts are probabilistic code components

A prompt is **code** — versioned logic a component executes — so it is documented,
linked, and guarded like code (the B/C/D machinery). But unlike the rest of the
codebase it is **probabilistic**: an LLM runs it; output is non-deterministic. So it
gets the same treatment as code **with a classification flag** distinguishing it.

This introduces a **determinism axis** to the whole graph: **deterministic code**
(functions, routes, CLI — reconciled *exactly*) vs **probabilistic components**
(prompts — reconciled, but understood to behave stochastically). v1 realises the axis
by making the prompt domain carry `classification: probabilistic`; retrofitting a
`deterministic` tag onto B/C/D is out of scope (future).

## What each prompt carries (the facet model)

| Facet | Meaning | Source |
|---|---|---|
| `classification` | `probabilistic` (vs deterministic code) | intrinsic |
| `used_for` | the capability: extraction / classification / segmentation / ingestion / synthesis / lens | **authored in the prompt YAML** |
| `audience` | the consumer roles it is *enabled for* | **authored in the prompt YAML** (internal roles + external) |
| `consumers` | the code that *actually* loads/runs it | **derived from code** |
| `values` | enumerated closed set (if any) | the prompt → **reconciled to the glossary (A)** |

### Audience taxonomy

Audience is a set spanning two kinds, and it **grows** as a prompt is reused:

- **Internal roles** (verifiable — map to code): `ingestion`, `enrichment`,
  `lens`, `ask`, `api`, `agent`.
- **External roles** (declared-only — not in this repo's code): `cli`, `skill`,
  `coding-harness`, `other`.

The prompt domain therefore sits at the **center** of the graph: a prompt links to the
**code that runs it (B)**, the **API that exposes it (C)**, and the **CLI/harness that
invokes it (D)**. `used_for` + `audience` are authored; `consumers` are derived; the
guard reconciles the two.

## Canonicity (locked)

- **The prompt registry is the operative source of truth** for the value vocabularies;
  the glossary follows. Value mismatch → fix path is **registry → glossary**. For
  **code-pinned** vocabularies (`Enum`s, the `claim` `Literal`) the code is truth.
- **The domain is living** — the glossary grows in type and amount; model + guards must
  not assume a closed universe; adding a new vocabulary kind needs no tooling change.

## The value vocabulary, by where truth lives (confirmed vs code + prompts)

| Term(s) | Truth | Values |
|---|---|---|
| 7 `str,Enum` classes | code (A already) | from the enum |
| `claim-kind` | code (`Literal` in `extractor_responses.py`) + prompt (agree) | assertion, commitment, request |
| `function_type` / `structure_type` | registry (`core_extractors.yaml`) | 4 / 4 |
| `purpose` | registry | 24 |
| `topic_level_1` / `topic_level_3` | registry | 15 each |
| `entity-type` | registry (prompt Format line) | person, organization, product, tool, other |
| `overall_keywords` / `domain_keywords` | — | open-ended |

Live source = `core_extractors.yaml` (loaded by `src/enrichment/orchestrator.py`).
`task_prompts.yaml` is legacy (ported → core_extractors); `domain_prompts.yaml` is
unreferenced reference data.

## Part 1 — Glossary: living, corrected, extended

- **`docs/glossary/README.md`** — states plainly: a *growing* reference (type + amount);
  absence ≠ nonexistence; `kind` is free-form.
- **Corrections** (registry → glossary): `purpose` → 24; `topic_level_1` / `topic_level_3`
  → their real 15 sets (read from the prompt). `kind` stays `dimension`.
- **New terms:** `claim-kind` (`source: src/models/extractor_responses.py`, **code-pinned**)
  and `entity-type` (`source: prompts/core_extractors.yaml`, **registry-pinned**).
- **`tools/glossary` extension:** `code_literals(root)` — an AST reader for
  `field: Literal[...]` closed sets (v1: the `claim` `kind`) returned as `CodeTerm`s, so
  A's existing `check_enum_values` code-checks `claim-kind` like an enum.

## Part 2 — Probabilistic-components registry (`tools/prompts/`)

### Prompt metadata (authored in `prompts/*.yaml`)

Add two keys to each prompt entry (a one-time backfill across the live files):

```yaml
function_type:
  used_for: [classification]      # capability
  audience: [enrichment]          # roles it is enabled for (internal + any external)
  prompt: "Determine the function type…"
```

### Reader

- Loads `prompts/*.yaml`; per entry yields `PromptEntry(file, key, used_for, audience,
  values, consumers)`.
- **values**: extracts the enumerated set when present, handling **both** shapes —
  `Options:` bullet lists and `"field": "a|b|c"` Format-line pipes.
- **consumers (derived)**: a `file → loading-module(s)` map built by scanning `src/` for
  `"prompts/<file>"`; each module maps to an internal role via a small module-prefix→role
  map (`src/enrichment`→enrichment, `src/ingestion`→ingestion, `src/ask`→ask,
  `src/lens`→lens, `src/api`→api). A file with no loader → no consumers.

### Catalog

`docs/prompts/index.md` — generated, never hand-edited: each prompt as
`classification · key · used_for · audience · consumers · values`, grouped by file/stage.
The complete probabilistic-component map for agents.

### The guard — `make prompt-check` (non-blocking, always exit 0)

1. **values ↔ glossary** (registry → glossary) — enumerated prompt values vs the mapped
   glossary term (`purpose`→purpose, `claims`→claim-kind, `entity_mentions`→entity-type,
   …). Mismatch → `glossary term X out of sync with the registry — update the glossary`.
2. **audience ↔ consumers** — for each declared **internal** audience role, a derived
   consumer of that role must exist; declared-but-absent → `declares audience <role> but
   no code consumes it`. A derived consumer role **not** in the declared audience →
   `consumed by <role> but audience does not list it`. **External** roles (cli/skill/
   harness) are surfaced, not reconciled (not in-repo).
3. **orphan / unused** — a prompt with no derived consumers and no external audience →
   `unused (no consumer)` (catches `task_prompts.yaml`, `domain_prompts.yaml`).
4. **missing metadata** (informational) — a prompt with no `used_for` or `audience`.
5. **catalog-in-sync.**

### CLI / Makefile

`python -m tools.prompts {index|check}`; `make prompt-index`, `prompt-check`
(self-documented per D's `##` convention).

## Module design

```
tools/prompts/
  reader.py    # load prompts/*.yaml; value extraction (2 shapes); consumer derivation (file->module->role)
  render.py    # render_catalog(entries) -> str
  check.py     # values<->glossary, audience<->consumers, orphan, missing-metadata, catalog-sync; run_all(root=".")
  __main__.py  # index | check
tools/glossary/reader.py   # + code_literals(root) for Literal[...] (claim-kind)
prompts/*.yaml             # + used_for + audience on each live entry (backfill)
docs/glossary/             # README.md; purpose/topic-level-1/topic-level-3 fixed; claim-kind, entity-type new
docs/prompts/index.md      # generated catalog
Makefile                   # + prompt-index, prompt-check
```

`PromptEntry` / `Finding` local to `tools/prompts`; reads the glossary via
`tools.glossary.load_glossary`.

## Testing

- **Glossary (Part 1)** — `code_literals` extracts a `Literal[...]` set from a synthetic
  model; `check_enum_values` flags a `claim-kind` mismatch; the fixed/new term files
  parse; `glossary-check` clean on the real repo after the fixes.
- **F (Part 2)** —
  - reader: value extraction over both shapes + a no-enumeration entry; `used_for`/
    `audience` read from YAML; consumer derivation over a synthetic `src` tree + prompt
    file (a file with a loader → that role; a file with none → no consumers).
  - `render_catalog` shape.
  - checks: values-reconcile (match → none; prompt-has/glossary-lacks → finding naming
    the glossary); audience-reconcile (declared internal role with a consumer → none;
    declared-but-absent → finding; consumed-but-undeclared → finding; external role →
    not reconciled); orphan (no consumer → finding); catalog-sync. Assert **no check
    raises**.
  - Smoke: `make prompt-check` on the real repo is clean **after** the Part-1 glossary
    fixes + the metadata backfill (values reconcile; each live prompt has a consumer +
    `used_for`; task/domain flagged unused).

## Non-goals (this round)

- **Retrofitting a `deterministic` classification onto B/C/D** — the axis is introduced
  via the prompt domain only.
- **Key-level consumer derivation** — consumers are derived per prompt *file* (the loader
  loads the whole file); call-graph analysis to attribute individual keys is out of scope.
- **External-audience verification** — cli/skill/harness are declared and surfaced, not
  reconciled (not in this repo).
- **Prompt versioning / provenance**, **governed-by ADR links**, **deleting legacy
  `task_prompts.yaml`** (flagged, owner's call).
- **Lens / ask value-reconciliation** (no closed value sets), **node/graph vocabulary**
  (graph-query sub-project), **blocking**.
