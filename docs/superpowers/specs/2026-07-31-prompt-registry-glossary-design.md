# Prompt Registry (F) + Glossary Made Living & Extended (design)

**Status:** approved by owner 2026-07-31 (brainstorm dialogue).
**Program:** sub-project "F" of the *guarded knowledge graph over the codebase*,
done together with a revision + extension of the glossary (A) it reconciles against.
Follows A (glossary) and reuses the reader→render→check→CLI pattern.

## Goal

The prompts the product actually runs (`prompts/*.yaml`) enumerate the allowed values
for the classification / extraction vocabularies. The glossary (A) is a human-readable
**reference** that should track them. Today they have drifted (the glossary's
`purpose`, `topic_level_1`, `topic_level_3` values are wrong). Make the prompt registry
**catalogued and reconciled against the glossary**, and make the glossary a **living,
correct** vocabulary:

1. Correct and extend the glossary; frame it explicitly as a growing set.
2. A `docs/prompts/` catalog + a guard that flags when the glossary is out of sync
   with the registry (the operative source).

## Canonicity (locked in brainstorm)

- **The prompt registry is the operative source of truth** — it is what the product
  runs. The glossary is a reference that follows it. When a registry-pinned value set
  and the glossary disagree, the **fix path is registry → glossary** (update the
  glossary). For **code-pinned** vocabularies (`Enum`s, the `claim` `Literal`) the code
  is truth (checked by A's existing enum reconciliation, extended here).
- **The domain is living.** The glossary grows in both *type* and *amount*; the model
  and guards must not assume a closed universe. Adding a new vocabulary kind must not
  require tooling changes.

## The vocabulary, by where truth lives (confirmed against code + prompts)

| Term(s) | Truth | Values |
|---|---|---|
| 7 `str,Enum` classes | **code** (A already) | from the enum |
| `claim-kind` | **code** (`Literal` in `src/models/extractor_responses.py`) + prompt (agree) | assertion, commitment, request |
| `function_type` / `structure_type` | **registry** (`core_extractors.yaml`) | 4 / 4 (grammatical) |
| `purpose` | **registry** | 24 (Statement, Query, …) |
| `topic_level_1` / `topic_level_3` | **registry** | 15 each |
| `entity-type` | **registry** (prompt Format line; no code enum) | person, organization, product, tool, other |
| `overall_keywords` / `domain_keywords` | — | open-ended (no closed set) |

**Live source** = `core_extractors.yaml` (loaded by `src/enrichment/orchestrator.py`).
`task_prompts.yaml` is the **legacy** predecessor (`core_extractors` says "Ported
from task_prompts.yaml"); `domain_prompts.yaml` is unreferenced reference data.

## Part 1 — Glossary: living, corrected, extended

- **`docs/glossary/README.md`** — states plainly: this is a *growing* reference,
  expanding in type and amount; absence of a term means "not yet catalogued," not
  "doesn't exist"; `kind` is free-form, new kinds welcome.
- **Corrections** (registry → glossary): update `docs/glossary/purpose.md` (24 values),
  `topic-level-1.md` + `topic-level-3.md` (their real 15-value sets, read from the
  prompt). Their `kind` stays `dimension`.
- **New terms:**
  - `claim-kind` (`kind: claim-kind`, `source: src/models/extractor_responses.py`,
    values assertion/commitment/request) — **code-pinned**.
  - `entity-type` (`kind: entity-type`, `source: prompts/core_extractors.yaml`,
    values person/organization/product/tool/other) — **registry-pinned**.
- **`tools/glossary` extension:** add a small AST reader `code_literals(root)` that
  finds `field: Literal[...]` closed sets (v1: the `claim` `kind`) and returns them as
  `CodeTerm`s, so A's existing `check_enum_values` code-checks `claim-kind` exactly
  like an enum. The prompt-pinned terms (dimensions, `entity-type`) remain authored and
  are reconciled by F (Part 2), not by A's code checks.

## Part 2 — F: the prompt-registry guard (`tools/prompts/`)

- **Reader** — loads `prompts/*.yaml`; classifies each entry by the loading stage
  (ingestion / enrichment / lens / ask / legacy) via a small static map of file→stage;
  extracts an entry's enumerated value set when present, handling **both** shapes:
  - `Options:` bullet lists (`function_type`, `structure_type`, `purpose`, `topic_*`);
  - `"field": "a|b|c"` Format-line pipes (`claim` kind, `entity_type`).
  Returns `PromptEntry(file, key, stage, values)`.
- **Render** — `docs/prompts/index.md`: prompts grouped by stage (file · key), a map
  for agents. Generated, never hand-edited.
- **Check — `make prompt-check` (non-blocking):**
  1. **registry↔glossary reconciliation** (headline) — for each prompt entry whose key
     maps to a glossary term (via a small key→term map: `purpose`→purpose,
     `entity_mentions`→entity-type, `claims`→claim-kind, …), compare the extracted
     values to the glossary term's `values`. Mismatch → `glossary term X is out of sync
     with the registry (prompts/…): missing …, extra … — update the glossary`.
     (Bidirectional detection; the message names the glossary as the thing to fix.)
  2. **orphan enumerated prompt** — an enumerated prompt entry mapping to no glossary
     term → informational (a candidate to catalogue — the growth path).
  3. **catalog-in-sync** — `docs/prompts/index.md` matches the live prompts.
  4. **legacy/unused** (informational) — `task_prompts.yaml` present (legacy);
     `domain_prompts.yaml` unreferenced.
- **CLI / Makefile** — `python -m tools.prompts {index|check}`; `make prompt-index`,
  `prompt-check` (self-documented per D's `##` convention).

`PromptEntry` / `Finding` local to `tools/prompts`. Reads the glossary via
`tools.glossary` (`load_glossary`).

## Module design

```
tools/prompts/
  reader.py    # load prompts/*.yaml, stage map, value extraction (Options + Format-line)
  render.py    # render_catalog(entries) -> str
  check.py     # reconcile-with-glossary, orphan, catalog-sync, legacy; run_all(root=".")
  __main__.py  # index | check
tools/glossary/
  reader.py    # + code_literals(root) for Literal[...] closed sets (claim-kind)
docs/glossary/
  README.md, purpose.md, topic-level-1.md, topic-level-3.md (fixed), claim-kind.md, entity-type.md (new)
docs/prompts/index.md          # generated catalog
Makefile                        # + prompt-index, prompt-check
```

## Testing

- **Glossary (Part 1)** — `code_literals` extracts a `Literal[...]` set from a synthetic
  model; `check_enum_values` now flags a `claim-kind` mismatch; the fixed/new term files
  parse; `glossary-check` clean on the real repo after the fixes.
- **F (Part 2)** — `reader` value extraction over both prompt shapes (a bulleted
  `Options:` entry and a `"field": "a|b|c"` Format entry) + a no-enumeration entry;
  stage classification; `render_catalog` shape; `check` reconciliation (a matching set →
  no finding; a value the prompt has but the glossary lacks → finding naming the
  glossary; an orphan enumerated prompt → informational; catalog out of sync). Assert
  **no check raises**. Smoke: `make prompt-check` on the real repo is clean **after** the
  Part-1 glossary fixes (proving the loop closes).

## Non-goals (this round)

- **Lens / ask prompt value-reconciliation** — catalogued only (no closed value sets to
  check against yet).
- **Prompt versioning / provenance**, **governed-by ADR links**.
- **Deleting the legacy `task_prompts.yaml`** — flagged, not removed (owner's call).
- **Node labels / graph vocabulary** — the tracked graph-query sub-project.
- **Blocking** on any finding.
