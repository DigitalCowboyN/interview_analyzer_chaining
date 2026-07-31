# Glossary / Taxonomy Domain — hybrid generate + author (design)

**Status:** approved by owner 2026-07-31 (brainstorm dialogue).
**Program:** sub-project "A" of the *guarded knowledge graph over the codebase* —
the foundational vocabulary layer. Unblocks F (prompt registry) and the tracked
graph-query sub-project. Reuses the reader→render→check→CLI pattern of the ADR/CLI/API
domains.

## Goal

The project's vocabulary — the enums and the analysis dimensions — is real but has
no single home, and the values a dimension may take (`purpose: statement/question/…`)
live only in the prompts, not in code. Make the vocabulary **catalogued, defined,
and drift-checked**:

1. A `docs/glossary/` bundle: one authored entry per term, with its definition and
   its allowed values.
2. A guard that keeps the glossary honest against the code that defines the terms —
   and holds the canonical **dimension values** that F will check the prompts against.

## Design decisions (locked in brainstorm)

- **Hybrid: generate the skeleton, author the meaning.** The term *list* and, for
  enums, the *values* are read from code (AST); the definitions and the *dimension
  values* are authored. (Rejected: fully authored — lags code; rejected:
  generated-only — can't hold dimension values, which aren't in code, so it wouldn't
  unblock F.)
- **v1 scope = enums + dimensions.** Node labels / relationship types (scattered in
  Cypher) and lens `node_type`s (per-lens config) are deferred to the graph-query
  sub-project. (Owner accepted the wait to avoid fragile Cypher-scraping now.)
- **Asymmetric value reconciliation.** Enum term `values` are checked against the
  code enum (code is truth). Dimension term `values` are **authored canonical** —
  code has no source for them, and the glossary becomes that source (what F reconciles
  prompts against).
- **AST-based reader, no import.** The reader parses source with `ast` (stdlib) — it
  never imports the app (no env/deps needed; safe and fast).

## v1 vocabulary (confirmed against code)

- **Enums (7):** `TranscriptFormat` (`src/ingestion/models.py`), `EditorType` /
  `TagType` / `SentenceStatus` (`src/events/sentence_events.py`), `InterviewStatus`
  (`src/events/interview_events.py`), `ActorType` / `AggregateType`
  (`src/events/envelope.py`). Members auto-read (e.g. `ActorType` = HUMAN/SYSTEM/AI).
- **Dimensions (7):** `function_type`, `structure_type`, `purpose`, `topic_level_1`,
  `topic_level_3`, `overall_keywords`, `domain_keywords` — Pydantic fields on
  `src/models/analysis_result.py`. Names auto-read; values authored.

## Structure — `docs/glossary/` bundle

One authored markdown file per term (OKF-conformant):

```yaml
---
type: Term
term: purpose
kind: dimension            # enum | dimension
source: src/models/analysis_result.py
values: [statement, question, ...]   # enum: must equal code members; dimension: authored
---
The communicative purpose of a fragment: whether it states, asks, proposes, …
```

```yaml
---
type: Term
term: ActorType
kind: enum
source: src/events/envelope.py
values: [HUMAN, SYSTEM, AI]           # reconciled against the code enum
---
Who caused an event — used across the edit-observability taxonomy.
```

Plus a **generated** `docs/glossary/index.md` (term · kind · source), reserved/generated
like the ADR index (guarded for sync).

## The guard — `make glossary-check` (non-blocking)

`python -m tools.glossary check` returns findings; **never raises, always exits 0**:

1. **coverage** — every code enum and every dimension field has a glossary term file;
   missing → `code defines enum ActorType (src/events/envelope.py) with no glossary term`.
2. **enum-value reconciliation** — for `kind: enum` terms, the file's `values` must
   equal the code enum's members; mismatch → `glossary term ActorType values differ
   from the code enum (added: …, removed: …)`. (Dimension terms: values authored, not
   code-checked.)
3. **stale term** — a term whose `source` no longer defines it (renamed/removed enum
   or dimension) → `glossary term X: source no longer defines it`.
4. **index-in-sync** — `docs/glossary/index.md` matches the term files.

## Module design — new `tools/glossary/`

AST-based, stdlib-only, mirrors the established split.

- `reader.py` — `code_enums(root) -> dict[name, CodeTerm]` (AST-scan `src/` for
  `class X(...Enum)`, collect member names + source path); `code_dimensions(root) ->
  dict[name, CodeTerm]` (AST-parse `src/models/analysis_result.py` for the model's
  field names). No import.
- `model.py` — `@dataclass Term(term, kind, source, values, definition, path)`;
  `parse_term(text, path)` (reuses `parse_front_matter`); `load_glossary(dir)`.
- `check.py` — `@dataclass Finding`; the four checks + `run_all(root=".")`.
- `render.py` — `render_index(terms) -> str`.
- `scaffold.py` — `new_term(name, kind, root)` pre-fills frontmatter from code
  (enum → members as `values`; dimension → empty `values` to author).
- `__main__.py` — `python -m tools.glossary {index|check|scaffold}`.
- **Makefile** — `glossary-index`, `glossary-check` (self-documented, per D's convention).

`Term` / `Finding` are local to `tools/glossary`.

## Backfill

Author the ~14 term files: run `glossary scaffold` per term, then fill definitions
(and, for dimensions, the allowed values from the prompts). `make glossary-index`;
`make glossary-check` clean.

## Testing

- **Unit** — `code_enums` / `code_dimensions` over a synthetic `src` tree (an enum
  with members, a Pydantic model with fields, a non-enum class ignored); `parse_term`;
  each check fires on a fixture (code enum with no term; enum-value mismatch;
  stale source; index out of sync); `render_index` shape. Assert **no check raises**.
- **Smoke** — `glossary scaffold ActorType` pre-fills the real enum's members; after
  backfill, `make glossary-check` on the real repo is clean.

## Non-goals (v1)

- **Node labels / relationship types / lens `node_type`s** — deferred (the
  graph-query sub-project extracts them; the glossary extends then).
- **Importing code** to read enums — AST only.
- **Governed-by / ADR links** for terms (possible v-next overlay).
- **Checking dimension values against the prompts** — that is F's job (prompt
  registry); the glossary only *holds* the canonical values here.
- **Blocking** on any finding.
