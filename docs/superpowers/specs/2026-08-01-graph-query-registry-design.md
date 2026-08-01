# Graph-Query Registry + Glossary Graph-Vocabulary (design)

**Status:** approved by owner 2026-08-01 (thorough two-round brainstorm).
**Program:** the graph-query sub-project of the *guarded knowledge graph over the
codebase*. Completes the graph: the write side (projections) defines the schema
vocabulary → the glossary (A) grows to hold it → the read queries reconcile against
it. Reuses the reader→render→check→CLI pattern.

## The framing (locked in brainstorm)

- Graph queries are their own domain — **deterministic but schema-coupled** components
  (the other pole from F's probabilistic prompts). They are used by **code and agents**,
  are both **impacted by code and impact code**, and are driven partly by architecture,
  partly by reporting/UI need.
- **Write = schema source; read = the registry.** The projection/write queries
  (`src/projections/`) *define* the graph vocabulary (labels / rel types / properties).
  The glossary grows to hold that vocabulary. The **read-query bundles** (export / ui /
  ask / resolution) are the registry that reconciles against it.
- All three couplings are in scope this round: **query→schema drift**, **query→consumers
  output-contract**, and **scope/shape classification**.

## The vocabulary (confirmed against `src/projections/`)

- **~20 node labels** (Fragment, Claim, Speaker, Interview, Entity, LensItem, Person, …).
- **~17 relationship types** (SPOKE, MENTIONS, MADE_BY, HAS_SENTENCE, SUPPORTED_BY, …).
- **~27 handler-written properties** (interview_id, claim_id, confidence, locked, method,
  node_type, …). Owner chose **all handler-written props**, not just constrained ones.

Source of truth = `src/projections/schema.py` (constraints/indexes → labels + constrained
props) **plus** the handler `MERGE`/`CREATE`/`SET` patterns (rel types + all written props).

## G1 — Glossary grows to hold the graph vocabulary

New free-form `kind`s: `graph-label`, `rel-type`, `graph-property`. (The glossary is
already living/open — no tooling gymnastics to add kinds.)

- **`tools/glossary` extension** — `graph_vocabulary(root) -> dict[str, CodeTerm]`:
  regex-scan `src/projections/**` for labels (`(:Label)`, `FOR (n:Label)`), rel types
  (`[:REL_TYPE]`), and properties (`n.prop =`, `SET n.prop`, constraint `REQUIRE n.prop`).
  Returns `CodeTerm`s keyed by name, `kind` = graph-label / rel-type / graph-property.
- **Reconciliation** (code-pinned, existence only — labels/rels/props have no value-set):
  extend `check_coverage` (a write-side vocab item with no glossary term → finding) and
  `check_stale_source` (a `graph-*` glossary term not in the write-side extraction →
  stale). `check_enum_values` does not apply.
- **Backfill** — author the ~64 terms with **terse** definitions (a graph-property term may
  be one line). Living-domain README already covers "grows in type and amount."

## G2 — Graph-query registry (`tools/graphq/`)

The **read-query bundles** = the reader modules (`src/export/reader.py`,
`src/ui/reader.py`, `src/ask/reader.py`, `src/resolution/reader.py`) + any inline query
(e.g. `src/api/routers/segments.py`). Each **named query** is a function whose body holds a
Cypher string.

- **Authored metadata — a `graphq:` docstring marker** on each query function:
  ```python
  def worklist_rows(...):
      """Low-confidence review queue.

      graphq: purpose=export scope=domain-broad audience=[export, api]
      """
  ```
  `purpose` (export / ui / ask / resolution / projection), `scope` (task | domain-broad),
  `audience` (roles: internal code roles + agents).
- **Parsed from the Cypher** — `labels`, `rels`, `props` (schema deps) and `returns`
  (the `RETURN … AS <alias>` output fields).
- **Derived** — `consumers`: the modules/functions that call this query function.
- `QueryEntry(bundle, name, purpose, scope, audience, labels, rels, props, returns, consumers)`.

**`docs/graph-queries/index.md`** — generated catalog, queries grouped by bundle, each row:
`name · purpose · scope · audience · consumers · schema-deps · output-fields`.

## G3 — The guards (`make graphq-check`, non-blocking, exit 0)

1. **query→schema drift** — every `label`/`rel`/`prop` a read query references must be in
   the glossary graph-vocabulary (write-side truth). Referenced-but-absent →
   `graph-queries: <bundle>.<name> references label :X not produced by any projection`.
2. **query→consumers output-contract** — for each query, scan its **direct callers** for
   row-field accesses (`row["x"]`, `r.get("x")`, `row.get("x")`) and flag any field not in
   the query's `RETURN` aliases → `<caller> reads field 'x' not returned by <bundle>.<name>`.
   Bounded to direct callers; heuristic on access patterns (documented limitation).
3. **scope/shape + missing metadata** — surface each query's `scope`; a query function with
   Cypher but no `graphq:` marker → informational (author it).
4. **catalog-in-sync** — `docs/graph-queries/index.md` matches the live queries.

## Module design

```
tools/glossary/reader.py     # + graph_vocabulary(root) (labels/rels/props from src/projections)
tools/glossary/check.py      # coverage/stale extended for graph-* kinds
tools/graphq/
  reader.py    # find query functions (AST); parse Cypher (labels/rels/props/returns);
               #   read graphq: markers; derive consumers (callers)
  render.py    # render_catalog(entries) -> str
  check.py     # query->schema-drift, output-contract, missing-marker, catalog-sync; run_all
  __main__.py  # index | check
docs/glossary/               # + ~64 graph-* term files (terse); README already living
docs/graph-queries/index.md  # generated catalog
Makefile                     # + graphq-index, graphq-check (self-documented per D)
```

`QueryEntry` / `Finding` local to `tools/graphq`; reads the glossary via `tools.glossary`.

## Cypher parsing (regex, no Cypher engine)

- labels: `[(\[]\s*\w*:(\w+)` (node + rel-pattern label positions).
- rel types: `\[\s*\w*:([A-Z_]{3,})`.
- props referenced by a query: `\.\s*(\w+)` on bound variables (best-effort) + explicit
  `{prop:` map keys — properties are the noisiest to parse; v1 extracts what it can and the
  schema-drift check tolerates misses (a missed prop simply isn't checked, never a false
  drift). labels/rels are the reliable, high-value part.
- returns: parse the `RETURN` clause, capture `AS (\w+)` aliases (and bare `x.y` fallbacks).

## Testing

- **Glossary (G1)** — `graph_vocabulary` over a synthetic `src/projections` tree (a label,
  a rel type, a prop); `check_coverage`/`check_stale_source` handle `graph-*` kinds
  (uncovered label → finding; stale graph-term → finding); real `glossary-check` clean after
  backfill. No check raises.
- **Graphq (G2/G3)** — reader: find a query function, parse its labels/rels/props/returns,
  read a `graphq:` marker, derive a caller; render shape; each guard on a fixture
  (drift: query references an unknown label → finding; output-contract: a caller reads a
  non-returned field → finding; missing marker → finding; catalog sync). Assert **no check
  raises**. Smoke: `make graphq-check` on the real repo — after backfilling markers + graph
  vocab, clean except any genuine drift it legitimately finds (report those as real).

## Non-goals (this round)

- **A real Cypher parser** — regex extraction; property parsing is best-effort (misses are
  silent non-drift, never false positives).
- **Transitive consumer analysis** — output-contract scans direct callers only.
- **Write-query registry** — the write side is treated as the schema *source*, not
  catalogued as queries (its "registry" is the projection handlers + ADR-0003 already).
- **Governed-by ADR links** for queries, **query performance/EXPLAIN**, **blocking**.
