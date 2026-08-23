# KG-2 — derived event-and-label flow overlay (design)

**Status:** proposed (brainstorm dialogue with owner, 2026-08-23).
**Program:** the first-class knowledge graph
(`docs/superpowers/specs/2026-08-15-first-class-knowledge-graph-program-design.md`); milestone **KG-2**
in `docs/superpowers/kg-program-roadmap.md`. Closes the behavioral seams the KG-1 agentic eval proved
invisible — the event-sourced write path and the analysis pipeline — plus the verified schema-lineage
gap. **Measured target:** the eval's `pipeline-write-path` / `pipeline-ingestion-flow` (`gap`, recall
0.25) and `deploy-neo4j-schema-blast` (`partial`) scenarios should rise on re-run.

## The problem (proven, not hypothesized)

KG-1's agents showed — and I verified — that three behavioral seams have **no graph edge**, because
the graph models Python imports and the coupling here is *events* and *Neo4j labels*, not imports:

- **Write path:** `command_handler → [event] → projection_handler → [Neo4j] → api`. `depends_on` never
  connects the append side to the projection side (they only share an import of `events.store`).
- **Pipeline:** `ingestion → enrichment → lens → export`. Only `ingestion → enrichment` is a real edge;
  the rest is event- and read-model-mediated.
- **Schema lineage:** `projections.schema → read-consumers`. Verified: `walk(projections.schema, in,
  full)` reaches 58 nodes and **zero** api/export read-consumers — the link is a Neo4j-label string
  match, not an edge.

## The key insight — the flow is already latent in existing nodes/metadata

The coupling is mediated by **events** and **Neo4j labels**, and both already exist as graph nodes:

- **Event types are symbol nodes** — the 44 `events.*Data` payload classes (e.g.
  `code:events.sentence_events.SentenceEditedData`).
- **Neo4j labels are glossary-term nodes** — 19 `GlossaryTerm`s `defined_in → code:projections.schema`
  (`glossary:Fragment`, `glossary:Interview`, …).
- **Aggregates already call their event constructors** — verified: `events.aggregates.add_speaker`
  `calls` `SpeakerCreatedData`, etc. (the existing symbol-grain `calls` edge already captures emission).
- **The handler registry is explicit** — `registry.register("InterviewCreated", InterviewCreatedHandler(...))`.
- **Graph-queries already declare their labels** — `graph-query` nodes carry `labels=[...]` metadata
  (e.g. `reader.name_rows: labels=['CanonicalEntity','Entity','Person']`).

So KG-2 is a **mostly-derived overlay** that connects things that already exist — not an authored flow
domain. (Owner decision, 2026-08-23: hybrid, derive-first; markers only for the irreducible.)

## Decision — four derived edges (no new node types)

Add four derived edge types to the registry (`tools/graph/registry.py`); reuse event-class symbols and
glossary-term labels as endpoints:

1. **`emits` / `emitted_by`** (CodeUnit symbol → CodeUnit event-class symbol). **Derived** as the
   subset of the existing symbol `calls` where the callee is an `events.*` payload class (name ends
   `Data`, module under `events.`). Free from Task-3 (`calls_of`) already computed. So an aggregate
   method that constructs `SentenceEditedData(...)` gets an `emits` edge to that event node.
   - *Marker fallback:* `# emits: <EventClassDottedId>` (sibling of `# calls:`) for emission our
     pragmatic call-resolution misses (dynamic construction, `_apply(cls(**data))`).

2. **`handled_by` / `handles`** (event-class symbol → projection-handler-class symbol). **Derived** by
   parsing `registry.register("<Type>", <HandlerClass>(...))` in `projections.bootstrap` (and any
   sibling registration site): the string `"<Type>"` bridges to the payload class `"<Type>Data"` by
   convention; the handler class resolves to its symbol node. A registration whose event string has no
   matching `…Data` class is reported (a small check), not silently dropped.

3. **`writes` / `written_by`** (projection-handler-class symbol → GlossaryTerm label). **Derived** by
   scanning the handler's source for Cypher `MERGE (<var>:<Label>` / `CREATE (<var>:<Label>` and
   mapping `<Label>` to `glossary:<Label>` when that term exists (the labels are already glossary
   terms `defined_in` the schema).

4. **`reads` / `read_by`** (GraphQuery → GlossaryTerm label). **Derived** from the existing graph-query
   `labels=[...]` metadata: each label string → `glossary:<Label>`. Free (metadata already parsed).
   Graph-queries are already `consumed_by` code, so `read_consumer → graph_query → :Label` composes.

**Result — the full path becomes walkable** (mostly at symbol grain, lazily):

```
command_handler --calls--> aggregate_method --emits--> Event
   --handled_by--> projection_handler --writes--> :Label(glossary)
        <--reads-- graph_query <--consumed_by-- read_consumer (api/export)
```

An agent tracing the write path walks `command → aggregate → event → handler → label`, then across the
label to `graph_query → api/export`. Schema blast-radius = walk a label's `written_by` (who populates
it) and `read_by`/`reads`⁻¹ (who consumes it) — closing schema-gap #2 structurally.

## Grain, laziness, and where these live

- `emits`, `handled_by`, `writes` are **symbol-grain** (their endpoints are symbol nodes) — they
  materialize only under a `level="symbol"` walk, on the lazy frontier (ADR-0027). No eager cost; a
  module-grain walk is unchanged (harvest-equivalence preserved).
- `reads` connects `GraphQuery` → `GlossaryTerm`, both harvest-grain — so `reads` is available in
  `harvest()`/module-grain (cheap, static metadata).
- **`emits` and `handled_by` are walk-time/lazy** (like `calls`): NOT added to the eager `harvest()`
  edge set, to avoid parsing all aggregates/handlers on every catalog build. `reads` (cheap, metadata)
  and `writes` (needs handler-Cypher parse) — `reads` can be harvested; `writes` is symbol-lazy.

## Fidelity ceiling (documented, like `calls`)

- **`emits`** inherits `calls`'s ceiling — statically-resolved constructor calls only; dynamic/`_apply`
  emission needs the `# emits:` marker. Under-linking, not false edges.
- **`handled_by`** relies on the `register("Type", Handler)` shape and the `<Type>Data` convention; a
  registration not matching either is flagged, not guessed.
- **`writes`** is a Cypher-label regex over handler source — catches literal `MERGE (:Label`; a
  handler that builds a label dynamically is missed (flagged if it writes Neo4j but yields no label).
- **`reads`** is only as complete as the graph-query `labels=[...]` metadata (already authored per
  query); a raw-Cypher read outside the `graph-queries` registry is invisible (a known pre-existing
  blind spot, noted).

## Scope

**This milestone (KG-2):** the four derived edges (+ `# emits:` marker); registry entries; walk-time
wiring for the symbol-lazy ones and harvest wiring for `reads`; non-blocking checks (registration
without a matching event class; a handler that writes Neo4j but resolves no label); re-run the KG-1
`pipeline-*` and `deploy-neo4j-schema-blast` scenarios to record the lift; a new ADR.

**Deferred:** authored *prose* flow narratives (not needed — the overlay is derived); the
ingestion→enrichment→lens→export **stage ordering** as an explicit sequence (it emerges from
emit/handle/read chains through the event+read-model, which is the honest structure — a named
"pipeline" node is a later nicety); raw-Cypher reads outside the graph-query registry; KG-3
infra/deployment topology (still its own milestone).

## Testing

- **`emits` derivation:** a fixture aggregate method constructing `FooData()` yields
  `emits → code:events.…​.FooData`; a dynamic-emit method with a `# emits:` marker yields the marked
  edge; a method constructing a non-event class yields none.
- **`handled_by`:** a fixture `register("Foo", FooHandler(...))` + a `FooData` class yields
  `code:…​.FooData --handled_by--> code:…​.FooHandler`; a register with no matching `…Data` is flagged.
- **`writes`:** a handler with `MERGE (n:Bar ...)` and a `glossary:Bar` term yields
  `handler --writes--> glossary:Bar`; a label with no glossary term is skipped (reported).
- **`reads`:** a graph-query with `labels=['Bar']` yields `graph_query --reads--> glossary:Bar`.
- **End-to-end walk:** on the real repo, a `level="symbol"` walk from a command handler reaches its
  aggregate → event → handler; a walk from `glossary:Fragment` reaches both its `written_by` handlers
  and `read_by` graph-queries (schema blast-radius traversable).
- **No dangling / equivalence:** every new edge endpoint resolves; module-grain `walk` unchanged
  (harvest-equivalence still green); freshness clean.
- **Eval lift:** re-running `pipeline-write-path`, `pipeline-ingestion-flow`, `deploy-neo4j-schema-blast`
  agentically, the flow is now traversable — recorded in `evals/graph/RESULTS.md`.

## ADR

New ADR: **the event-and-label flow overlay is derived, not authored.** It **extends ADR-0020** (adds
`emits`/`handled_by`/`writes`/`reads` edge types over existing event-class + glossary-label nodes) and
is **consistent with ADR-0027** (symbol-grain, lazy) and **ADR-0019** (still no authored code→intent
links; flow is structure-derived with a `# emits:` marker only for the dynamic-emit ceiling). `source:`
= this spec.

## Knowledge-graph check

Reviewed against `docs/index.md` on 2026-08-23.

| domain | touched? | note |
| --- | --- | --- |
| graph | yes — four derived edge types in the registry; walk-time (`emits`/`handled_by`/`writes`) + harvest (`reads`) wiring; two non-blocking checks | the subject |
| code | yes — `emits` from `calls` subset; `# emits:` marker in `calls_of`/reader; handler-Cypher `writes` derivation | the derivation subject |
| glossary / graph-queries | yes (read-only) — labels reused as `writes`/`reads` endpoints; graph-query `labels` metadata drives `reads` | reused, not changed |
| adr | yes — new ADR (extends 0020, consistent with 0027/0019) | — |

**Verdict:** reconciled — a mostly-derived event-and-label overlay makes the event-sourced write path,
the analysis pipeline, and the Neo4j schema-lineage traversable by connecting nodes/metadata that
already exist (event-class symbols, glossary labels, `calls`, the handler registry, graph-query
`labels`). Symbol-grain and lazy (ADR-0027); authored only via a `# emits:` fallback for the dynamic
ceiling. Stage-ordering-as-a-named-node and infra topology (KG-3) are deferred.
