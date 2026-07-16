# M4.6 — GraphRAG Ask-the-Corpus Retrieval (design)

**Status:** approved 2026-07-16
**Parent spec:** `docs/superpowers/specs/2026-07-04-mine-layers-design.md` (Layer 5 §3, "GraphRAG retrieval")
**Depends on:** M4.5 schema v2 (complete): Fragments, canonical entities (`ALIAS_OF`),
persons (`IDENTIFIED_AS`), segments (`CONTAINS`), per-model vector indexes on
fragment/utterance embeddings.

## Goal

Ask a project a question and get a grounded answer: retrieval over the schema-v2
graph fans one question across vector, fulltext, and graph-anchored channels, fuses
the results, assembles context blocks, and one focused LLM call synthesizes an
answer that cites verbatim fragments. Purely read-side — no new events, no new wire
format, projection handlers remain the sole Neo4j writers.

## Decisions locked during brainstorming

- **Full ask-the-corpus** (question → answer + citations), not retrieval-only and
  not context-packs-only. API endpoint + CLI.
- **Project-scoped**: every question runs against one `project_id` — matching how
  canonical entities and persons are keyed. Cross-project asking is deferred.
- **Purely read-side**: asks are not recorded as events. An ask-history aggregate
  can be added later without rework if it earns its keep.
- **Hand-rolled hybrid retrieval in the repo's idiom** (approach A), with
  reciprocal-rank fusion borrowed as an idea (not a dependency) from
  `neo4j-graphrag-python`. The parent spec's "borrow `neo4j-graphrag-python`
  retrievers" line (2026-07-04) is **superseded**: it predates the repo growing its
  own mature async query layer. Adding that dependency would mean a sync-driver
  adapter and an embedder-interface wrapper for what is ~3 Cypher patterns the repo
  already knows how to write and test. Rejected alternatives recorded below.
- **Sequential by design**: amended 2026-07-16 (M4.7 W3).

## Architecture & data flow

```
question + project_id
  → QUERY ANALYSIS (deterministic, no LLM): match question tokens against the
    project's canonical-entity names/surfaces and person display names
  → RETRIEVAL (three channels, sequential — one embed, then per-channel queries):
      vector    — embed the question (same model as the indexed vectors);
                  db.index.vector.queryNodes over the per-model fragment and
                  utterance indexes; hits filtered to the project's interviews
      fulltext  — db.index.fulltext.queryNodes over a new fulltext index on
                  Fragment.text (created lazily, IF NOT EXISTS); project-filtered
      graph     — anchored expansion from matched canonical entities/persons:
                  (:CanonicalEntity)<-[:ALIAS_OF]-(:Entity)<-[:MENTIONS]-(:Fragment)
                  and (:Person)<-[:IDENTIFIED_AS]-(:Speaker)<-[:SPOKEN_BY]-(:Fragment)
  → FUSION: reciprocal-rank fusion across channels — score(f) = Σ_channels
    1/(60 + rank_channel(f)) — dedupe by fragment, take top-K (default 12).
    Utterance-index vector hits expand to their member fragments (each inheriting
    the utterance's rank) BEFORE fusion, so fusion operates on fragments only
  → CONTEXT ASSEMBLY: per fragment — verbatim text, speaker (+ linked person
    display name), segment topic, entity tags, and the other fragments of its
    utterance (so a mid-thought hit reads as the whole thought); blocks ordered
    by (interview, sequence_order)
  → SYNTHESIS: ONE focused LLM call via the existing failover agent; the prompt
    carries context blocks tagged with fragment ids and instructs answering ONLY
    from the supplied context, citing fragment ids, and saying so when the context
    is insufficient
  → AskResult {answer, citations: [{fragment_id, interview_id, quote}],
    retrieval: per-channel diagnostics + flags}
```

Consumers: `POST /ask/{project_id}` (body: `{question, top_k?}`) and
`python -m src.ask <project_id> "<question>"`.

## Components

One responsibility per file, boundaries matching the repo's established idioms:

| Unit | Responsibility |
|---|---|
| `src/ask/reader.py` | The three retrieval channels as plain async functions (session + params → rows), fake-session testable like `src/export/reader.py` |
| `src/ask/fusion.py` | Pure functions: RRF merge, fragment dedupe, top-K |
| `src/ask/context.py` | Context-block assembly + synthesis prompt rendering |
| `prompts/ask_prompts.yaml` | The synthesis prompt (extractor-prompt convention) |
| `src/ask/engine.py` | `AskEngine.ask(project_id, question, top_k=12)` orchestration; `_build_embedder` / `_build_agent` monkeypatchable seams (resolution/lens-engine precedent); `AskResult` pydantic model; lazy fulltext-index ensure |
| `src/ask/__main__.py` | CLI |
| `src/api/routers/ask.py` | Endpoint (per-router private-helper convention), mounted in `src/main.py` |

Query analysis lives in the engine (it needs the project's canonical/person names,
fetched via one reader call) — deterministic substring/token matching, no LLM call.

## Error handling (degrade, don't guess)

| Failure | Behavior |
|---|---|
| Embedder unavailable (e.g. quota) | Vector channel drops out; `retrieval.flags["vector_unavailable"]`; fulltext + graph still answer |
| No entity/person matches in question | Graph channel returns empty (not an error) |
| Zero hits across all channels | NO LLM call; `answer` states no grounding was found; empty citations |
| LLM synthesis fails after failover chain | API 502 / CLI non-zero, but the response body still carries the retrieved context blocks and diagnostics |
| Unknown project_id (no `:Project` node) | 404 |
| Fulltext index missing | Created lazily `IF NOT EXISTS` on first ask (vector-index idiom) |

## Testing

- **Unit:** reader channels with fake sessions and query-text pins (incl. the
  project-scoping clauses); fusion table-driven (RRF math, dedupe, tie ordering,
  top-K); context assembly (block content/order); engine with canned channel rows,
  fake embedder, mocked agent (citation extraction, each degradation path, the
  no-hits early exit, prompt content); router tests (200/404/502 shapes).
- **Integration:** a seventh smoke extending the corpus story — ingest two small
  interviews, run canned enrichment + resolution + segments (existing smoke
  idioms), replay through the real registry, then `AskEngine.ask` with a fake
  embedder and canned synthesis agent; assert the citations point at the expected
  fragments and that the graph channel surfaced the entity-anchored ones. All
  existing smokes stay green.
- No live-LLM dependencies anywhere in the suite.

## Non-goals (M4.6) — onto the backlog

- **Text2Cypher as an alternate route for novel inquiries** — an LLM-written-Cypher
  channel for questions the fixed channels can't shape (aggregations, exotic
  traversals). Nice-to-have, not a necessity; deferred to the ROADMAP backlog.
- Ask-history events / Q&A review surface (read-side only in v1).
- Cross-project asking (canonical identity is project-keyed by design).
- Conversation memory / multi-turn asking.
- Reranking beyond RRF (cross-encoder rerankers, MMR diversity).
- Streaming responses.

## Rejected alternatives

- **`neo4j-graphrag-python` as a dependency** — sync-driver adapter + embedder
  wrapper + un-idiomatic testing for ~3 Cypher patterns; borrow the RRF idea only.
- **Text2Cypher as the primary mechanism** — ungrounded generated Cypher against a
  frozen schema, non-deterministic tests, against the focused-calls/traceability
  doctrine. Survives only as the deferred alternate-route backlog item above.
- **Retrieval-only first slice** — would defer the part that proves the graph
  answers questions; synthesis is one focused call, small enough to include.
