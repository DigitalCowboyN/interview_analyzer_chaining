# Layer 3 (M4.3): Debt Burndown + Generic Lens Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Burn down the M4.2-exit debt, then ship the fully generic lens engine (Approach A) with the meeting_minutes lens — one YAML + prompts per future lens, zero per-lens code.

**Architecture:** Debt fixes land first on the same branch (embedding per-model property, resilient failover construction, dead-code deletion, provenance/edge minors). Then: lens extractors are ordinary `ExtractorSpec`s run by the M4.2 executor (document scope added); a `LensEngine` resolves owners against Layer 1 speakers and emits three generic events on the Interview stream (`LensApplied` supersession marker, `LensExtractionGenerated` items, `LensExtractionOverridden` corrections); one generic projection handler MERGEs dynamically-labeled nodes (label validated against the lens YAML's `projects_to`, sanitized as defense) with `SUPPORTED_BY` fragment grounding and declarative speaker links.

**Tech Stack:** Python 3.10, Pydantic v2, EventStoreDB (existing repositories), Neo4j 5.26, FastAPI. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-07-04-mine-layers-design.md` — Layer 3 section + "M4.3 design decisions".

## Global Constraints

- **Zero per-lens code**: adding a lens must require only a YAML file under `lenses/` and a prompts YAML. No new event types, handlers, allowlist entries, or Python.
- **Focused calls, not one-shots**: each lens extractor is its own `ExtractorSpec` with its own prompt, response model, scope, and confidence.
- **Projection-delivery checklist** for the three new event types (once, permanently): registered in bootstrap; on the interview subscription allowlist; handlers raise (never no-op) when cross-stream MATCH targets aren't projected; drift-guard test stays green. Bootstrap pins go 19 → 22.
- **Dynamic node labels are NEVER raw LLM output**: `node_type` is validated at emit time against the lens spec's `projects_to` keys AND sanitized (`^[A-Za-z][A-Za-z0-9_]*$`) at the handler as defense-in-depth.
- **Deterministic item ids**: `uuid5(NAMESPACE_DNS, f"{interview_id}:lens:{lens}:{node_type}:{source_unit}:{ordinal}")` where `source_unit` is the utterance_id or the literal `"document"`.
- **Supersession**: `LensApplied {lens, lens_version}` precedes a run's items; its handler deletes that interview+lens's prior UNLOCKED nodes. Same-version re-run without `--force` = idempotent skip of existing item_ids. Human-overridden items are locked: survive re-runs, skipped by the engine, never deleted by LensApplied.
- All confidences numeric [0,1] validated at command time (payload models). Human corrections carry actor HUMAN and lock fields.
- **Every Sentence-stream payload carries `interview_id`** (LaneManager drops it otherwise). The three new events are Interview-stream (aggregate_id IS the routing key) — no payload key needed.
- Unit tests need no API keys/infra (mock at module seams); integration tests get `@pytest.mark.integration`.
- Environment: `set -a; source .env; set +a; ~/.pyenv/versions/3.10.7/bin/python -m pytest ...` (plain `python` not on PATH). Full unit suite: `... -m "not integration" -q`. From a worktree, pin `COMPOSE_PROJECT_NAME=interview_analyzer_chaining` for anything touching docker.
- Style: `black`, `flake8`; follow existing patterns (payload models + aggregate command methods; `BaseProjectionHandler`; prompts YAML with `{{`/`}}`-escaped JSON braces).

## File Structure (new/major)

```
src/enrichment/executor.py     # + document scope, public run_spec_on_text (SpecOutcome)
src/lens/
  __init__.py
  models.py                    # LensSpec, LensNodeMapping, load_lens()
  engine.py                    # LensEngine (owner resolution, dedup, events)
  __main__.py                  # python -m src.lens <interview_id> <lens_name> [--force]
src/models/lens_responses.py   # meeting lens response models (StrictResult)
lenses/meeting_minutes.yaml    # the first lens profile
prompts/lens_meeting_minutes.yaml
src/events/interview_events.py # + LensApplied/LensExtractionGenerated/LensExtractionOverridden payloads
src/events/aggregates.py       # + Interview lens state & command methods
src/projections/handlers/lens_handlers.py  # the three generic handlers
src/api/routers/lenses.py      # override endpoint (corrections surface)
```

---

### Task 1: Debt — per-model embedding property + dim validation

**Files:**
- Modify: `src/projections/handlers/embedding_handlers.py`
- Modify: `src/enrichment/embedder.py` (dim validation; OpenAI `dimensions` param)
- Test: `tests/projections/test_embedding_handlers.py`, `tests/enrichment/test_embedder.py`

**Interfaces:**
- Produces: embeddings stored on per-model properties `embedding_<sanitized_model>` (sanitize: `re.sub(r"[^A-Za-z0-9]", "_", model)`), each per-model vector index targeting ITS property (`FOR (n:Sentence) ON n.embedding_<sanitized>`). The generic `embedding`/`embedding_model`/`embedding_dim` convenience properties remain (latest write wins) for simple queries; cross-model isolation lives in the per-model properties/indexes.
- `Embedder.embed` implementations raise `ValueError` if any returned vector length != `self.dim`; `OpenAIEmbedder` passes `dimensions=self.dim` to the API.

- [ ] **Step 1: Write the failing tests.** In `tests/projections/test_embedding_handlers.py`, update `test_embedding_written_with_model_tag` and add:

```python
@pytest.mark.asyncio
async def test_embedding_written_to_per_model_property():
    handler = EmbeddingGeneratedHandler()
    handler._ensured_models = {"text-embedding-3-small"}
    tx = AsyncMock()
    counters = MagicMock(nodes_created=0, properties_set=4, relationships_created=0)
    tx.run.return_value.consume = AsyncMock(return_value=MagicMock(counters=counters))
    event = EventEnvelope(
        event_type="EmbeddingGenerated", aggregate_type=AggregateType.SENTENCE,
        aggregate_id=F1, version=3,
        data={"interview_id": IID, "model": "text-embedding-3-small", "dim": 3,
              "vector_b64": encode_vector([0.1, 0.2, 0.3])},
    )
    await handler.apply(tx, event)
    query = tx.run.call_args[0][0]
    assert "embedding_text_embedding_3_small = $vector" in query  # per-model property
    assert "s.embedding = $vector" in query  # convenience property kept
```

In `tests/enrichment/test_embedder.py` add:

```python
@pytest.mark.asyncio
async def test_embedder_rejects_wrong_dim():
    from src.enrichment.embedder import OpenAIEmbedder

    embedder = OpenAIEmbedder.__new__(OpenAIEmbedder)
    embedder.model_name = "text-embedding-3-small"
    embedder.dim = 4  # API returns 3 -> mismatch
    embedder.client = MagicMock()
    item = MagicMock(embedding=[0.1, 0.2, 0.3])
    embedder.client.embeddings.create = AsyncMock(return_value=MagicMock(data=[item]))
    with pytest.raises(ValueError, match="dim"):
        await embedder.embed(["hello"])


@pytest.mark.asyncio
async def test_openai_embedder_passes_dimensions_param():
    from src.enrichment.embedder import OpenAIEmbedder

    embedder = OpenAIEmbedder.__new__(OpenAIEmbedder)
    embedder.model_name = "text-embedding-3-small"
    embedder.dim = 3
    embedder.client = MagicMock()
    item = MagicMock(embedding=[0.1, 0.2, 0.3])
    embedder.client.embeddings.create = AsyncMock(return_value=MagicMock(data=[item]))
    await embedder.embed(["hello"])
    assert embedder.client.embeddings.create.call_args.kwargs["dimensions"] == 3
```

- [ ] **Step 2: Run to verify fail** — `pytest tests/projections/test_embedding_handlers.py tests/enrichment/test_embedder.py -q --no-cov` → new tests FAIL.

- [ ] **Step 3: Implement.** In `embedding_handlers.py`, change both apply queries to also SET the per-model property and point each index at it:

```python
        prop = f"embedding_{_sanitize(data['model'])}"
        query = f"""
        MATCH (s:Sentence {{aggregate_id: $aggregate_id}})
        SET s.{prop} = $vector, s.embedding = $vector,
            s.embedding_model = $model, s.embedding_dim = $dim
        """
```

(`prop` derives from `_sanitize`, never raw input; same pattern for the Utterance handler.) In `_ensure_vector_index`, index `ON n.{prefix_prop}` where the property name is passed in (`f"embedding_{_sanitize(model)}"`) instead of the shared `embedding`.

In `embedder.py`: `OpenAIEmbedder.embed` passes `dimensions=self.dim` to `embeddings.create`; add to BOTH embedders after collecting vectors:

```python
        for v in vectors:
            if len(v) != self.dim:
                raise ValueError(f"Embedder returned dim {len(v)}, configured dim {self.dim}")
```

(For `LocalEmbedder`, apply to the converted float lists.)

- [ ] **Step 4: Run to verify pass** — same command + `pytest tests/projections tests/enrichment -m "not integration" -q` → green, no regressions.
- [ ] **Step 5: Commit** — `git add -A src tests && git commit -m "fix(debt): per-model embedding properties/indexes; embedder dim validation"`

---

### Task 2: Debt — resilient failover construction + provenance/edge minors

**Files:**
- Modify: `src/agents/failover_agent.py` (`get_failover_agent` skips unconstructible providers)
- Modify: `src/projections/handlers/entity_handlers.py` (materialize provider; MENTIONS keyed by span)
- Modify: `src/enrichment/executor.py` (mixed-provider flag)
- Modify: `src/enrichment/orchestrator.py` (embed only non-failed fragments)
- Modify: `src/main.py` (batch CLI per-file isolation)
- Modify: `tests/enrichment/test_final_review_fixes.py` (harden `_assert_strict` recursion)
- Test: `tests/agents/test_failover_agent.py`, `tests/projections/test_entity_claim_handlers.py`, `tests/enrichment/test_executor.py`

**Interfaces:**
- `get_failover_agent`: providers that raise on construction are skipped with a logged warning; empty surviving chain raises `ValueError("No usable LLM provider...")`.
- `MENTIONS` edges keyed by span: `MERGE (s)-[m:MENTIONS {start: ent.start, end: ent.end}]->(e)` (two mentions of one entity in one fragment = two edges); Sentence guard statement also sets `s.entities_provider = $provider`.
- Executor: when a unit's successful calls span multiple providers, set `flags["mixed_providers"] = "<p1>,<p2>"` (provenance coarseness made visible).
- Orchestrator: `embedder.embed` is called AFTER enrichments return, only for fragments with `enrichment.model` set (no wasted compute on fully-failed fragments).
- Batch CLI: one failing file logs and continues; exit 1 at the end if any failed.
- `_assert_strict` recurses into `properties` values and `items` schemas, not just `$defs`.

- [ ] **Step 1: Write the failing tests.**

`tests/agents/test_failover_agent.py` add:

```python
def test_factory_skips_unconstructible_providers():
    from unittest.mock import patch

    from src.agents.failover_agent import get_failover_agent

    good = MagicMock()

    def create(name):
        if name == "anthropic":
            raise ValueError("API key is not set.")
        return good

    with patch("src.agents.agent_factory.AgentFactory.create_agent", side_effect=create):
        agent = get_failover_agent({"llm": {"chain": ["anthropic", "openai"]}})
    assert agent.providers == [good]


def test_factory_raises_when_no_provider_constructible():
    from unittest.mock import patch

    import pytest as _pytest

    from src.agents.failover_agent import get_failover_agent

    with patch(
        "src.agents.agent_factory.AgentFactory.create_agent",
        side_effect=ValueError("no key"),
    ):
        with _pytest.raises(ValueError, match="No usable LLM provider"):
            get_failover_agent({"llm": {"chain": ["anthropic", "openai"]}})
```

`tests/projections/test_entity_claim_handlers.py` add:

```python
@pytest.mark.asyncio
async def test_mentions_keyed_by_span_and_provider_materialized():
    handler = EntitiesExtractedHandler()
    tx = AsyncMock()
    mock_write_counters(tx)
    two_spans = [
        {"text": "ECU", "entity_type": "product", "start": 0, "end": 3, "confidence": 0.9},
        {"text": "ECU", "entity_type": "product", "start": 20, "end": 23, "confidence": 0.8},
    ]
    event = make_event(
        "EntitiesExtracted", AggregateType.SENTENCE, F1,
        {"interview_id": IID, "model": "haiku", "provider": "anthropic", "entities": two_spans},
    )
    await handler.apply(tx, event)
    queries = [c.args[0] for c in tx.run.call_args_list]
    assert any("entities_provider" in q for q in queries)
    assert any("MENTIONS {start: ent.start, end: ent.end}" in q for q in queries)
```

`tests/enrichment/test_executor.py` add:

```python
@pytest.mark.asyncio
async def test_mixed_providers_flagged():
    calls = {"n": 0}

    agent = MagicMock()

    async def call(prompt, schema=None):
        for key, marker in MARKERS.items():
            if marker in prompt:
                calls["n"] += 1
                provider = "anthropic" if calls["n"] % 2 else "openai"
                return CallResult(data=RESPONSES[key], provider=provider, model="m")
        raise AssertionError

    agent.call = AsyncMock(side_effect=call)
    executor = make_executor(agent)
    fragments = [FragmentView(index=0, text="Can you hear me?", speaker_handle="S1")]
    results = await executor.enrich_fragments(fragments, [CONTEXT])
    assert "mixed_providers" in results[0].flags
```

- [ ] **Step 2: Run to verify fail.**
- [ ] **Step 3: Implement.**

`get_failover_agent` loop becomes:

```python
    providers = []
    for name in chain:
        try:
            providers.append(AgentFactory.create_agent(name))
        except Exception as exc:
            logger.warning(f"Provider {name!r} unavailable at construction ({exc}); skipping")
    if not providers:
        raise ValueError(f"No usable LLM provider in chain {chain!r}")
    return FailoverAgent(providers)
```

`entity_handlers.py`: guard statement adds `s.entities_provider = $provider` (param `provider=data.get("provider")`); edges query MERGE becomes `MERGE (s)-[m:MENTIONS {start: ent.start, end: ent.end}]->(e)` with `SET m.text = ent.text, m.confidence = ent.confidence`.

`executor.py` `_enrich_fragment`: collect `providers_seen = {r.provider for r in results if not isinstance(r, BaseException)}`; after the loop, `if len(providers_seen) > 1: out.flags["mixed_providers"] = ",".join(sorted(providers_seen))`. Add a docstring note that `out.provider` is the last successful call's provider.

`orchestrator.py` `_emit_fragment_results`: run `enrich_fragments` first, then `ok = [(e, by_index[e.index]) for e in enrichments if e.model]`, embed only `[s.text for _, s in ok]`, and zip vectors against `ok` (fully-failed fragments logged + skipped BEFORE spending embedding calls).

`main.py` `_batch_ingest_enrich`: wrap the per-file body in try/except, collect failures, `raise RuntimeError(f"{len(failures)} file(s) failed: {failures}")` at the end so `main()`'s existing failed-path exits 1.

`test_final_review_fixes.py` `_assert_strict`: after the object checks, also recurse: `for sub in schema.get("properties", {}).values(): _assert_strict(sub, ...)` and `if "items" in schema: _assert_strict(schema["items"], ...)`.

- [ ] **Step 4: Run** — `pytest tests/agents tests/projections tests/enrichment -m "not integration" -q` → green.
- [ ] **Step 5: Commit** — `git commit -am "fix(debt): resilient chain construction; span-keyed MENTIONS; provider provenance; batch isolation"`

---

### Task 3: Debt — dead-code deletion

**Files:**
- Delete: `src/io/local_storage.py`, `src/io/protocols.py`, `src/io/neo4j_analysis_writer.py`, `src/io/neo4j_map_storage.py` (entire `src/io/` — verified unreferenced by surviving `src/` code)
- Delete: `tests/io/` (whole dir), `tests/integration/test_neo4j_fault_tolerance.py`, `tests/integration/test_neo4j_performance_benchmarks.py` (long-skipped legacy suites that import the deleted modules; their ESDB rewrites are separate future work — note in ROADMAP)
- Modify: `config.yaml` (delete the `classification:` block — sole reader was the deleted SentenceAnalyzer)
- Modify: `docs/ROADMAP.md` (skipped-tests inventory + debt list updated)

- [ ] **Step 1: Inventory before deleting.** Run: `command grep -rln "src\.io\|from src.io" src tests --include="*.py"` — every hit must be in the deletion list above. Anything else → STOP, report DONE_WITH_CONCERNS.
- [ ] **Step 2: Delete** (`git rm -r src/io tests/io`; `git rm` the two integration files), edit config.yaml and ROADMAP (move the fault-tolerance/perf-benchmark rewrite note to Future Improvements).
- [ ] **Step 3: Full unit suite** — `pytest tests -m "not integration" -q` → green, zero collection errors.
- [ ] **Step 4: Commit** — `git commit -am "refactor(debt): delete dead src/io and legacy skipped suites; drop dead config block"`

---

### Task 4: Executor — document scope + public SpecOutcome API

**Files:**
- Modify: `src/enrichment/executor.py`
- Test: `tests/enrichment/test_executor.py`

**Interfaces:**
- Produces: `SpecOutcome(BaseModel)`: `data: Optional[Dict[str, Any]]` (validated `model_dump()` or None), `flags: Dict[str, str]`, `provider: str = ""`, `model: str = ""`.
- `EnrichmentExecutor.run_spec_on_text(spec: ExtractorSpec, text: str, context: Optional[Dict[str, str]] = None) -> SpecOutcome` — public, semaphore-bounded, schema-enforced, validation/call-error handling identical to the fragment path (`<name>_invalid_response` / `<name>_call_error` flags). The internal fragment/utterance paths refactor onto it (behavior unchanged — existing tests must stay green unmodified).
- `scope: "document"` specs are now legal end-to-end: `self.document_specs` split out in `__init__`; the LensEngine (Task 8) calls `run_spec_on_text` per document spec with the full transcript text. (Core `config/extractors.yaml` gains no document extractors — this is lens infrastructure.)

- [ ] **Step 1: Write the failing tests** (`tests/enrichment/test_executor.py`):

```python
@pytest.mark.asyncio
async def test_run_spec_on_text_returns_validated_outcome():
    from src.enrichment.models import ExtractorSpec

    executor = make_executor(make_agent())
    spec = ExtractorSpec(name="purpose", prompt_key="purpose",
                         response_model="PurposeResult",
                         context_needs=["observer_context"], scope="fragment")
    outcome = await executor.run_spec_on_text(spec, "Can you hear me?", {"observer_context": "c"})
    assert outcome.data == {"purpose": "Query", "confidence": 0.85}
    assert outcome.provider == "anthropic"
    assert outcome.flags == {}


@pytest.mark.asyncio
async def test_run_spec_on_text_flags_call_error():
    from src.enrichment.models import ExtractorSpec

    agent = MagicMock()
    agent.call = AsyncMock(side_effect=RuntimeError("down"))
    executor = make_executor(agent)
    spec = ExtractorSpec(name="purpose", prompt_key="purpose",
                         response_model="PurposeResult", scope="fragment")
    outcome = await executor.run_spec_on_text(spec, "text")
    assert outcome.data is None
    assert outcome.flags == {"purpose_call_error": "RuntimeError"}


def test_document_scope_specs_split():
    from src.enrichment.models import ExtractorSpec
    from src.enrichment.registry import ExtractorRegistry
    from src.utils.helpers import load_yaml

    specs = ExtractorRegistry.load("config/extractors.yaml")
    specs.append(ExtractorSpec(name="objectives", prompt_key="purpose",
                               response_model="PurposeResult", scope="document"))
    executor = EnrichmentExecutor(MagicMock(), specs,
                                  load_yaml("prompts/core_extractors.yaml"),
                                  domain_keywords=[], concurrency=2)
    assert [s.name for s in executor.document_specs] == ["objectives"]
```

- [ ] **Step 2: Run to verify fail** — AttributeError (`run_spec_on_text`, `document_specs`).
- [ ] **Step 3: Implement.** In `__init__` add `self.document_specs = [s for s in specs if s.scope == "document"]`. Add:

```python
class SpecOutcome(BaseModel):
    data: Optional[Dict[str, Any]] = None
    flags: Dict[str, str] = Field(default_factory=dict)
    provider: str = ""
    model: str = ""


    async def run_spec_on_text(
        self, spec: ExtractorSpec, text: str, context: Optional[Dict[str, str]] = None
    ) -> SpecOutcome:
        """One focused, schema-enforced call for one spec over one text unit."""
        outcome = SpecOutcome()
        try:
            call_result = await self._run_spec(spec, text, context or {})
        except Exception as exc:
            logger.warning(f"{spec.name}: call failed ({type(exc).__name__})")
            outcome.flags[f"{spec.name}_call_error"] = type(exc).__name__
            return outcome
        outcome.provider, outcome.model = call_result.provider, call_result.model
        try:
            parsed = spec.resolve_model().model_validate(call_result.data)
        except ValidationError as e:
            logger.warning(f"{spec.name}: invalid response ({e.error_count()} errors)")
            outcome.flags[f"{spec.name}_invalid_response"] = "validation failed"
            return outcome
        outcome.data = parsed.model_dump()
        return outcome
```

Refactor `_enrich_fragment` and `enrich_utterances.one` to call `run_spec_on_text` (fragment path: gather `run_spec_on_text` per spec — call errors already come back as flags, so drop the `return_exceptions` handling in favor of merging `outcome.flags`; classification routing reads `outcome.data`). All existing executor tests must pass unchanged.

- [ ] **Step 4: Run** — `pytest tests/enrichment -q --no-cov` → all green including pre-existing tests.
- [ ] **Step 5: Commit** — `git commit -am "feat: executor document scope + public run_spec_on_text (SpecOutcome)"`

---

### Task 5: Lens profile model + meeting_minutes lens + response models + prompts

**Files:**
- Create: `src/lens/__init__.py` (empty), `src/lens/models.py`, `src/models/lens_responses.py`, `lenses/meeting_minutes.yaml`, `prompts/lens_meeting_minutes.yaml`
- Test: `tests/lens/__init__.py` (empty), `tests/lens/test_lens_models.py`

**Interfaces:**
- `LensNodeMapping(BaseModel)`: `speaker_link: Optional[Dict[str, str]] = None` (keys: `field` — the extracted field holding a speaker reference; `relationship` — e.g. `OWNED_BY`).
- `LensSpec(BaseModel)`: `name: str`, `version: int (ge 1)`, `prompts_file: str`, `extractors: List[LensExtractorDecl]`, `projects_to: Dict[str, LensNodeMapping]` (node_type label → mapping; labels must match `^[A-Z][A-Za-z0-9]*$`, model_validator).
- `LensExtractorDecl(BaseModel)`: `name`, `prompt_key`, `response_model` (class in `src.models.lens_responses`), `scope: Literal["fragment","utterance","document"]`, `node_type: str` (must be a `projects_to` key — validated on LensSpec), `items_field: str` (the list field in the response model, e.g. `"decisions"`). Method `to_extractor_spec() -> ExtractorSpec` (maps name/prompt_key/response_model/scope; `resolve_model()` overridden to import from `src.models.lens_responses`). Simplest: give `ExtractorSpec` an optional `response_module: str = "src.models.extractor_responses"` field and have `resolve_model()` use it — one-line change in `src/enrichment/models.py`, declared here, implemented in this task.
- `load_lens(name: str, lenses_dir: str = "lenses") -> LensSpec` — loads `lenses/<name>.yaml`; `ValueError` on unknown name, bad label, or extractor referencing an undeclared node_type.
- Response models (all `StrictResult`-based, in `src/models/lens_responses.py`):
  - `ObjectiveItem(text: str, confidence)` / `ObjectivesResult(objectives: List[ObjectiveItem])`
  - `DecisionItem(text: str, made_by: Optional[str] = None, confidence)` / `DecisionsResult(decisions: List[DecisionItem])`
  - `ActionItem(text: str, owner: Optional[str] = None, due: Optional[str] = None, confidence)` / `ActionItemsResult(action_items: List[ActionItem])`
  - `FollowupItem(text: str, confidence)` / `FollowupsResult(followups: List[FollowupItem])`
  - NOTE (OpenAI strict mode): Optional fields still satisfy props⊆required because Pydantic marks them required-with-null in strict dumps? NO — Optional-with-default fields are NOT in `required`. To stay strict-compliant, declare `made_by`, `owner`, `due` as `Optional[str]` WITHOUT default (`made_by: Optional[str]` = required, nullable). The strict-compliance test from M4.2 (now hardened) enforces this — run it against these models by importing them in a lens variant of the test (Step 1 includes it).
- `lenses/meeting_minutes.yaml`:

```yaml
name: meeting_minutes
version: 1
prompts_file: prompts/lens_meeting_minutes.yaml
projects_to:
  Objective: {}
  Decision:
    speaker_link: {field: made_by, relationship: DECIDED_BY}
  ActionItem:
    speaker_link: {field: owner, relationship: OWNED_BY}
  FollowUp: {}
extractors:
  - name: objectives
    prompt_key: objectives
    response_model: ObjectivesResult
    scope: document
    node_type: Objective
    items_field: objectives
  - name: decisions
    prompt_key: decisions
    response_model: DecisionsResult
    scope: utterance
    node_type: Decision
    items_field: decisions
  - name: action_items
    prompt_key: action_items
    response_model: ActionItemsResult
    scope: utterance
    node_type: ActionItem
    items_field: action_items
  - name: followups
    prompt_key: followups
    response_model: FollowupsResult
    scope: utterance
    node_type: FollowUp
    items_field: followups
```

- `prompts/lens_meeting_minutes.yaml` — four prompts. All demand JSON with numeric confidence; utterance prompts receive the utterance as `{sentence}` (executor contract); the document prompt receives the full speaker-labeled transcript as `{sentence}`. Two shown in full; the implementer writes `objectives` and `followups` in the same voice:

```yaml
decisions:
  prompt: |
    The following is one speaker's complete utterance from a meeting.
    Extract any DECISIONS it contains — a decision is a resolved choice or
    commitment on behalf of the group ("we'll go with X", "let's not do Y").
    Statements of opinion or open questions are NOT decisions.

    If the utterance attributes the decision to a person by name or handle,
    set made_by to that exact string; otherwise set made_by to null.

    Format: {{"decisions": [{{"text": "<the decision>", "made_by": "<name or null>", "confidence": <number between 0 and 1>}}]}}
    If there are no decisions, return {{"decisions": []}}.

    Utterance:
    """
    {sentence}
    """

    Provide your response explicitly formatted as JSON.

action_items:
  prompt: |
    The following is one speaker's complete utterance from a meeting.
    Extract any ACTION ITEMS — concrete tasks someone is expected to do after
    the meeting. Include the owner when stated or clearly implied (the speaker
    volunteering counts: use their words like "I'll..." -> owner is the speaker,
    set owner to "SELF"). Include a due reference if one is stated (verbatim,
    e.g. "Friday", "next sprint"); otherwise null.

    Format: {{"action_items": [{{"text": "<the task>", "owner": "<name, SELF, or null>", "due": "<text or null>", "confidence": <number between 0 and 1>}}]}}
    If there are none, return {{"action_items": []}}.

    Utterance:
    """
    {sentence}
    """

    Provide your response explicitly formatted as JSON.
```

(`objectives`: document-scoped — "the full transcript of a meeting follows, speaker-labeled; extract the meeting's OBJECTIVES — what the participants convened to accomplish — as short statements"; `followups`: utterance-scoped — "topics explicitly deferred to a later conversation".)

- [ ] **Step 1: Write the failing tests** (`tests/lens/test_lens_models.py`):

```python
import pytest

from src.lens.models import LensSpec, load_lens


def test_load_meeting_minutes_lens():
    lens = load_lens("meeting_minutes")
    assert lens.name == "meeting_minutes"
    assert lens.version == 1
    assert set(lens.projects_to) == {"Objective", "Decision", "ActionItem", "FollowUp"}
    assert lens.projects_to["ActionItem"].speaker_link == {
        "field": "owner", "relationship": "OWNED_BY"
    }
    scopes = {e.name: e.scope for e in lens.extractors}
    assert scopes["objectives"] == "document"
    assert scopes["decisions"] == "utterance"


def test_extractor_decls_convert_to_specs_with_lens_module():
    lens = load_lens("meeting_minutes")
    spec = next(e for e in lens.extractors if e.name == "decisions").to_extractor_spec()
    from src.models.lens_responses import DecisionsResult

    assert spec.resolve_model() is DecisionsResult


def test_unknown_lens_rejected():
    with pytest.raises(ValueError, match="Unknown lens"):
        load_lens("no_such_lens")


def test_extractor_node_type_must_be_declared(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "name: bad\nversion: 1\nprompts_file: p.yaml\n"
        "projects_to:\n  Decision: {}\n"
        "extractors:\n  - {name: x, prompt_key: k, response_model: DecisionsResult, "
        "scope: utterance, node_type: NotDeclared, items_field: decisions}\n"
    )
    with pytest.raises(ValueError, match="node_type"):
        load_lens("bad", lenses_dir=str(tmp_path))


def test_invalid_label_rejected(tmp_path):
    bad = tmp_path / "bad2.yaml"
    bad.write_text(
        "name: bad2\nversion: 1\nprompts_file: p.yaml\n"
        "projects_to:\n  'DROP DATABASE': {}\nextractors: []\n"
    )
    with pytest.raises(ValueError, match="label"):
        load_lens("bad2", lenses_dir=str(tmp_path))


def test_lens_prompts_format_cleanly():
    from src.utils.helpers import load_yaml

    lens = load_lens("meeting_minutes")
    prompts = load_yaml(lens.prompts_file)
    for decl in lens.extractors:
        formatted = prompts[decl.prompt_key]["prompt"].format(sentence="Test text.")
        assert "{sentence}" not in formatted


def test_lens_response_models_are_openai_strict_compliant():
    from tests.enrichment.test_final_review_fixes import _assert_strict

    lens = load_lens("meeting_minutes")
    for decl in lens.extractors:
        _assert_strict(decl.to_extractor_spec().resolve_model().model_json_schema(), decl.name)
```

- [ ] **Step 2: Run to verify fail** — ModuleNotFoundError.
- [ ] **Step 3: Implement** per the Interfaces block: `response_module` field on `ExtractorSpec` (default `"src.models.extractor_responses"`, used in `resolve_model()`); `src/models/lens_responses.py` with the eight models (Optional fields WITHOUT defaults, per the strict-mode note — verify `test_lens_response_models_are_openai_strict_compliant` passes); `src/lens/models.py` with `LensExtractorDecl` (incl. `to_extractor_spec()` setting `response_module="src.models.lens_responses"`), `LensNodeMapping`, `LensSpec` (label-regex + node_type-declared validators), `load_lens` (FileNotFoundError → `ValueError(f"Unknown lens: {name}")`); the two YAML files (all four prompts written out).
- [ ] **Step 4: Run** — `pytest tests/lens tests/enrichment -m "not integration" -q --no-cov` → green.
- [ ] **Step 5: Commit** — `git commit -am "feat: lens profile model, meeting_minutes lens, response models, prompts"`

---

### Task 6: Lens events on the Interview aggregate

**Files:**
- Modify: `src/events/interview_events.py`, `src/events/aggregates.py`
- Test: `tests/events/test_lens_events.py`

**Interfaces:**
- Payloads (interview_events.py):
  - `LensAppliedData(lens: str, lens_version: int (ge 1))`
  - `LensExtractionGeneratedData(lens: str, lens_version: int, node_type: str, item_id: str, fields: Dict[str, Any], supporting_fragment_ids: List[str] = [], speaker_links: List[Dict[str, str]] = [] (each {relationship, speaker_id}), confidence: float [0,1], model: str, provider: str)`
  - `LensExtractionOverriddenData(item_id: str, fields_overridden: Dict[str, Any], note: Optional[str] = None)`
- Interview state: `self.lens_runs: Dict[str, int]` (lens → latest version applied); `self.lens_items: Dict[str, Dict[str, Any]]` (item_id → {lens, lens_version, node_type, locked: bool}).
- Command methods:
  - `apply_lens(lens, lens_version, **kw)` — guards: created; version >= previous run's version for that lens (re-running older versions is an error). Emits `LensApplied`.
  - `record_lens_extraction(lens, lens_version, node_type, item_id, fields, supporting_fragment_ids, speaker_links, confidence, model, provider, **kw)` — guards: created; a `LensApplied` for this lens+version must precede it (state check `self.lens_runs.get(lens) == lens_version`); duplicate item_id raises (`"already recorded"` — the ENGINE does idempotent skipping, mirroring claims); LOCKED item_id raises (`"locked"`).
  - `override_lens_extraction(item_id, fields_overridden, note=None, **kw)` — guards: item exists; sets `locked=True` on apply (human correction survives re-runs).
- Apply methods reconstruct all state on replay (lens_runs, lens_items incl. locked).

- [ ] **Step 1: Write the failing tests** (`tests/events/test_lens_events.py`):

```python
import pytest

from src.events.aggregates import Interview

IID = "22222222-2222-2222-2222-222222222222"
SP1 = "33333333-3333-3333-3333-333333333333"
ITEM = "88888888-8888-8888-8888-888888888801"


def make_interview():
    i = Interview(IID)
    i.create(title="t", source="s")
    i.add_speaker(SP1, "S1", "S1", True, 0.9, "inference")
    i.apply_lens("meeting_minutes", 1)
    return i


def record(i, item_id=ITEM, **over):
    kwargs = dict(
        lens="meeting_minutes", lens_version=1, node_type="Decision", item_id=item_id,
        fields={"text": "Go with X", "made_by": "S1"},
        supporting_fragment_ids=["77777777-7777-7777-7777-777777777771"],
        speaker_links=[{"relationship": "DECIDED_BY", "speaker_id": SP1}],
        confidence=0.9, model="haiku", provider="anthropic",
    )
    kwargs.update(over)
    return i.record_lens_extraction(**kwargs)


def test_apply_lens_records_run():
    i = make_interview()
    assert i.lens_runs == {"meeting_minutes": 1}


def test_apply_lens_rejects_version_downgrade():
    i = make_interview()
    i.apply_lens("meeting_minutes", 3)
    with pytest.raises(ValueError, match="version"):
        i.apply_lens("meeting_minutes", 2)


def test_record_requires_matching_lens_run():
    i = Interview(IID)
    i.create(title="t", source="s")
    with pytest.raises(ValueError, match="LensApplied"):
        record(i)


def test_record_and_replay():
    i = make_interview()
    event = record(i)
    assert event.event_type == "LensExtractionGenerated"
    assert i.lens_items[ITEM]["node_type"] == "Decision"
    replayed = Interview(IID)
    replayed.load_from_history(i.get_uncommitted_events())
    assert replayed.lens_items[ITEM]["lens_version"] == 1
    assert replayed.lens_runs == {"meeting_minutes": 1}


def test_duplicate_item_raises():
    i = make_interview()
    record(i)
    with pytest.raises(ValueError, match="already recorded"):
        record(i)


def test_override_locks_item_against_rerecord():
    i = make_interview()
    record(i)
    event = i.override_lens_extraction(ITEM, {"text": "Go with Y"}, note="fixed wording")
    assert event.event_type == "LensExtractionOverridden"
    assert i.lens_items[ITEM]["locked"] is True
    i.apply_lens("meeting_minutes", 2)
    with pytest.raises(ValueError, match="locked"):
        record(i, lens_version=2)


def test_override_unknown_item_raises():
    i = make_interview()
    with pytest.raises(ValueError, match="Unknown lens item"):
        i.override_lens_extraction("no-such", {"text": "x"})
```

- [ ] **Step 2: Run to verify fail** — AttributeError.
- [ ] **Step 3: Implement** following the exact established pattern (payload models with Field constraints; command methods validating via `.model_dump()`; pure apply methods; dispatch branches in `Interview.apply_event`). State updates: `_apply_lens_applied` sets `lens_runs[lens] = lens_version`; `_apply_lens_extraction_generated` sets `lens_items[item_id] = {lens, lens_version, node_type, locked: False}`; `_apply_lens_extraction_overridden` sets `locked = True`.
- [ ] **Step 4: Run** — `pytest tests/events -m "not integration" -q` → green, no regressions.
- [ ] **Step 5: Commit** — `git commit -am "feat: lens events (LensApplied/ExtractionGenerated/Overridden) on Interview aggregate"`

---

### Task 7: Generic lens projection handlers

**Files:**
- Create: `src/projections/handlers/lens_handlers.py`
- Modify: `src/projections/bootstrap.py`, `src/projections/config.py` (interview allowlist), `tests/projections/test_bootstrap_unit.py` (pins 19 → 22 + expected set)
- Test: `tests/projections/test_lens_handlers.py`

**Interfaces:**
- `_validate_label(node_type: str) -> str` — must match `^[A-Z][A-Za-z0-9]*$` else raise ValueError (defense-in-depth; emit-time validation is primary).
- `LensAppliedHandler`: deletes prior UNLOCKED lens nodes for this interview+lens with `lens_version < $lens_version`:

```cypher
MATCH (n:LensItem {interview_id: $interview_id, lens: $lens})
WHERE n.lens_version < $lens_version AND coalesce(n.locked, false) = false
DETACH DELETE n
```

- `LensExtractionGeneratedHandler`: MERGEs `(:LensItem:<Label> {item_id})` (two labels: the generic `LensItem` for lens-wide queries + the dynamic validated label), SETs lens/lens_version/node_type/confidence/model/provider/interview_id + each `fields` entry as a property (scalars and lists of scalars only — non-scalar values JSON-dumped), links `SUPPORTED_BY` → Sentences (`RETURN count(s)`; raise when supporting ids non-empty but 0 matched — out-of-order guard), and one relationship per `speaker_links` entry (relationship name validated `^[A-Z][A-Z_]*$`) → Speaker (raise if speaker missing).
- `LensExtractionOverriddenHandler`: SETs overridden fields + `locked = true` on the node (zero-write raise via `_raise_if_no_writes`).
- All three registered in bootstrap; `LensApplied`, `LensExtractionGenerated`, `LensExtractionOverridden` added to the interview allowlist.

- [ ] **Step 1: Write the failing tests** (`tests/projections/test_lens_handlers.py`):

```python
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.events.envelope import AggregateType, EventEnvelope
from src.projections.handlers.lens_handlers import (
    LensAppliedHandler,
    LensExtractionGeneratedHandler,
    LensExtractionOverriddenHandler,
)

IID = "22222222-2222-2222-2222-222222222222"
SP1 = "33333333-3333-3333-3333-333333333333"
F1 = "77777777-7777-7777-7777-777777777771"
ITEM = "88888888-8888-8888-8888-888888888801"


def make_event(event_type, data):
    return EventEnvelope(event_type=event_type, aggregate_type=AggregateType.INTERVIEW,
                         aggregate_id=IID, version=5, data=data)


def item_data(**over):
    d = {"lens": "meeting_minutes", "lens_version": 1, "node_type": "Decision",
         "item_id": ITEM, "fields": {"text": "Go with X", "made_by": "S1"},
         "supporting_fragment_ids": [F1],
         "speaker_links": [{"relationship": "DECIDED_BY", "speaker_id": SP1}],
         "confidence": 0.9, "model": "haiku", "provider": "anthropic"}
    d.update(over)
    return d


@pytest.mark.asyncio
async def test_lens_applied_deletes_only_unlocked_older_items():
    handler = LensAppliedHandler()
    tx = AsyncMock()
    await handler.apply(tx, make_event("LensApplied", {"lens": "meeting_minutes", "lens_version": 2}))
    query = tx.run.call_args[0][0]
    assert "lens_version < $lens_version" in query
    assert "coalesce(n.locked, false) = false" in query
    assert "DETACH DELETE" in query


@pytest.mark.asyncio
async def test_extraction_merges_dual_label_node_with_links():
    handler = LensExtractionGeneratedHandler()
    tx = AsyncMock()
    tx.run.return_value.single = AsyncMock(return_value={"supported": 1, "linked": 1})
    await handler.apply(tx, make_event("LensExtractionGenerated", item_data()))
    query = tx.run.call_args[0][0]
    assert "MERGE (n:LensItem:Decision {item_id: $item_id})" in query
    assert "SUPPORTED_BY" in query
    assert "DECIDED_BY" in query


@pytest.mark.asyncio
async def test_extraction_rejects_invalid_label():
    handler = LensExtractionGeneratedHandler()
    tx = AsyncMock()
    with pytest.raises(ValueError, match="label"):
        await handler.apply(tx, make_event("LensExtractionGenerated",
                                           item_data(node_type="Bad Label; DROP")))
    tx.run.assert_not_called()


@pytest.mark.asyncio
async def test_extraction_raises_when_fragments_missing():
    handler = LensExtractionGeneratedHandler()
    tx = AsyncMock()
    tx.run.return_value.single = AsyncMock(return_value={"supported": 0, "linked": 1})
    with pytest.raises(ValueError, match="not yet projected"):
        await handler.apply(tx, make_event("LensExtractionGenerated", item_data()))


@pytest.mark.asyncio
async def test_override_sets_fields_and_lock():
    handler = LensExtractionOverriddenHandler()
    tx = AsyncMock()
    counters = MagicMock(nodes_created=0, properties_set=2, relationships_created=0)
    tx.run.return_value.consume = AsyncMock(return_value=MagicMock(counters=counters))
    await handler.apply(tx, make_event("LensExtractionOverridden",
                                       {"item_id": ITEM, "fields_overridden": {"text": "Go with Y"},
                                        "note": "fixed"}))
    query = tx.run.call_args[0][0]
    assert "locked = true" in query


def test_lens_events_in_bootstrap_and_allowlists():
    from src.projections.bootstrap import create_handler_registry
    from src.projections.config import get_all_allowed_event_types

    registry = create_handler_registry(parked_events_manager=MagicMock())
    allowed = set(get_all_allowed_event_types())
    for et in ("LensApplied", "LensExtractionGenerated", "LensExtractionOverridden"):
        assert registry.has_handler(et)
        assert et in allowed
```

- [ ] **Step 2: Run to verify fail** — ModuleNotFoundError.
- [ ] **Step 3: Implement.** The extraction handler builds the query with the VALIDATED label and relationship names interpolated (post-validation only), field properties via a `props` param map (fields flattened: scalar/list-of-scalar kept, others `json.dumps`ed, key prefix-checked against reserved props), single statement ending `RETURN count(DISTINCT s) AS supported, count(DISTINCT sp) AS linked`; raise when `supporting_fragment_ids` non-empty and `supported == 0`, or `speaker_links` non-empty and `linked == 0`. Register all three in bootstrap; extend the interview allowlist; bump bootstrap-unit pins 19 → 22 and the expected-types set.
- [ ] **Step 4: Run** — `pytest tests/projections -m "not integration" -q` → green incl. drift guard.
- [ ] **Step 5: Commit** — `git commit -am "feat: generic lens projection handlers (dual-label nodes, declarative links)"`

---

### Task 8: LensEngine + CLI

**Files:**
- Create: `src/lens/engine.py`, `src/lens/__main__.py`
- Test: `tests/lens/test_engine.py`

**Interfaces:**
- `LensResult(BaseModel)`: `interview_id, lens, lens_version, items_extracted: int, items_skipped_existing: int, items_skipped_locked: int, units_processed: int`.
- `LensEngine(config_dict=None)` with `async apply(interview_id: str, lens_name: str, force: bool = False, correlation_id=None) -> LensResult`:
  1. `load_lens(lens_name)`; load Interview (ValueError if missing); load fragments (deterministic uuid5 ids, as the enrichment orchestrator does) → utterance texts (join member fragments) + full document text (all fragments in index order, speaker-labeled `[S1]: text` lines).
  2. Emit `apply_lens(lens.name, lens.version)` when `lens_runs.get(name) != version` OR force (same-version re-run without force skips the LensApplied emit — idempotent run).
  3. Build an `EnrichmentExecutor` (failover agent, `[d.to_extractor_spec() for d in lens.extractors]`, `load_yaml(lens.prompts_file)`, domain_keywords=[]); for each extractor: document scope → one `run_spec_on_text(spec, document_text)`; utterance scope → one call per utterance text; fragment scope → one call per fragment text.
  4. For each outcome with data: iterate `decl.items_field` list; deterministic `item_id = uuid5(f"{interview_id}:lens:{lens}:{node_type}:{source_unit}:{ordinal}")` (source_unit = utterance_id | fragment aggregate_id | "document"); skip item_ids already in `interview.lens_items` (count as existing) and locked ones (count as locked); resolve speaker links: for the mapping's `field`, look up the extracted value against speakers (handle OR display_name, case-insensitive; the literal `"SELF"` resolves to the source utterance's speaker); unresolved → keep raw string in fields + add `fields["<field>_unresolved"] = True`, emit no link. `supporting_fragment_ids`: utterance scope → the utterance's fragment_ids; fragment scope → that fragment; document scope → [].
  5. `record_lens_extraction(...)` per item; save interview once at the end. Actor SYSTEM `user_id="lens"`, shared correlation_id.
- CLI `python -m src.lens <interview_id> <lens_name> [--force]` printing `LensResult` JSON.

- [ ] **Step 1: Write the failing tests** (`tests/lens/test_engine.py`) — mocked repos + patched executor seam:

```python
import uuid as uuid_mod
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.enrichment.executor import SpecOutcome
from src.events.aggregates import Interview, Sentence
from src.lens.engine import LensEngine

IID = "22222222-2222-2222-2222-222222222222"
SP1 = "33333333-3333-3333-3333-333333333333"
U1 = "55555555-5555-5555-5555-555555555555"


def build_world():
    interview = Interview(IID)
    interview.create(title="t", source="s", metadata={"fragment_count": 2})
    interview.add_speaker(SP1, "S1", "Alice", True, 0.9, "inference")
    f_ids = [str(uuid_mod.uuid5(uuid_mod.NAMESPACE_DNS, f"{IID}:{i}")) for i in range(2)]
    sentences = {}
    for i, fid in enumerate(f_ids):
        s = Sentence(fid)
        s.create(interview_id=IID, index=i, text=f"Fragment {i}.")
        s.attribute_speaker(SP1, 0.9, "inference")
        s.mark_events_as_committed()
        sentences[fid] = s
    interview.identify_utterance(U1, SP1, f_ids, 0.9)
    interview.mark_events_as_committed()
    return interview, sentences


def outcome_for(spec_name):
    canned = {
        "objectives": {"objectives": [{"text": "Ship the ECU tool", "confidence": 0.9}]},
        "decisions": {"decisions": [{"text": "Go with X", "made_by": "Alice", "confidence": 0.9}]},
        "action_items": {"action_items": [{"text": "Draft the doc", "owner": "SELF",
                                           "due": None, "confidence": 0.8}]},
        "followups": {"followups": []},
    }
    return SpecOutcome(data=canned[spec_name], provider="anthropic", model="haiku")


def patch_engine(interview, sentences):
    interview_repo = MagicMock()
    interview_repo.load = AsyncMock(return_value=interview)
    interview_repo.save = AsyncMock(side_effect=lambda a, **k: a.mark_events_as_committed())
    sentence_repo = MagicMock()
    sentence_repo.load = AsyncMock(side_effect=lambda sid: sentences.get(sid))
    executor = MagicMock()
    executor.run_spec_on_text = AsyncMock(side_effect=lambda spec, text, ctx=None: outcome_for(spec.name))
    return (
        patch("src.lens.engine.get_interview_repository", return_value=interview_repo),
        patch("src.lens.engine.get_sentence_repository", return_value=sentence_repo),
        patch.object(LensEngine, "_build_executor", return_value=executor),
        executor,
    )


@pytest.mark.asyncio
async def test_apply_extracts_items_with_links_and_grounding():
    interview, sentences = build_world()
    p1, p2, p3, executor = patch_engine(interview, sentences)
    with p1, p2, p3:
        result = await LensEngine().apply(IID, "meeting_minutes")

    assert result.items_extracted == 3  # objective + decision + action item
    assert result.items_skipped_existing == 0
    assert interview.lens_runs == {"meeting_minutes": 1}
    decision = next(v for v in interview.lens_items.values() if v["node_type"] == "Decision")
    assert decision["lens"] == "meeting_minutes"
    types = {v["node_type"] for v in interview.lens_items.values()}
    assert types == {"Objective", "Decision", "ActionItem"}  # empty followups -> no item


@pytest.mark.asyncio
async def test_owner_self_resolves_to_utterance_speaker_and_alice_by_display_name():
    interview, sentences = build_world()
    captured = []
    original = interview.record_lens_extraction

    def spy(*a, **k):
        captured.append(k)
        return original(*a, **k)

    interview.record_lens_extraction = spy
    p1, p2, p3, _ = patch_engine(interview, sentences)
    with p1, p2, p3:
        await LensEngine().apply(IID, "meeting_minutes")

    by_type = {k["node_type"]: k for k in captured}
    assert by_type["Decision"]["speaker_links"] == [
        {"relationship": "DECIDED_BY", "speaker_id": SP1}  # "Alice" display-name match
    ]
    assert by_type["ActionItem"]["speaker_links"] == [
        {"relationship": "OWNED_BY", "speaker_id": SP1}  # SELF -> utterance speaker
    ]
    assert by_type["Decision"]["supporting_fragment_ids"]  # utterance fragments
    assert by_type["Objective"]["supporting_fragment_ids"] == []  # document scope


@pytest.mark.asyncio
async def test_second_run_same_version_is_idempotent():
    interview, sentences = build_world()
    p1, p2, p3, executor = patch_engine(interview, sentences)
    with p1, p2, p3:
        await LensEngine().apply(IID, "meeting_minutes")
        result2 = await LensEngine().apply(IID, "meeting_minutes")

    assert result2.items_extracted == 0
    assert result2.items_skipped_existing == 3
    assert len([i for i in interview.lens_items]) == 3  # no duplicates


@pytest.mark.asyncio
async def test_locked_items_survive_forced_rerun():
    interview, sentences = build_world()
    p1, p2, p3, _ = patch_engine(interview, sentences)
    with p1, p2, p3:
        await LensEngine().apply(IID, "meeting_minutes")
        locked_id = next(iid for iid, v in interview.lens_items.items()
                         if v["node_type"] == "Decision")
        interview.override_lens_extraction(locked_id, {"text": "Go with Y"})
        interview.mark_events_as_committed()
        result = await LensEngine().apply(IID, "meeting_minutes", force=True)

    assert result.items_skipped_locked >= 1
    assert interview.lens_items[locked_id]["locked"] is True
```

- [ ] **Step 2: Run to verify fail** — ModuleNotFoundError.
- [ ] **Step 3: Implement** `src/lens/engine.py` per the Interfaces block (structure mirrors `EnrichmentOrchestrator`: `_build_executor` patchable seam via `get_failover_agent` + lens specs + `load_yaml(lens.prompts_file)`; `_load_world` reusing the deterministic-uuid5 fragment loading; `_resolve_speaker(value, utterance_speaker_id, interview)`; `_source_units(decl, world)` yielding (source_unit_key, text, utterance_speaker_id, supporting_ids)). Forced same-version re-run: emit `apply_lens` only when version changes OR force (aggregate permits same-version re-apply — verify `apply_lens` guard allows `>=`); items already present are skipped (idempotent) unless... NOTE: force + same version still skips existing UNLOCKED item_ids in the aggregate (they raise on duplicate) — the LensApplied handler only deletes GRAPH nodes with `lens_version <`. For force-same-version the engine emits `override`-free re-records only for item_ids NOT in lens_items; document this limitation in the CLI help ("--force refreshes results for a NEW lens version; same-version force re-extracts only novel items"). `src/lens/__main__.py` mirrors `src/enrichment/__main__.py`.
- [ ] **Step 4: Run** — `pytest tests/lens -q --no-cov` → green; then `pytest tests -m "not integration" -q` full suite.
- [ ] **Step 5: Commit** — `git commit -am "feat: LensEngine with owner resolution and idempotent re-runs; python -m src.lens CLI"`

---

### Task 9: Corrections endpoint

**Files:**
- Create: `src/api/routers/lenses.py`
- Modify: `src/main.py` (include router)
- Test: `tests/api/test_lenses_router.py`

**Interfaces:**
- `POST /lenses/{interview_id}/items/{item_id}/override` — body `{"fields_overridden": {...}, "note": "..."}`; 202 `{"status": "accepted", "version": <n>}`; 404 missing interview; 409 domain ValueError; actor HUMAN from `X-User-ID` header (`_human_actor` pattern from `src/api/routers/speakers.py`).

- [ ] **Step 1: Write the failing tests** (`tests/api/test_lenses_router.py`):

```python
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.main import app

IID = "22222222-2222-2222-2222-222222222222"
ITEM = "88888888-8888-8888-8888-888888888801"


@pytest.fixture
def client():
    return TestClient(app)


def make_repo(load_result):
    repo = MagicMock()
    repo.load = AsyncMock(return_value=load_result)
    repo.save = AsyncMock()
    return repo


def test_override_returns_202_with_actor_from_header(client):
    interview = MagicMock()
    interview.version = 7
    repo = make_repo(interview)
    with patch("src.api.routers.lenses.get_interview_repository", return_value=repo):
        resp = client.post(
            f"/lenses/{IID}/items/{ITEM}/override",
            json={"fields_overridden": {"text": "Go with Y"}, "note": "fixed"},
            headers={"X-User-ID": "nathan"},
        )
    assert resp.status_code == 202
    assert resp.json()["version"] == 7
    actor = interview.override_lens_extraction.call_args.kwargs["actor"]
    assert actor.user_id == "nathan"


def test_override_missing_interview_404(client):
    with patch("src.api.routers.lenses.get_interview_repository", return_value=make_repo(None)):
        resp = client.post(f"/lenses/{IID}/items/{ITEM}/override",
                           json={"fields_overridden": {"text": "x"}})
    assert resp.status_code == 404


def test_override_unknown_item_409(client):
    interview = MagicMock()
    interview.override_lens_extraction.side_effect = ValueError("Unknown lens item")
    with patch("src.api.routers.lenses.get_interview_repository", return_value=make_repo(interview)):
        resp = client.post(f"/lenses/{IID}/items/{ITEM}/override",
                           json={"fields_overridden": {"text": "x"}})
    assert resp.status_code == 409
```

- [ ] **Step 2: Run to verify fail.**
- [ ] **Step 3: Implement** the router (mirror `speakers.py`: `_human_actor`, `_load_interview` helper, 202/404/409 mapping) + include in `src/main.py`.
- [ ] **Step 4: Run** — `pytest tests/api -m "not integration" -q` → green.
- [ ] **Step 5: Commit** — `git commit -am "feat: lens item override endpoint (corrections surface)"`

---

### Task 10: Integration smoke — lens end-to-end

**Files:**
- Create: `tests/integration/test_layer3_lens_smoke.py`

**Interfaces:** none new. Mirrors the Layer 2 smoke: ingest the small labeled transcript against live ESDB, apply the lens with a mocked executor (canned SpecOutcomes — no live LLM), replay ALL events in `occurred_at` order through the real handler registry into real Neo4j, then assert:

```cypher
MATCH (d:LensItem:Decision {interview_id: $iid})-[:DECIDED_BY]->(sp:Speaker) ...
MATCH (a:LensItem:ActionItem {interview_id: $iid})-[:SUPPORTED_BY]->(s:Sentence) ...
MATCH (o:LensItem:Objective {interview_id: $iid}) ...
```

(counts per canned data; all queries scoped by `interview_id` — the persistent test DB lesson). Marked `@pytest.mark.integration`; runner notes: `COMPOSE_PROJECT_NAME=interview_analyzer_chaining`, `NEO4J_URI=bolt://localhost:7688 NEO4J_USER=neo4j NEO4J_PASSWORD=testpassword ESDB_CONNECTION_STRING="esdb://localhost:2113?tls=false"`.

- [ ] **Step 1: Write the test** (transcribe the Layer 2 smoke structure: `tests/integration/test_layer2_enrichment_smoke.py`, swapping enrichment for `LensEngine.apply` with `monkeypatch.setattr(LensEngine, "_build_executor", ...)` returning canned outcomes; resolve the real utterance id from the aggregate as that smoke does).
- [ ] **Step 2: Run against live infra** (`make test-infra-up` first if down): expected PASS.
- [ ] **Step 3: Commit** — `git commit -am "test: Layer 3 lens smoke (end-to-end through real projection)"`

---

### Task 11: Docs

**Files:**
- Modify: `docs/ROADMAP.md` (M4.3 milestone ✅ with checklist mirroring Tasks 1–10; Quick Status; Current Phase → M4.4 planning; decision log: generic-engine decision, debt burndown completion), `docs/architecture/database-schema.md` (LensItem dual-label nodes, SUPPORTED_BY/dynamic speaker relationships, the 3 event types, per-model embedding properties note from Task 1), `README.md` (lens step in What-It-Does; `python -m src.lens` command)

- [ ] **Step 1: Edit all three docs.**
- [ ] **Step 2: Full unit suite green** — final check.
- [ ] **Step 3: Commit** — `git commit -am "docs: M4.3 milestone, lens schema, README"`

---

## Verification (whole-plan)

1. Full unit suite green; drift guard green (pins 22).
2. Integration: both existing smokes + the new lens smoke against live infra.
3. Manual e2e (live LLM, Anthropic): `python -m src.ingestion data/input/GMT20231026-210203_Recording.txt --enrich` then `python -m src.lens <interview_id> meeting_minutes`, then in Neo4j: `MATCH (a:ActionItem)-[:OWNED_BY]->(sp:Speaker) RETURN sp.display_name, a.text, a.due`.
4. Zero-per-lens-code check: confirm a hypothetical second lens needs only YAML+prompts (no Python diffs beyond `lenses/`+`prompts/`).

## Deferred / explicitly out of scope for M4.3

- persona lens (next lens, after meeting_minutes proves the engine).
- Lens apply via ingest flag / API endpoint (owner deferred to post-v1).
- Same-version `--force` full re-extraction of unlocked items (needs an item-clearing event; documented CLI limitation).
- OKF export of lens outputs (M4.4 — Layer 5).
