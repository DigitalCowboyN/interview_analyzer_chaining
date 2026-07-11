# M4.5a: Debt Burndown + Fragment Rename Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Burn down the M4.4-exit debt, then land the Sentence→Fragment rename as a dual-label overlay with a migration CLI — so every M4.5b/c handler is born writing `:Fragment`.

**Architecture:** Debt first (renderer text-safety, slug uniquification, reader caps, typed 422, staged atomic writes off the event loop, test gaps). Then the rename in three moves: writers anchor MERGE on `:Sentence` and `SET s:Fragment` (safe against pre-existing nodes), a one-shot idempotent migration relabels history, then all reads flip to `:Fragment`. Wire format (event types, `aggregate_type` values, stream names) is frozen forever.

**Tech Stack:** Python 3.10, Pydantic v2, Neo4j 5.26, FastAPI. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-07-10-layer4-schema-v2-design.md` (M4.5a section — the debt list there is verbatim scope).

## Global Constraints

- **Wire format frozen:** event type names (`SentenceCreated`, …), `aggregate_type` values (`"Sentence"`), and stream names (`Sentence-{id}`) never change. The rename lives in graph labels, code names, and docs only.
- **Dual-label safety:** writer queries MUST keep matching on `:Sentence` (`MERGE (s:Sentence {sentence_id: $id}) SET s:Fragment` — NEVER `MERGE (s:Fragment:Sentence {...})`, which would duplicate nodes that lack the new label). The `:Sentence` label continues to be written through all of M4.5.
- **Vector index DDL stays on `:Sentence`** for M4.5 (nodes carry both labels so the indexes keep serving; re-targeting duplicates indexes — deferred to the shim-label drop).
- Read-side queries we own flip to `:Fragment` only AFTER the migration task lands (task order is load-bearing).
- OKF conformance rules from M4.4 continue to bind the renderer (frontmatter `type` on non-reserved files; `index.md` frontmatter-less; bundle-absolute links).
- Environment: unit tests via `./scripts/test.sh [pytest args]` (defaults to the full unit suite); integration via `./scripts/test-integration.sh <paths> -q --no-cov` (test infra: Neo4j bolt://localhost:7688, ESDB localhost:2113; bring up with `COMPOSE_PROJECT_NAME=interview_analyzer_chaining make test-infra-up`). Do NOT invoke python/pytest directly (not on PATH).
- Style: flake8 clean (max line 120) on all touched files; follow existing module patterns.

## File Structure (touched)

```
src/export/renderer.py        # T1 link/cell text safety + DRY single pass; T2 slug registry
src/export/reader.py          # T3 rollup scan caps
src/export/bundler.py         # T3 LensNeverAppliedError; T4 staged atomic write via thread
src/api/routers/exports.py    # T3 422 mapping; T4 (no change needed if bundler offloads)
src/projections/handlers/*.py # T6 writers SET :Fragment; T7 reads -> :Fragment
src/projections/migrate_fragment_label.py  # T6 migration CLI (new)
src/events/aggregates.py      # T8 class rename + alias
src/events/repository.py      # T8 get_fragment_repository + alias
tests/...                     # per task
```

---

### Task 1: Renderer text safety + single-pass DRY

**Files:**
- Modify: `src/export/renderer.py`
- Test: `tests/export/test_renderer.py`

**Interfaces:**
- Produces: `_link_text(value: str, limit: int = 80) -> str` — collapses all whitespace runs (incl. newlines) to single spaces, escapes `[` and `]` as `\[`/`\]`, strips, truncates to `limit`. Used for EVERY markdown link label built from LLM/graph text (index entries, speaker back-links, relationship lines, grounding attributions).
- Produces: `_cell(value) -> str` — `str(value)`, whitespace runs collapsed to spaces, `|` escaped as `\|`; used for every `analysis.md` table cell.
- The item `(path, title)` derivation happens ONCE per item in `render_bundle` and is reused by the item file loop, index sections, and speaker back-link references (removes the duplicated derivation flagged in review).

- [ ] **Step 1: Write the failing tests** (append to `tests/export/test_renderer.py`; reuse its existing HEADER/TRANSCRIPT/... fixtures and `render()` helper):

```python
def test_link_text_escapes_and_collapses():
    from src.export.renderer import _link_text

    assert _link_text("line one\nline two") == "line one line two"
    assert _link_text("a [bracketed] thing") == "a \\[bracketed\\] thing"
    assert len(_link_text("x" * 200)) == 80


def test_table_cells_escape_pipes_and_newlines():
    import copy

    analysis = copy.deepcopy(ANALYSIS)
    analysis[0]["text"] = "cell | with pipe\nand newline"
    lens = load_lens("meeting_minutes")
    files = dict(render_bundle(HEADER, TRANSCRIPT, SPEAKERS, ITEMS, CLAIMS,
                               ENTITIES, analysis, lens, exported_at="2026-07-10T12:00:00+00:00"))
    table_lines = [ln for ln in files["analysis.md"].splitlines() if "cell" in ln]
    assert table_lines, "analysis row missing"
    assert "\\|" in table_lines[0] and "\n" not in table_lines[0]


def test_index_link_labels_survive_hostile_item_text():
    import copy

    items = copy.deepcopy(ITEMS)
    items[0]["props"]["text"] = "Decide [now]\nor never | maybe"
    lens = load_lens("meeting_minutes")
    files = dict(render_bundle(HEADER, TRANSCRIPT, SPEAKERS, items, CLAIMS,
                               ENTITIES, ANALYSIS, lens, exported_at="2026-07-10T12:00:00+00:00"))
    index = files["index.md"]
    # the link label must not contain raw newlines or unescaped brackets
    label_line = next(ln for ln in index.splitlines() if "decision-" in ln)
    assert "\n" not in label_line and "[now]" not in label_line
```

- [ ] **Step 2: Run to verify fail** — `./scripts/test.sh tests/export/test_renderer.py -q --no-cov` → new tests FAIL (`_link_text` undefined; raw text leaks).

- [ ] **Step 3: Implement.** Add near the other helpers in `renderer.py`:

```python
def _link_text(value: str, limit: int = 80) -> str:
    """LLM/graph text used as a markdown link label: one line, brackets escaped."""
    collapsed = re.sub(r"\s+", " ", str(value)).strip()
    escaped = collapsed.replace("[", "\\[").replace("]", "\\]")
    return escaped[:limit]


def _cell(value: Any) -> str:
    """LLM/graph text used as a markdown table cell: one line, pipes escaped."""
    collapsed = re.sub(r"\s+", " ", str(value)).strip()
    return collapsed.replace("|", "\\|")
```

Then: (a) route every link label through `_link_text` — the index section entries, `_render_speaker`'s references list, `_render_lens_item`/`_render_claim` relationship + grounding attribution labels, and the existing `title = (text[:80] ...)` derivations become `_link_text(text) or item_id`; (b) route every `_render_analysis` cell through `_cell`; (c) in `render_bundle`, compute each item's `(rel_path, title)` exactly once into a list and pass it to the item loop, index sections, and speaker references (delete the duplicated derivations).

- [ ] **Step 4: Run** — `./scripts/test.sh tests/export -q --no-cov` → all green (existing conformance/link tests must pass unchanged).
- [ ] **Step 5: Commit** — `git add -u src/export tests/export && git commit -m "fix(renderer): escape LLM text in link labels and table cells; single-pass item derivation"`

---

### Task 2: Slug uniquification

**Files:**
- Modify: `src/export/renderer.py`
- Test: `tests/export/test_renderer.py`

**Interfaces:**
- Produces: `class _SlugRegistry` with `slug_for(value: str) -> str` — first caller of a given slug gets it verbatim; collisions get `-2`, `-3`, …; an empty result (punctuation-only input) falls back to `x-<sha1(value)[:8]>`. Same registry instance shared across speakers AND entities within one `render_bundle` call (bundle-wide uniqueness), so a speaker `ECU` and entity `ecu` cannot collide either.

- [ ] **Step 1: Write the failing tests:**

```python
def test_slug_registry_uniquifies_and_hashes():
    from src.export.renderer import _SlugRegistry

    reg = _SlugRegistry()
    assert reg.slug_for("ECU") == "ecu"
    assert reg.slug_for("ecu") == "ecu-2"          # collision
    assert reg.slug_for("ECU!") == "ecu-3"          # normalizes then collides
    hashed = reg.slug_for("!!!")
    assert hashed.startswith("x-") and len(hashed) == 10


def test_bundle_entity_slug_collision_yields_two_files():
    import copy

    entities = copy.deepcopy(ENTITIES) + [
        {"surface": "ECU", "entity_type": "product", "mentions": []}
    ]  # ENTITIES already has surface "ecu"
    lens = load_lens("meeting_minutes")
    files = dict(render_bundle(HEADER, TRANSCRIPT, SPEAKERS, ITEMS, CLAIMS,
                               entities, ANALYSIS, lens, exported_at="2026-07-10T12:00:00+00:00"))
    entity_files = [p for p in files if p.startswith("entities/")]
    assert len(entity_files) == 2 and len(set(entity_files)) == 2
```

- [ ] **Step 2: Run to verify fail.**

- [ ] **Step 3: Implement:**

```python
class _SlugRegistry:
    """Bundle-wide unique slugs: collisions suffixed, empty slugs hashed."""

    def __init__(self) -> None:
        self._taken: set = set()

    def slug_for(self, value: str) -> str:
        base = slugify(value)
        if not base:
            base = "x-" + hashlib.sha1(value.encode("utf-8")).hexdigest()[:8]
        candidate, n = base, 1
        while candidate in self._taken:
            n += 1
            candidate = f"{base}-{n}"
        self._taken.add(candidate)
        return candidate
```

(`import hashlib` at module top.) In `render_bundle`, create ONE `_SlugRegistry` and use `slug_for` for every speaker file and entity file; keep a `value -> slug` dict per kind so links (`/speakers/<slug>.md`) resolve to the same slug the file got.

- [ ] **Step 4: Run** — `./scripts/test.sh tests/export -q --no-cov` → green.
- [ ] **Step 5: Commit** — `git add -u src/export tests/export && git commit -m "fix(renderer): bundle-wide slug uniquification with hash fallback"`

---

### Task 3: Reader caps + never-applied-lens 422 + rollup behavioral test

**Files:**
- Modify: `src/export/reader.py` (`speaker_rollup_rows` scan caps), `src/export/bundler.py` (new error), `src/api/routers/exports.py` (mapping)
- Test: `tests/export/test_reader.py`, `tests/export/test_bundler.py`, `tests/api/test_exports_router.py`

**Interfaces:**
- Produces: `class LensNeverAppliedError(ValueError)` in `src/export/bundler.py`; `_guard` raises it when `interview.lens_runs.get(lens_name) is None` (lens valid but never applied to this interview) BEFORE the expected/projected comparison. Router maps it to 422 with its message.
- `speaker_rollup_rows`: both Cypher queries gain `LIMIT $scan_cap` (parameter, default `5000`) with a comment that grouping/pagination happen in Python over a bounded scan; signature gains `scan_cap: int = 5000`.

- [ ] **Step 1: Write the failing tests.**

`tests/export/test_bundler.py` (reuse `make_interview`/`patch_world` helpers; build the interview WITHOUT `apply_lens`):

```python
@pytest.mark.asyncio
async def test_never_applied_lens_raises_422_error(tmp_path):
    from src.export.bundler import LensNeverAppliedError

    interview = Interview(IID)
    interview.create(title="t", source="s", metadata={"fragment_count": 1})
    interview.mark_events_as_committed()
    patches = patch_world(interview, projected_items=[])
    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7]:
        with pytest.raises(LensNeverAppliedError, match="never applied"):
            await OkfExporter().export(IID, "meeting_minutes", out_dir=str(tmp_path))
```

`tests/api/test_exports_router.py` — extend the parametrized error-mapping test with `(LensNeverAppliedError("lens 'meeting_minutes' never applied to this interview"), 422)` (import it from `src.export.bundler`).

`tests/export/test_reader.py`:

```python
@pytest.mark.asyncio
async def test_rollup_queries_carry_scan_cap():
    session = make_session([])
    await reader.speaker_rollup_rows(session, scan_cap=123)
    for call in session.run.call_args_list:
        assert "LIMIT $scan_cap" in call[0][0]
        assert call.kwargs["scan_cap"] == 123


@pytest.mark.asyncio
async def test_rollup_substring_filter_on_grouped_data():
    rows = [
        {"display_name": "Alice Johnson", "node_type": "ActionItem", "relationship": "OWNED_BY",
         "text": "t", "interview_id": "i1", "item_id": "x1"},
        {"display_name": "Bob Reyes", "node_type": "Decision", "relationship": "DECIDED_BY",
         "text": "t", "interview_id": "i1", "item_id": "x2"},
    ]
    # first session.run call (items query) returns rows; second (claims) returns empty
    results = [rows, []]

    call_count = {"n": 0}

    def run_side_effect(query, **kw):
        res = MagicMock()
        data = results[min(call_count["n"], 1)]
        call_count["n"] += 1

        async def aiter(self):
            for r in data:
                yield r
        res.__aiter__ = aiter
        return res

    session = MagicMock()
    session.run = AsyncMock(side_effect=run_side_effect)
    grouped = await reader.speaker_rollup_rows(session, name="ali")
    assert [g["display_name"] for g in grouped] == ["Alice Johnson"]
```

- [ ] **Step 2: Run to verify fail.**
- [ ] **Step 3: Implement.** Bundler: add the class next to the other two error types and the guard clause `if current is None: raise LensNeverAppliedError(f"lens {lens_name!r} never applied to this interview")` as the FIRST line of `_guard`. Router: catch `LensNeverAppliedError` before the generic `ValueError` branch → `HTTPException(422, str(e))`. Reader: add `scan_cap: int = 5000` param, append `LIMIT $scan_cap` to both rollup queries, bind it, one comment line: `# bounded scan: grouping/pagination happen in Python; raise scan_cap for very large projects`.
- [ ] **Step 4: Run** — `./scripts/test.sh tests/export tests/api -q --no-cov` → green.
- [ ] **Step 5: Commit** — `git add -u src/export src/api tests && git commit -m "fix(export): never-applied-lens 422; rollup scan caps; rollup filter behavioral test"`

---

### Task 4: Staged atomic bundle writes, off the event loop

**Files:**
- Modify: `src/export/bundler.py`
- Test: `tests/export/test_bundler.py`

**Interfaces:**
- Produces: `OkfExporter._write_bundle(bundle_dir: Path, files, log_content, zip_bundle: bool) -> str` — a SYNC method doing ALL file IO: write everything into a sibling temp dir (`bundle_dir.with_name(bundle_dir.name + ".staging")`, cleaned if left over), then swap — `rmtree(bundle_dir)` + `staging.rename(bundle_dir)` — so a failure mid-write leaves the OLD bundle intact (the write window shrinks to the rename). Zip built after the swap. Returns `bundle_path`.
- `export()` calls it via `await asyncio.to_thread(self._write_bundle, ...)` — no sync IO left on the event loop.

- [ ] **Step 1: Write the failing test:**

```python
@pytest.mark.asyncio
async def test_failed_write_preserves_previous_bundle(tmp_path, monkeypatch):
    patches = patch_world(make_interview(), PROJECTED)
    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7]:
        await OkfExporter().export(IID, "meeting_minutes", out_dir=str(tmp_path))
        bundle = tmp_path / f"{IID}-meeting_minutes"
        original_index = (bundle / "index.md").read_text()

        real_write = Path.write_text
        calls = {"n": 0}

        def flaky_write(self, *a, **k):
            calls["n"] += 1
            if calls["n"] >= 3:
                raise OSError("disk full")
            return real_write(self, *a, **k)

        monkeypatch.setattr(Path, "write_text", flaky_write)
        with pytest.raises(OSError):
            await OkfExporter().export(IID, "meeting_minutes", out_dir=str(tmp_path))
        monkeypatch.setattr(Path, "write_text", real_write)

    assert (bundle / "index.md").read_text() == original_index  # old bundle intact
```

- [ ] **Step 2: Run to verify fail** (current code rmtree's the old bundle before writing → old index.md gone).
- [ ] **Step 3: Implement** per the Interfaces block: extract the current write/zip section of `export()` into `_write_bundle` with the staging-dir + swap logic (`shutil.rmtree(staging, ignore_errors=True)` first to clear leftovers; write files into staging; `if bundle_dir.exists(): shutil.rmtree(bundle_dir)`; `staging.rename(bundle_dir)`; then zip as today), and call it with `await asyncio.to_thread(...)` (`import asyncio` at top). `_log_entry` (reads the OLD log) must still run BEFORE the swap — keep its call where it is.
- [ ] **Step 4: Run** — `./scripts/test.sh tests/export tests/api -q --no-cov` → green (existing re-export/log tests unchanged).
- [ ] **Step 5: Commit** — `git add -u src/export tests/export && git commit -m "fix(bundler): staged atomic bundle writes via thread offload"`

---

### Task 5: Flat-path offsets test

**Files:**
- Test: `tests/ingestion/test_front_matter.py`

- [ ] **Step 1: Parametrize the invariant test.** Replace `test_normalize_preserves_offsets_invariant_with_front_matter`'s single body with a parametrized version covering LABELED and FLAT:

```python
FM_FLAT_TEXT = """---
title: Flat Notes
participants: [Alice Johnson]
---
This is plain prose without speaker labels. It has several sentences.
Nobody is labeled here at all. The segmenter must still ground offsets.
"""


@pytest.mark.parametrize("text", [FM_TEXT, FM_FLAT_TEXT], ids=["labeled", "flat"])
def test_normalize_preserves_offsets_invariant_with_front_matter(text):
    transcript = normalize(text)
    assert transcript.front_matter is not None
    assert len(transcript.fragments) >= 2
    for frag in transcript.fragments:
        assert text[frag.start_char:frag.end_char] == frag.text
```

- [ ] **Step 2: Run** — `./scripts/test.sh tests/ingestion -q --no-cov` → green (this should pass immediately — the shift code is shared; the test closes the coverage gap. If FLAT fails, that is a real bug: STOP and report).
- [ ] **Step 3: Commit** — `git add -u tests/ingestion && git commit -m "test: offsets invariant with front matter parametrized for the flat path"`

---

### Task 6: Dual-label writers + migration CLI

**Files:**
- Modify: `src/projections/handlers/sentence_handlers.py` (the `MERGE (s:Sentence {sentence_id: $sentence_id})` create query)
- Create: `src/projections/migrate_fragment_label.py`
- Test: `tests/projections/test_projection_handlers_unit.py` (or the handler test file covering SentenceCreated), `tests/projections/test_fragment_migration.py`

**Interfaces:**
- The ONLY writer that creates Sentence nodes is `SentenceCreatedHandler` (every other handler MATCHes). Its create query gains `SET s:Fragment` — anchored MERGE stays on `:Sentence` (Global Constraints: never `MERGE (s:Fragment:Sentence ...)`).
- Produces: `python -m src.projections.migrate_fragment_label` — runs `MATCH (s:Sentence) WHERE NOT s:Fragment SET s:Fragment` in batches (`CALL {} IN TRANSACTIONS OF 1000 ROWS` form), prints `{"relabeled": <n>}`, idempotent (second run relabels 0).

- [ ] **Step 1: Write the failing tests.** Handler test — find the existing SentenceCreated apply test and add:

```python
@pytest.mark.asyncio
async def test_sentence_created_sets_fragment_label():
    # reuse the file's existing handler/tx fixtures for SentenceCreatedHandler
    ...  # call apply as the neighboring test does
    query = tx.run.call_args_list[0][0][0]
    assert "MERGE (s:Sentence {sentence_id: $sentence_id})" in query  # anchor unchanged
    assert "s:Fragment" in query  # dual label
```

(Adapt the fixture names to the file's existing SentenceCreated test — copy its arrange block verbatim; the two assertions above are the requirement.)

`tests/projections/test_fragment_migration.py`:

```python
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.projections.migrate_fragment_label import migrate


@pytest.mark.asyncio
async def test_migrate_runs_batched_relabel_and_reports_count():
    session = MagicMock()
    record = {"relabeled": 42}
    result = MagicMock()
    result.single = AsyncMock(return_value=record)
    session.run = AsyncMock(return_value=result)
    count = await migrate(session)
    query = session.run.call_args[0][0]
    assert "MATCH (s:Sentence)" in query and "WHERE NOT s:Fragment" in query
    assert "SET s:Fragment" in query and "IN TRANSACTIONS" in query
    assert count == 42
```

- [ ] **Step 2: Run to verify fail.**
- [ ] **Step 3: Implement.** Handler: add `SET s:Fragment` immediately after the MERGE line (before the version-guard WITH). Migration module:

```python
"""One-shot idempotent migration: add :Fragment to every :Sentence node.

Usage: python -m src.projections.migrate_fragment_label
Safe to re-run (relabels 0 the second time). The :Sentence label is retained
through M4.5 as a deprecation shim; wire format (event/stream names) is frozen.
"""

import asyncio
import json

from src.utils.neo4j_driver import Neo4jConnectionManager

QUERY = """
MATCH (s:Sentence)
WHERE NOT s:Fragment
CALL {
    WITH s
    SET s:Fragment
} IN TRANSACTIONS OF 1000 ROWS
RETURN count(s) AS relabeled
"""


async def migrate(session) -> int:
    result = await session.run(QUERY)
    record = await result.single()
    return record["relabeled"] if record else 0


async def main() -> None:
    async with await Neo4jConnectionManager.get_session() as session:
        relabeled = await migrate(session)
    print(json.dumps({"relabeled": relabeled}))


if __name__ == "__main__":
    asyncio.run(main())
```

NOTE for the implementer: `CALL {} IN TRANSACTIONS` requires an implicit (auto-commit) transaction — `session.run` provides that; do NOT wrap in `session.execute_write`. If the driver/Neo4j version rejects the syntax in the integration environment (Task 9 verifies live), fall back to a plain `MATCH (s:Sentence) WHERE NOT s:Fragment SET s:Fragment RETURN count(s) AS relabeled` and note it — dataset sizes here don't need batching yet.

- [ ] **Step 4: Run** — `./scripts/test.sh tests/projections -q --no-cov` → green.
- [ ] **Step 5: Commit** — `git add -u src/projections tests/projections && git add src/projections/migrate_fragment_label.py tests/projections/test_fragment_migration.py && git commit -m "feat(rename): dual-label Fragment writes + idempotent migration CLI"`

---

### Task 7: Flip all reads to :Fragment

**Files:**
- Modify (every `:Sentence` in a MATCH becomes `:Fragment`; MERGE anchors from Task 6 stay `:Sentence`):
  - `src/export/reader.py` (5 sites)
  - `src/projections/handlers/claim_handlers.py` (1), `embedding_handlers.py` (1 MATCH — the vector-index DDL string stays `:Sentence` per Global Constraints), `entity_handlers.py` (2), `lens_handlers.py` (1), `speaker_handlers.py` (3), `utterance_handlers.py` (1), `sentence_handlers.py` (remaining 5 non-create MATCHes)
- Test: update the query-shape assertions in `tests/export/test_reader.py`, `tests/projections/test_lens_handlers.py`, `tests/projections/test_entity_claim_handlers.py`, `tests/projections/test_embedding_handlers.py`, and any other test asserting `(s:Sentence` in a MATCH — they now assert `(s:Fragment` (create-path assertions from Task 6 keep `:Sentence`).

**Interfaces:** none new — this is the mechanical flip. Exemplar (claim_handlers.py):

```cypher
MATCH (s:Fragment)-[:PART_OF_UTTERANCE]->(u)
```

(was `MATCH (s:Sentence)-...`). Every flipped MATCH keeps its property anchors (`{aggregate_id: $...}` etc.) unchanged.

- [ ] **Step 1: Flip the test assertions first** (grep each listed test file for `:Sentence` in MATCH-shape assertions; change to `:Fragment`) and run `./scripts/test.sh tests/projections tests/export -q --no-cov` → the flipped assertions FAIL against unflipped queries (RED).
- [ ] **Step 2: Flip the source queries** in the files listed above. Verify with `grep -rn ":Sentence" src/ --include="*.py"`: remaining hits must be ONLY (a) the Task 6 MERGE anchor + `SET s:Fragment`, (b) the vector-index DDL in embedding_handlers, (c) the migration CLI, (d) comments/docstrings.
- [ ] **Step 3: Run** — `./scripts/test.sh` (full unit suite) → green.
- [ ] **Step 4: Commit** — `git add -u src tests && git commit -m "feat(rename): all owned reads query :Fragment (writers keep :Sentence anchor + shim)"`

---

### Task 8: Code-surface rename with deprecation aliases

**Files:**
- Modify: `src/events/aggregates.py` (class rename + alias), `src/events/repository.py` (factory rename + alias)
- Test: `tests/events/test_fragment_alias.py`

**Interfaces:**
- Produces: `class Fragment(AggregateRoot)` (the class previously named `Sentence`, renamed in place — its `AggregateType.SENTENCE` usage, event types, and stream naming are wire format and DO NOT change); module-level `Sentence = Fragment` alias with a deprecation comment (removal tracked on the backlog with the shim-label drop).
- Produces: `get_fragment_repository()` (the function previously `get_sentence_repository`, renamed) plus `get_sentence_repository = get_fragment_repository` alias. Same for any `create_sentence_repository` factory method → add `create_fragment_repository` alias TO THE NEW NAME (rename the implementation, alias the old name).
- All `src/` imports move to the new names (`grep -rn "get_sentence_repository\|Sentence(" src/` and update call sites; tests may keep old names — the aliases guarantee they pass unchanged, which itself proves the aliases work).

- [ ] **Step 1: Write the failing test:**

```python
def test_fragment_rename_aliases():
    from src.events.aggregates import Fragment, Sentence
    from src.events.repository import get_fragment_repository, get_sentence_repository

    assert Sentence is Fragment
    assert get_sentence_repository is get_fragment_repository
    # wire format frozen: the aggregate still stamps "Sentence"
    f = Fragment("77777777-7777-7777-7777-777777777771")
    f.create(interview_id="22222222-2222-2222-2222-222222222222", index=0, text="Hi.")
    event = f.get_uncommitted_events()[0]
    assert event.aggregate_type.value == "Sentence"
    assert event.event_type == "SentenceCreated"
```

- [ ] **Step 2: Run to verify fail** (ImportError: Fragment).
- [ ] **Step 3: Implement** per Interfaces: rename class + factory, add aliases with a one-line comment each (`# deprecated alias — wire format keeps "Sentence"; removal rides the :Sentence shim-label drop`), update `src/` call sites (`src/ingestion/orchestrator.py`, `src/enrichment/orchestrator.py`, `src/lens/engine.py`, `src/export/bundler.py`, `src/main.py`, `src/api/routers/*.py` — grep is authoritative), leave `tests/` untouched.
- [ ] **Step 4: Run** — `./scripts/test.sh` (full unit suite) → green with tests unmodified (proves the aliases).
- [ ] **Step 5: Commit** — `git add -u src && git add tests/events/test_fragment_alias.py && git commit -m "feat(rename): Fragment aggregate + repository names; deprecated Sentence aliases; wire format frozen"`

---

### Task 9: Live verification + smokes to :Fragment + docs

**Files:**
- Modify: `tests/integration/test_layer1_projection_smoke.py`, `test_layer2_enrichment_smoke.py`, `test_layer3_lens_smoke.py`, `test_layer5_export_smoke.py` (Cypher asserts → `:Fragment`), `tests/integration/test_idempotency.py`, `test_projection_replay.py`, `test_neo4j_data_integrity.py` (same flip)
- Modify: `docs/architecture/database-schema.md` (Fragment is the node's primary name; `:Sentence` shim + frozen wire format note), `docs/ROADMAP.md` (M4.5 marked ⏳ In Progress with the a/b/c plan structure; M4.5a checklist)

- [ ] **Step 1: Flip the integration tests' `:Sentence` MATCHes to `:Fragment`** (the smokes create NEW nodes through the dual-label writer, so `:Fragment` matches without migration; `test_neo4j_data_integrity.py` and replay tests likewise go through the handler path).
- [ ] **Step 2: Live run** — infra up (`COMPOSE_PROJECT_NAME=interview_analyzer_chaining make test-infra-up` if down), then: `./scripts/test-integration.sh tests/integration/test_layer1_projection_smoke.py tests/integration/test_layer2_enrichment_smoke.py tests/integration/test_layer3_lens_smoke.py tests/integration/test_layer5_export_smoke.py -q --no-cov` → 4 passed. Then run the migration against the test DB: `set -a; source .env; set +a; NEO4J_URI=bolt://localhost:7688 NEO4J_USER=neo4j NEO4J_PASSWORD=testpassword ~/.pyenv/versions/3.10.7/bin/python -m src.projections.migrate_fragment_label` → prints `{"relabeled": <n>}` (n ≥ 0); run it AGAIN → `{"relabeled": 0}` (idempotence proven live). If the batched-transactions syntax errors here, apply Task 6's documented fallback and re-run.
- [ ] **Step 3: Docs.** database-schema.md: rename the `:Sentence` node-type section heading to `:Fragment` (note: "carries the deprecated `:Sentence` shim label through M4.5; event types and `Sentence-{id}` streams are frozen wire format"); update relationship snippets to `(:Fragment)`. ROADMAP: Quick Status row `M4.5 | ⏳ In Progress | Layer 4: schema v2 (a: debt+rename ✅, b: resolution, c: segments)`; add an M4.5a checklist section above M4.4 mirroring Tasks 1-8; Current Phase → "M4.5b (resolution core)".
- [ ] **Step 4: Full unit suite** — `./scripts/test.sh` → green; flake8 clean on all files this plan touched.
- [ ] **Step 5: Commit** — `git add -u tests docs && git commit -m "test+docs: smokes and schema docs on :Fragment; migration verified live; M4.5a checklist"`

---

## Verification (whole-plan)

1. Full unit suite green; all four smokes green on live infra.
2. Migration idempotence proven live (second run relabels 0).
3. `grep -rn ":Sentence" src/` → only the four sanctioned sites (MERGE anchor, index DDL, migration CLI, comments).
4. Bundle regression: export a smoke interview and confirm hostile-text link labels and duplicate-surface entities render correctly (covered by unit tests; spot-check in the smoke bundle output if convenient).

## Deferred / out of scope (M4.5a)

- M4.5b (Project aggregate, resolution engine, corrections) and M4.5c (segments) — next plans from the same spec.
- Dropping the `:Sentence` shim label and the deprecated code aliases (backlog, after M4.5).
- Re-targeting vector index DDL to `:Fragment` (rides the shim drop).
