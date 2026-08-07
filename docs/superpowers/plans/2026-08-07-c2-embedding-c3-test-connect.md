# C2 (embedding capability) + C3 (connect every test) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** (C2) make `embed-fragments` its own capability with `pinned-embeddings` as its child; (C3) connect the ~29 "unmapped" tests to what they verify (resolution alias + authored `# verifies:` markers), aiming for zero unmapped.

**Architecture:** C2 is a capability-corpus correction (one new node + one reparent + regen). C3 is a small reader change (`_TARGET_ALIASES`) plus an authored-marker pass across ~25 test files. Independent parts on one branch.

**Tech Stack:** Python 3 (stdlib), pytest, Makefile. No new deps.

**Spec:** `docs/superpowers/specs/2026-08-07-c2-embedding-c3-test-connect-design.md`.

## Global Constraints

- **Non-blocking guards unchanged** — every `*-check` stays exit-0.
- **C2 keeps capabilities-as-intent (ADR-0019):** no capability→capability dependency edge; the embedding→provider dependency stays in the code graph. `implemented_by` targets are documented code-unit slugs.
- **C3 markers are comment-only** — a `# verifies: <domain>:<id>` line added at module level (just after the docstring); NO test logic/imports/assertions touched. Every marker target must resolve to a real node (`graph-check` catches a dangling one) — verify each before adding; do not guess.
- **The infra residual is surfaced, not fabricated** — `test_config` / `test_celery_app` / `test_tasks` have no node to point at; flag them for an owner decision rather than inventing a target.
- **No new ADR** (C2 within ADR-0019, C3 within ADR-0022).
- **Names verbatim:** capability `embed-fragments`; `_TARGET_ALIASES`.

---

### Task 1: C2 — new `embed-fragments` capability + reparent `pinned-embeddings`

**Files:**
- Create: `docs/capabilities/embed-fragments.md`
- Modify: `docs/capabilities/pinned-embeddings.md` (kind + parent)
- Regenerate: `docs/capabilities/index.md`, `docs/graph/index.md`, `docs/graph/graph.md`

**Interfaces:** none (docs + regen).

- [ ] **Step 1: Create the embedding capability**

`docs/capabilities/embed-fragments.md`:

```markdown
---
type: Capability
kind: primary
tier: enabling
category: product
implemented_by: [enrichment, projections]
---
Turn each fragment into a vector embedding so fragments can be found by meaning, not just by keyword.
```

- [ ] **Step 2: Reparent `pinned-embeddings`**

Edit `docs/capabilities/pinned-embeddings.md` frontmatter — change `kind: variant` → `kind: child` and `parent: provider-strategy-and-focused-calls` → `parent: embed-fragments`. Leave `implemented_by: [enrichment]` and the statement unchanged:

```markdown
---
type: Capability
kind: child
parent: embed-fragments
implemented_by: [enrichment]
---
Pin embedding calls to one configured provider/model — never failed over, since vectors from different models aren't comparable.
```

- [ ] **Step 3: Regenerate + verify**

```bash
make capability-index
make graph-index
make capability-check    # clean — embed-fragments classified (primary/enabling/product); pinned-embeddings resolves parent embed-fragments; enrichment/projections claimed
make graph-check         # clean — Capability node embed-fragments; child_of pinned-embeddings->embed-fragments; implements ->enrichment,projections resolve
make code-check          # clean
python -m pytest tests/capability tests/graph -q
```

Expected: capability-check + graph-check clean; `pinned-embeddings` no longer a child of `provider-strategy-and-focused-calls` (which now has only `chat-failover`).

- [ ] **Step 4: Commit**

```bash
git add docs/capabilities/embed-fragments.md docs/capabilities/pinned-embeddings.md \
        docs/capabilities/index.md docs/graph/index.md docs/graph/graph.md
git commit -m "feat(capability): embed-fragments as its own capability; pinned-embeddings its child"
```

---

### Task 2: C3 — `api_surface` resolution alias

**Files:**
- Modify: `tools/testmap/reader.py` (`_TARGET_ALIASES` + `_target`)
- Test: `tests/testmap/test_reader.py` (add a case)

**Interfaces:**
- Produces: `_TARGET_ALIASES: dict[str, str]`; `_target` consults it before returning "".

- [ ] **Step 1: Write the failing test**

Add to `tests/testmap/test_reader.py`:

```python
def test_target_alias_resolves_api_surface(tmp_path):
    # tests/api_surface/* verify tools.api; the segment name doesn't match, so an alias maps it
    t = tmp_path / "tests" / "api_surface"
    t.mkdir(parents=True)
    (t / "test_check.py").write_text("def test_a():\n    pass\n", encoding="utf-8")
    (tmp_path / "tools" / "api").mkdir(parents=True)   # make tools.api a real unit
    tests = {x.slug: x for x in load_tests(str(tmp_path))}
    assert tests["api_surface.test_check"].target == "tools.api"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/testmap/test_reader.py::test_target_alias_resolves_api_surface -v`
Expected: FAIL — target is `""` (no alias yet).

- [ ] **Step 3: Write minimal implementation**

In `tools/testmap/reader.py`, add the alias map near the other module constants:

```python
# dir segments under tests/ whose name doesn't match their code unit (resolved after the
# direct/`tools.`-prefixed attempts). One entry per known mismatch.
_TARGET_ALIASES = {"api_surface": "tools.api"}
```

and extend `_target` to consult it:

```python
def _target(rel: str, units: Set[str]) -> str:
    seg = rel.split(os.sep, 1)[0]
    if seg in units:
        return seg
    if f"tools.{seg}" in units:
        return f"tools.{seg}"
    alias = _TARGET_ALIASES.get(seg)
    if alias and alias in units:
        return alias
    return ""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/testmap/test_reader.py -v`
Expected: PASS (new test + existing).

- [ ] **Step 5: Commit**

```bash
git add tools/testmap/reader.py tests/testmap/test_reader.py
git commit -m "feat(testmap): _TARGET_ALIASES — resolve api_surface tests to tools.api"
```

---

### Task 3: C3 — author `# verifies:` markers on the cross-cutting tests

This is a judgment task (like the use-case corpus derivation): read each unmapped test, confirm what it exercises, and add one honest `# verifies:` marker. Under subagent-driven execution, dispatch it as a marker subagent; the controller regenerates + verifies; the owner reviews the residual.

**Files:**
- Modify (marker-only, one comment line each): the ~19 `tests/integration/*.py` + root tests below
- Regenerate: `docs/tests/index.md`, `docs/graph/index.md`, `docs/graph/graph.md`

**Starting marker map** (VERIFY each by reading the test + confirming the target node exists before adding — every listed code unit was confirmed to exist as a node):

| test | `# verifies:` target |
| --- | --- |
| `integration/test_layer1_projection_smoke` | `code:projections` |
| `integration/test_layer2_enrichment_smoke` | `code:enrichment` |
| `integration/test_layer3_lens_smoke` | `code:lens` |
| `integration/test_layer4_resolution_smoke` | `code:resolution` |
| `integration/test_layer5_export_smoke` | `code:export` |
| `integration/test_ask_smoke` | `use-cases:get-a-grounded-answer-from-my-corpus` |
| `integration/test_anthropic_api_messages` | `code:agents` |
| `integration/test_openai_api_responses` | `code:agents` |
| `integration/test_multi_provider_api_live` | `code:agents` |
| `integration/test_prompt_validation_live` | `code:agents` |
| `integration/test_api_calls` | `code:api` |
| `integration/test_neo4j_connection_reliability` | `code:persistence` |
| `integration/test_neo4j_data_integrity` | `code:persistence` |
| `integration/test_projection_ordering_smoke` | `code:projections` |
| `integration/test_projection_replay` | `code:projections` |
| `integration/test_deployed_projection_smoke` | `code:projections` |
| `integration/test_idempotency` | `code:projections` |
| `integration/test_migrate_shim_drop_live` | `code:projections` |
| `integration/test_live_feed_smoke` | `code:ui` |
| `test_anthropic_agent_response` | `code:agents` |
| `test_openai_agent_response` | `code:agents` |
| `test_prompts` | `code:tools.prompts` |
| `test_config` | **residual** — surface (no config node) |
| `test_celery_app` | **residual** — surface (no task-queue node) |
| `test_tasks` | **residual** — surface (no task-queue node) |

- [ ] **Step 1: Add the markers.** For each test above (except the three residuals), read the file, confirm it genuinely exercises the mapped target end-to-end (adjust the target if the read shows otherwise — e.g. a smoke test that really validates a use-case gets a `use-cases:` marker instead), and insert a module-level `# verifies: <target>` line just after the module docstring. Comment-only; touch nothing else.

- [ ] **Step 2: Handle the residual.** `test_config` / `test_celery_app` / `test_tasks` have no node. Do NOT invent one. Report them to the controller/owner with what each tests, so the owner decides: (a) add a small operations capability / document the module as a code unit and mark to it, or (b) accept a documented residual. Record the decision.

- [ ] **Step 3: Regenerate + verify**

```bash
make testmap-index
make graph-index
make testmap-check       # unmapped count drops from 29 toward the residual (was 4 api_surface [now aliased] + 25 [now marked/residual])
make graph-check         # clean — every verifies marker target resolves (a typo'd/nonexistent target shows here)
make health
python -m pytest tests/testmap tests/graph -q
```

Expected: `graph-check` clean (all markers resolve); `testmap-check` "unmapped" findings reduced to only the consciously-accepted residual; the `verifies` edge count in `docs/graph/index.md` grows by the number of markers added.

- [ ] **Step 4: Commit**

```bash
git add tests/integration/ tests/test_*.py docs/tests/index.md docs/graph/index.md docs/graph/graph.md
git commit -m "feat(testmap): connect cross-cutting tests via # verifies: markers"
```

---

## After all tasks

No ADR. Run the final whole-branch review on the most capable model, then use **superpowers:finishing-a-development-branch**.

## Knowledge-graph check

Reviewed against `docs/index.md` on 2026-08-07.

| domain | touched? | consulted / reconciled | notes |
| --- | --- | --- | --- |
| capabilities | yes (Task 1) | new `embed-fragments`; reparent `pinned-embeddings`; regen | ADR-0019 corpus correction |
| tests | yes (Tasks 2-3) | `_TARGET_ALIASES`; ~22 authored markers; regen | ADR-0022 model |
| graph | yes | regen: new Capability node + moved `child_of` + new `implements`/`verifies` edges; endpoints checked | derived |
| code | yes (read-only) | marker targets + `embed-fragments` implemented_by resolve against code units | possible new node if owner documents config/celery |
| use-cases | yes (read-only) | some markers target use-cases | no node change |
| adr / cli / knowledge / glossary / api / prompts / graph-queries | no | — | unaffected |

**Verdict:** reconciled — capabilities (C2) + tests (C3) subjects; graph regenerated; code/use-cases read-only; infra-test residual surfaced for an owner decision.
