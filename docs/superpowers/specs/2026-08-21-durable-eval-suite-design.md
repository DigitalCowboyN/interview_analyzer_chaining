# Durable agentic-fitness eval suite (design)

**Status:** proposed (brainstorm dialogue with owner, 2026-08-21).
**Program:** the first-class knowledge graph
(`docs/superpowers/specs/2026-08-15-first-class-knowledge-graph-program-design.md`). Turns the lean
3-scenario proof (`evals/graph/`, from PR #44) and the validated agent-driven eval *method* (the
2-eval redo before PR #46) into a **durable, re-runnable suite** that tracks the graph's fitness for
real agentic work over time.

## Purpose

Measure — repeatably — whether an agent can use the knowledge graph to gather the correct, minimal
context for real development tasks, across a broad spread (bug-fix → new component → large refactor,
governance, pipeline, deployment). The suite is a **regression tracker** (catch when a change
degrades usability) *and* a **roadmap** (scenarios the graph can't answer yet are marked and their
scores rise when a future milestone closes the gap).

## Two layers

- **Layer 1 — deterministic context eval (the fast regression core).** For each scenario, run
  `gather_context`/`walk` and score whether the graph *surfaces the gold context*: **recall** (gold
  nodes reached), **minimality** (over-fetch vs gold), and **answer-reach** (the answer node is on
  the path). Pure, fast, deterministic — run it every change. Extends today's `evals/graph/run.py`.
- **Layer 2 — full agentic eval, driven by the Claude Code *subscription* (owner decision,
  2026-08-21; revised — NO API / usage-based billing).** For each scenario an agent-under-test is
  given the generic goal and **only the graph CLI** as a tool, drives its own exploration, and a
  **judge** agent scores its answer + trajectory against a rubric + the reference solution. **Runs
  locally / pre-commit, never CI, never an API key.** See "Layer 2 mechanism" for how the subscription
  drives it.

## Scenario schema

`evals/graph/scenarios/<id>.json`:

```json
{
  "id": "refactor-resolution-engine",
  "category": "bug-fix | new-component | refactor | governance | pipeline | deployment | exploration | meta",
  "level": "symbol | module | subsystem",
  "task": "generic goal — what a human/agent wants, NO prescribed walk steps",
  "entry": ["code:resolution.engine"],
  "gold_context": ["code:resolution.*", "adr:11", "tests:...", "capabilities:resolve-entities-and-people"],
  "gold_answer": "optional — for answer-checkable tasks",
  "expected": "solvable | partial | gap",
  "gap_note": "for partial/gap: what the graph can't reach yet and which milestone would fix it",
  "source": "mined from PR #.. / hand-authored"
}
```

`expected` is the **gap tracker**: a `gap` scenario scoring low is expected (it documents a missing
capability); when a future milestone lands the score rises and the scenario is re-marked `solvable`.
Per eval best-practice, the suite deliberately includes can't-do cases — an all-pass suite is too easy.

## The scenario corpus (v1 — ~16, across the owner's categories)

Reference solutions (`gold_context`/`gold_answer`) are **hand-verified against the live graph** during
the build, and each `entry`/`gold` address must resolve (a test enforces no dangling gold).

| id | category / level | task | expected |
| --- | --- | --- | --- |
| fix-calls-resolution | bug-fix / symbol | fix a bug in how `calls_of` resolves imported calls | solvable |
| fix-speaker-inference | bug-fix / module | fix speaker inference for a messy transcript | solvable |
| add-enrichment-extractor | new-component / module | add a new enrichment extractor (lens dimension) | solvable |
| add-projection-handler | new-component / module | add a projection handler for a new event type | solvable |
| refactor-resolution-engine | refactor / subsystem | refactor the entity-resolution engine | solvable-broad |
| split-export-bundler | refactor / subsystem | split the export bundler into smaller modules | solvable |
| govern-event-envelope | governance | am I violating a decision by changing `events.envelope`? | solvable |
| govern-projection-service | governance | which ADR governs the projection service; still current? | solvable |
| govern-superseded-near-ingestion | governance | is a superseded decision in play near ingestion? | partial |
| pipeline-ingestion-flow | pipeline | trace the ingestion→analysis flow for a transcript | **gap** (runtime order not modeled) |
| pipeline-write-path | pipeline | what happens to a command after submit (event→projection→read)? | **gap** (async seam) |
| deploy-projection-service | deployment | what does the projection-service container need to run? | **gap** (infra not in graph) |
| deploy-service-topology | deployment | which services must be up for the API to serve reads? | **gap** |
| deploy-neo4j-schema-blast | deployment | what breaks downstream if the Neo4j schema changes? | partial |
| explore-ask-subsystem | exploration | what does the ask/retrieval subsystem do, and what serves it? | solvable (control) |
| meta-unverified-capabilities | meta | how many capabilities have no verifying test? | solvable (control) |

The `gap` scenarios (pipeline runtime flow, deployment topology/infra) map exactly onto the deferred
**flow/architecture-nodes** and **infra-modeling** milestones — the suite will show their scores rise
when those land.

## Layer 1 — deterministic runner (`evals/graph/run.py`, extended)

- Loads all scenarios; per scenario runs `gather_context(entry, level)` (and/or `walk` per the
  scenario's needs) and computes: `recall` = |gold ∩ reached| / |gold|; `coverage` = fraction of gold
  **code** nodes reached *with substantive context*; `overfetch` = |reached \ gold|; `answer_reached`
  = the `gold_answer` node (if any) is present.
- Prints a **scorecard**: per scenario + aggregated **by category** and **by `expected`**
  (solvable/partial/gap). Writes `evals/graph/RESULTS.md`. A `gap` scenario is reported as *expected
  low* (not a failure).
- CI-safe and fast (no agents). This is the durable regression number.

## Layer 2 — agentic harness (subscription-driven; NO API, NO usage-based billing)

**Hard constraint (owner):** the LLM is the Claude Code **subscription**, never an API key, never a
usage-metered endpoint. `--bare` is therefore **forbidden** — it forces `ANTHROPIC_API_KEY`. Two
subscription-native mechanisms; the harness prefers Mode A and falls back to Mode B, and **either way
it is a local pre-commit routine, not CI.**

### Mode A — headless `claude -p` on the subscription (if it runs cleanly here)

Per scenario, spawn the agent-under-test as a subprocess that authenticates via the **logged-in
subscription** (no `--bare`, no `ANTHROPIC_API_KEY`):

```bash
claude -p "<scenario.task + node-addressing + the graph CLI is your only tool>" \
  --max-turns 8 \
  --allowedTools "Bash(python -m tools.graph walk:*),Bash(python -m tools.graph context:*)" \
  --append-system-prompt "You are an eval agent. Investigate ONLY via the graph CLI shown; do NOT \
     read source files or repo docs. Start coarse, expand progressively (walk up to intent, then \
     out). If the graph cannot answer, say so explicitly — never infer relevance from proximity. \
     End with a clear final answer." \
  --output-format stream-json --verbose
```

- **Isolation without `--bare`:** the `--append-system-prompt` instructs the agent to ignore repo
  docs/CLAUDE.md, and **`--allowedTools` scoped to the two graph commands physically prevents it from
  reading files** — so actions are graph-only even though CLAUDE.md is nominally in context. That
  tool restriction is the real isolation; the system prompt handles the soft part.
- The harness parses `stream-json` for `tool_use` events (the **trajectory**) and the final `result`
  (the **answer**).
- The **judge** is a second subscription `claude -p` (no `--bare`), given the rubric + the scenario's
  gold + the agent's answer + trajectory, `--output-format json`, returning a JSON verdict.

### Mode B — pre-commit routine driven *inside* a Claude Code session (the robust primary)

If headless `claude -p` does not run cleanly on the subscription in this environment (e.g. nested
invocation is unreliable — observed during the brainstorm), the harness runs **as a routine within a
Claude Code session**, using **subagents** (subscription-backed, already the reliable mechanism this
whole project was built with) as the agent-under-test and the judge:

- A driver (a small skill/script + a checklist the session follows) reads each scenario, dispatches a
  subagent whose prompt is the generic task + the tool-loop protocol (the subagent emits a
  `WALK …`/`CONTEXT …` request; the session executes the graph CLI and returns the result; loop until
  the subagent concludes — the exact validated loop from the pre-PR-46 eval redo). The subagent's
  final message = the answer; the emitted requests = the trajectory.
- A second subagent is the **judge**: given the rubric + gold + answer + trajectory, it returns a
  structured verdict.
- A **pre-commit hook** invokes this routine (or prints the one command to launch it), so the suite
  runs before commits locally.

Mode B needs no headless subprocess and no API key at all — it is the same subscription-backed
subagent mechanism used throughout, wrapped as a repeatable, scenario-driven routine.

### Which mode ships

The build **verifies Mode A locally first** (does subscription `claude -p` run headless here without
an API key?). If yes, Mode A is the automated script. If not, **Mode B ships as the routine** and Mode
A is documented as an optional upgrade. Both produce the same scorecard; neither uses the API.

3. **Aggregate** — verdicts (from whichever mode) join the Layer-1 numbers in the scorecard.

## The judge rubric (`evals/graph/RUBRIC.md`)

Independent LLM-as-judge, one verdict per scenario, with an **escape hatch** and **grade-the-goal-not-
the-path** discipline (per Anthropic's agent-eval guidance). Dimensions, each scored + justified:

- **Answer correctness** — does the agent's answer match the reference (`gold_answer`), or for a
  `gap` scenario, does it *correctly report the graph can't answer* (honesty)?
- **Context sufficiency** — did the trajectory reach the `gold_context` (recall)?
- **Trajectory quality** — did it explore *progressively* (coarse → walk-up-to-intent → horizontal),
  efficiently, without prescribing? (grade the shape, not an exact path.)
- **Honesty** — did it use the escape hatch when appropriate rather than hallucinating governance/
  relevance from proximity?

Overall verdict: `pass | partial | fail` + a one-line rationale. The rubric is versioned and its
wording is fixed so runs are comparable.

## Scope

**This milestone:** the ~16-scenario corpus (schema + hand-verified gold + gap flags); Layer 1
deterministic runner + scorecard by category/expected; Layer 2 **subscription-driven** agentic harness
(agent-under-test + judge — Mode A headless `claude -p` on subscription if it runs here, else Mode B
subagent-driven pre-commit routine) + the rubric; a `RESULTS.md` scorecard; one full local run
recorded as the baseline. Make target(s) + a pre-commit hook entry to run it locally.

**Deferred:** CI integration (deliberately local); any API/usage-based path (forbidden); auto-mining
scenarios from PR history; the flow/architecture-nodes and infra-modeling that would flip the `gap`
scenarios to `solvable` (own milestones); a golden-answer human-calibration pass on the judge.

## Testing

- **Scenario validity:** every `entry`/`gold_context` address resolves on the live graph (no dangling
  gold); each scenario JSON parses and has the required fields + a valid `category`/`expected`.
- **Layer 1 runner:** `score` computes recall/coverage/overfetch correctly on a fixture; on the real
  repo it runs all scenarios and a `solvable` control (e.g. `explore-ask-subsystem`) scores high while
  a `gap` scenario (e.g. `pipeline-write-path`) scores low — both as expected.
- **Layer 2 harness:** exercised on at least one scenario end-to-end (agent → trajectory captured →
  judge → verdict parsed) via whichever mode ships; **no `ANTHROPIC_API_KEY` is set or required** (a
  guard asserts the harness never invokes `--bare` / an API path). Resilient to a failed/empty agent
  run (records a `fail`, never crashes the suite). Full local baseline run recorded.
- **Freshness/regression:** adding `evals/` files doesn't disturb generated indexes beyond the new
  test nodes; full unit suite green.

## ADR

No new ADR — the suite is measurement tooling under the existing program (ADR-0016/0023 govern
non-blocking visibility; the eval *method* is grade-the-goal-not-path per the earlier research). It
does not change the graph model.

## Knowledge-graph check

Reviewed against `docs/index.md` on 2026-08-21.

| domain | touched? | note |
| --- | --- | --- |
| evals (new area) | yes — scenario corpus + Layer-1 runner + Layer-2 subscription-driven harness (headless `claude -p` or subagent routine) + rubric + results | the suite; a measurement harness, not a guarded domain |
| graph / code | yes (read-only) — Layer 1 consumes `gather_context`/`walk`; Layer 2 drives the `walk`/`context` CLI | consumed, not changed |
| capabilities / adr | no (logic) — referenced as scenario gold only | gold references |

**Verdict:** reconciled — a durable, subscription-driven agentic-fitness suite (NO API / usage-based
billing): a deterministic regression core + a local agentic harness (headless subscription `claude -p`,
or a subagent-driven pre-commit routine) over a broad scenario corpus that doubles as a gap roadmap. No
graph-model change, no new ADR; CI integration and the gap-closing milestones are deferred.
